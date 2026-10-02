"""3D 분할 학습 루프.

AMP, Gradient Accumulation, EarlyStopping, ReduceLROnPlateau 를 씀.
검증은 sliding-window 추론 + Dice/HD95/ASD. 결정성(seed) 고정으로 재현 가능.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from hippo.config import Config
from hippo.data.transforms import post_transforms
from hippo.inference.predictor import Predictor
from hippo.metrics.segmentation import SegMetrics
from hippo.training.callbacks import EarlyStopping, ModelCheckpoint


@dataclass
class TrainResult:
    best_metric: float
    best_epoch: int
    ckpt_path: str
    history: list[dict] = field(default_factory=list)


class Trainer:
    def __init__(self, cfg: Config, model: torch.nn.Module, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.num_classes = cfg.model.out_channels
        self.roi = tuple(cfg.model.roi)

        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.optim = torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode="max", factor=cfg.train.lr_factor, patience=cfg.train.lr_patience
        )
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        self.post_pred, self.post_label = post_transforms(self.num_classes)
        self.metrics = SegMetrics()
        # 평가 때 최대 연결성분만 남길지. 기본값은 켬(config.py). 설정 객체에 이 속성이 아예 없을 때만 끔.
        # 켜면 보고하는 Dice 가 `hippo predict` 가 내놓는 마스크와 같은 후처리를 거침(절제 #1 과 #13 에서 비교).
        self.postproc = bool(getattr(cfg.train, "postproc_largest_component", False))

        os.makedirs(cfg.out_dir, exist_ok=True)
        self.ckpt = ModelCheckpoint(os.path.join(cfg.out_dir, "best_model.pth"), mode="max")
        self.early = EarlyStopping(patience=cfg.train.early_stop_patience, mode="max")

    def _train_epoch(self, loader) -> float:
        self.model.train()
        accum = max(1, self.cfg.train.grad_accum)
        total, nstep = 0.0, 0
        self.optim.zero_grad(set_to_none=True)
        pending = False   # 아직 step 되지 않은 누적 기울기가 있는가
        for i, batch in enumerate(loader):
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                loss = self.loss_fn(self.model(x), y) / accum
            self.scaler.scale(loss).backward()
            pending = True
            if (i + 1) % accum == 0:
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
                pending = False
            total += loss.item() * accum
            nstep += 1

        # 배치 수가 accum 의 배수가 아니면 마지막 묶음이 step 되지 않은 채 남고, 다음 에폭의 zero_grad 가
        # 지워 버림(accum=4, 배치 10개면 20%). 남은 묶음을 여기서 step 함.
        if pending:
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad(set_to_none=True)

        return total / max(1, nstep)

    def _keep_largest(self, onehot: torch.Tensor) -> torch.Tensor:
        """원-핫 예측에 최대 연결성분 후처리를 적용함.

        `post_pred` 가 (C, ...) 원-핫을 내놓으므로 라벨 인덱스로 되돌려 후처리하고
        다시 원-핫으로 만듦. `hippo predict` 가 쓰는 함수와 같은 것을 씀:
        평가와 배포가 다른 연산을 하면 애초에 이 실험의 의미가 없음.
        """
        from hippo.inference.qc import keep_largest_component

        lab = onehot.argmax(dim=0).cpu().numpy()
        cleaned = keep_largest_component(lab, labels=tuple(range(1, self.num_classes)))
        out = torch.zeros_like(onehot)
        t = torch.as_tensor(cleaned, device=onehot.device)
        for c in range(self.num_classes):
            out[c] = (t == c)
        return out

    @torch.no_grad()
    def _validate(self, loader) -> dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        predictor = Predictor(self.model, self.roi, sw_batch_size=4,
                              device=self.device, amp=self.use_amp)
        for batch in loader:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)
            logits = predictor.logits(x)
            preds = [self.post_pred(o) for o in decollate_batch(logits)]
            labels = [self.post_label(o) for o in decollate_batch(y)]
            if self.postproc:
                preds = [self._keep_largest(o) for o in preds]
            self.metrics.update(preds, labels)
        return self.metrics.aggregate()

    def fit(self, train_loader, val_loader) -> TrainResult:
        set_determinism(seed=self.cfg.train.seed)
        best_metric, best_epoch = -1.0, -1
        history: list[dict] = []

        for epoch in range(1, self.cfg.train.epochs + 1):
            loss = self._train_epoch(train_loader)
            row = {"epoch": epoch, "loss": loss}

            if epoch % self.cfg.train.val_interval == 0 or epoch == self.cfg.train.epochs:
                scores = self._validate(val_loader)
                row.update(scores)
                dice = scores["dice"]
                self.scheduler.step(dice)
                if self.ckpt.maybe_save(dice, self.model):
                    row["saved"] = True
                    best_metric, best_epoch = dice, epoch
                self.early.step(dice)
                print(
                    f"epoch {epoch:3d}/{self.cfg.train.epochs}  loss={loss:.4f}  "
                    f"dice={dice:.4f} hd95={scores['hd95']:.2f} asd={scores['asd']:.2f}"
                    + ("  *best" if row.get("saved") else "")
                )
            else:
                print(f"epoch {epoch:3d}/{self.cfg.train.epochs}  loss={loss:.4f}")

            history.append(row)
            if self.early.should_stop:
                print(f"[early-stop] {self.early.patience} epoch 개선 없음 -> 중단")
                break

        return TrainResult(best_metric, best_epoch, self.ckpt.path, history)
