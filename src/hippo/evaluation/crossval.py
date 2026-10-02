"""K-fold 교차검증 (구버전: 성능 보고용으로 사용하지 말 것).

DEPRECATED. 낙관 편향이 있는 경로임.

이 모듈은 val Dice 가 최대인 에폭의 체크포인트를 같은 val fold 에서 다시 평가해 보고함(아래 `_validate` 호출).
최고점을 고른 값을 성능으로 내놓는 것이라 winner's curse 가 생김(트러블슈팅 1).

성능 보고는 `hippo.evaluation.benchmark`(`python scripts/run_benchmark.py`)로 함. 이 모듈은 하이퍼파라미터 탐색 등
모델 선택 목적으로만 남겨 둠.
"""
from __future__ import annotations

import numpy as np
import torch

from hippo.config import Config
from hippo.data.datamodule import HippoDataModule
from hippo.inference.predictor import Predictor
from hippo.metrics.volume_agreement import volume_agreement_report
from hippo.models import build_model
from hippo.training.trainer import Trainer
from hippo.volumetry.volume import compute_volumes, spacing_from_affine


@torch.no_grad()
def _collect_volumes(model, loader, cfg, device):
    """val 케이스별 (예측 total 부피, 정답 total 부피) 쌍을 모음.

    Dice 를 잰 것과 같은 후처리를 적용함(benchmark._collect_volumes 와 같게). 같은 결과 파일 안의 Dice 와 부피가
    다른 마스크의 값이 되지 않게 함.
    """
    from hippo.inference.qc import keep_largest_component

    roi = tuple(cfg.model.roi)
    predictor = Predictor(model, roi, device=device, amp=cfg.train.amp)
    postproc = bool(getattr(cfg.train, "postproc_largest_component", False))
    pred_v, ref_v = [], []
    for batch in loader:
        x = batch["image"].to(device)
        spacing = spacing_from_affine(batch["image"].affine)  # 격자에서 유도 (pred/gt 동일 공간)
        pred_mask = predictor.predict_mask(x).cpu().numpy()[0]
        if postproc:
            pred_mask = keep_largest_component(pred_mask)
        gt_mask = batch["label"].numpy()[0, 0]
        pred_v.append(compute_volumes(pred_mask, spacing).total_mm3)
        ref_v.append(compute_volumes(gt_mask, spacing).total_mm3)
    return pred_v, ref_v


def run_cross_validation(cfg: Config, device: torch.device | str, folds: list[int] | None = None) -> dict:
    device = torch.device(device)
    folds = folds if folds is not None else list(range(cfg.data.n_splits))
    per_fold: list[dict] = []
    all_pred, all_ref = [], []

    for fold in folds:
        print(f"\n===== fold {fold} / {cfg.data.n_splits} =====")
        dm = HippoDataModule(cfg).setup_fold(fold)
        model = build_model(cfg.model.name, cfg.model.in_channels,
                            cfg.model.out_channels, tuple(cfg.model.roi))
        trainer = Trainer(cfg, model, device)
        res = trainer.fit(dm.train_loader(), dm.val_loader())

        # best 가중치 로드 후 부피 수집
        model.load_state_dict(torch.load(res.ckpt_path, map_location=device, weights_only=True))
        scores = trainer._validate(dm.val_loader())
        pv, rv = _collect_volumes(model, dm.val_loader(), cfg, device)
        all_pred += pv
        all_ref += rv
        per_fold.append({"fold": fold, "best_dice": res.best_metric, **scores})
        print(f"[fold {fold}] dice={scores['dice']:.4f} hd95={scores['hd95']:.2f}")

    def agg(key):
        vals = [f[key] for f in per_fold]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "per_fold": per_fold,
        "summary": {k: agg(k) for k in ("dice", "hd95", "asd")},
        "volume_agreement": volume_agreement_report(np.array(all_pred), np.array(all_ref)),
        "volumes": {"predicted": all_pred, "reference": all_ref},
    }
