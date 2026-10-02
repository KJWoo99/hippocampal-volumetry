"""아키텍처 벤치마크: 홀드아웃 분리 + 3종 비교.

절차
----
1. 260개를 개발셋 208 / 홀드아웃 52 로 나눔(고정 시드, 아키텍처 무관 동일).
2. 개발셋 208 로 아키텍처별 5-fold CV 를 돌림.
   각 fold 의 체크포인트는 그 fold 의 val Dice 로 선택함.
3. CV 평균 Dice 가 가장 높은 아키텍처를 최종 모델로 선택함.
4. 선택된 아키텍처의 각 fold 체크포인트를 홀드아웃 52 에서 딱 한 번 평가함.

val fold 로 고른 체크포인트를 같은 val fold 에서 평가하면 낙관 편향(winner's curse)이 생김.
홀드아웃은 체크포인트와 아키텍처 선택에 일절 관여하지 않으므로 이 편향이 없음.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from hippo.config import Config
from hippo.data.datamodule import HippoDataModule, subject_key
from hippo.inference.predictor import Predictor
from hippo.metrics.volume_agreement import subject_bootstrap_ci, volume_agreement_report
from hippo.models import build_model
from hippo.training.trainer import Trainer
from hippo.volumetry.volume import compute_volumes, spacing_from_affine


@torch.no_grad()
def _collect_volumes(model, loader, cfg: Config, device) -> tuple[list[float], list[float]]:
    """케이스별 (예측 부피, 정답 부피) 쌍.

    예측 마스크에는 Dice 를 잰 것과 같은 후처리를 적용함(cfg.train.postproc_largest_component).
    Dice 와 부피 일치도가 같은 마스크에서 나와야 하고, `hippo predict` 가 내놓는 마스크와도 같아야 함(트러블슈팅 12).
    """
    from hippo.inference.qc import keep_largest_component

    predictor = Predictor(model, tuple(cfg.model.roi), device=device, amp=cfg.train.amp)
    postproc = bool(getattr(cfg.train, "postproc_largest_component", False))
    pred_v, ref_v = [], []
    for batch in loader:
        x = batch["image"].to(device)
        spacing = spacing_from_affine(batch["image"].affine)
        pred_mask = predictor.predict_mask(x).cpu().numpy()[0]
        if postproc:
            pred_mask = keep_largest_component(pred_mask)
        gt_mask = batch["label"].numpy()[0, 0]
        pred_v.append(compute_volumes(pred_mask, spacing).total_mm3)
        ref_v.append(compute_volumes(gt_mask, spacing).total_mm3)
    return pred_v, ref_v


def run_architecture_cv(
    cfg: Config, arch: str, device: torch.device | str, folds: list[int] | None = None
) -> dict:
    """단일 아키텍처의 개발셋 5-fold CV. 홀드아웃은 건드리지 않음."""
    device = torch.device(device)
    folds = folds if folds is not None else list(range(cfg.data.n_splits))
    per_fold: list[dict] = []
    ckpts: list[str] = []

    for fold in folds:
        # 체크포인트 경로를 (아키텍처, fold) 별로 분리함.
        # 공통 경로를 쓰면 3종 x 5fold 가 서로 덮어써 비교가 불가능해짐.
        # deepcopy 여야 함. 얕은 복사는 중첩 dataclass(model/data/train)를 공유해
        # fold_cfg.model.name 대입이 원본 cfg 까지 오염시킴.
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.out_dir = str(Path(cfg.out_dir) / "benchmark" / arch / f"fold{fold}")
        fold_cfg.model.name = arch
        os.makedirs(fold_cfg.out_dir, exist_ok=True)

        print(f"\n===== [{arch}] fold {fold}/{cfg.data.n_splits} =====")
        dm = HippoDataModule(fold_cfg).setup_fold(fold)
        model = build_model(arch, fold_cfg.model.in_channels,
                            fold_cfg.model.out_channels, tuple(fold_cfg.model.roi))
        trainer = Trainer(fold_cfg, model, device)
        res = trainer.fit(dm.train_loader(), dm.val_loader())

        model.load_state_dict(torch.load(res.ckpt_path, map_location=device, weights_only=True))
        scores = trainer._validate(dm.val_loader())
        per_fold.append({
            "fold": fold, "arch": arch,
            "best_epoch": res.best_epoch,
            "val_dice_selected": res.best_metric,  # 체크포인트 선택에 쓴 값(낙관 편향 있음)
            **scores,
        })
        ckpts.append(res.ckpt_path)
        print(f"[{arch} fold {fold}] val dice={scores['dice']:.4f} hd95={scores['hd95']:.2f}")

        del model, trainer, dm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def agg(key: str) -> dict:
        vals = [f[key] for f in per_fold]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "arch": arch,
        "per_fold": per_fold,
        "cv_summary": {k: agg(k) for k in ("dice", "hd95", "asd")},
        "ckpts": ckpts,
    }


def evaluate_on_holdout(
    cfg: Config, arch: str, ckpt_paths: list[str], device: torch.device | str
) -> dict:
    """홀드아웃 최종 평가. 모델 선택이 끝난 뒤 딱 한 번만 호출함.

    부피 일치도(ICC/Bland-Altman)를 계산할 때의 주의
    ------------------------------------------------
    fold 체크포인트 5개를 같은 홀드아웃 52건에 각각 적용하면 예측이 260개 나옴.
    이것을 한 배열에 합쳐 ICC 를 계산하면 같은 개체가 5번 들어가 독립성 가정이
    깨짐. n 이 5배로 부풀려져 신뢰도가 과대평가되고, 동일한 정답 부피가 반복되어
    개체 간 분산도 왜곡됨. ICC(2,1) 은 독립 개체를 전제하는 지표임.

    그래서 두 가지를 따로 보고함.
      1) per_ckpt_agreement : 체크포인트별로 각각 n=52 로 계산한 뒤 mean +/- std
      2) ensemble_agreement : 5개 예측을 케이스별로 평균 낸 단일 예측 으로 n=52 계산
                              (실제 배포 시 앙상블을 쓴다면 이 값이 최종 성능임)
    """
    device = torch.device(device)
    hcfg = copy.deepcopy(cfg)
    hcfg.model.name = arch
    dm = HippoDataModule(hcfg).setup_holdout()
    loader = dm.holdout_loader()

    per_ckpt: list[dict] = []
    pred_matrix: list[list[float]] = []   # [n_ckpt, n_case]
    ref_vector: list[float] | None = None

    for i, ckpt in enumerate(ckpt_paths):
        model = build_model(arch, hcfg.model.in_channels,
                            hcfg.model.out_channels, tuple(hcfg.model.roi))
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        trainer = Trainer(hcfg, model, device)
        scores = trainer._validate(loader)
        pv, rv = _collect_volumes(model, loader, hcfg, device)

        # 정답 부피는 체크포인트와 무관하므로 한 번만 잡아두고 이후 일치를 확인함.
        if ref_vector is None:
            ref_vector = rv
        elif not np.allclose(ref_vector, rv):
            raise RuntimeError("체크포인트마다 정답 부피가 다릅니다: 로더 순서가 불안정합니다")

        pred_matrix.append(pv)
        agree_i = volume_agreement_report(np.array(pv), np.array(rv))  # n=52, 독립
        per_ckpt.append({"fold": i, **scores,
                         "icc_2_1": agree_i["icc_2_1"],
                         "bias": agree_i["bland_altman"]["bias"]})
        print(f"[{arch}] holdout (fold{i} ckpt): dice={scores['dice']:.4f} "
              f"icc={agree_i['icc_2_1']:.4f}")
        del model, trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pred_arr = np.array(pred_matrix)            # [n_ckpt, n_case]
    ref_arr = np.array(ref_vector)              # [n_case]
    ensemble_pred = pred_arr.mean(axis=0)       # 케이스별 평균 -> 단일 예측
    ens_agree = volume_agreement_report(ensemble_pred, ref_arr)
    # 홀드아웃에는 같은 사람의 좌우가 함께 들어 있어, 케이스 독립 식과 함께 피험자 단위 부트스트랩 구간을 냄
    ens_agree["subject_bootstrap"] = subject_bootstrap_ci(
        ensemble_pred, ref_arr, [subject_key(d) for d in dm.holdout_items])

    def agg(key: str) -> dict:
        vals = [c[key] for c in per_ckpt]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "arch": arch,
        "n_holdout": int(len(ref_arr)),
        "n_checkpoints": len(ckpt_paths),
        # Dice 와 부피를 잰 마스크에 건 후처리. 둘은 같은 마스크여야 함.
        "postproc_largest_component": bool(getattr(hcfg.train, "postproc_largest_component", False)),
        "per_ckpt": per_ckpt,
        "holdout_summary": {k: agg(k) for k in ("dice", "hd95", "asd")},
        # 체크포인트별 n=52 계산 결과의 분포 (독립성 유지)
        "per_ckpt_agreement": {k: agg(k) for k in ("icc_2_1", "bias")},
        # 앙상블 단일 예측 기준 (n=52, 독립): 배포 관점의 최종 수치
        "ensemble_agreement": ens_agree,
        "volumes": {"predicted_ensemble": ensemble_pred.tolist(),
                    "predicted_per_ckpt": pred_arr.tolist(),
                    "reference": ref_arr.tolist(),
                    # 배열과 같은 순서의 케이스 이름. 피험자 단위 부트스트랩(recompute_agreement.py)이 씀
                    "case_ids": [Path(str(d["image"])).name for d in dm.holdout_items]},
    }


def run_benchmark(
    cfg: Config,
    device: torch.device | str,
    archs: list[str] | None = None,
    folds: list[int] | None = None,
    out_json: str | None = None,
) -> dict:
    """전체 파이프라인: 아키텍처별 CV -> 최고 성능 선택 -> 홀드아웃 1회 평가."""
    archs = archs or ["unet", "segresnet", "swinunetr"]
    cv_results = {a: run_architecture_cv(cfg, a, device, folds) for a in archs}

    # 아키텍처 선택은 개발셋 CV 결과만 보고 결정함.
    best_arch = max(archs, key=lambda a: cv_results[a]["cv_summary"]["dice"]["mean"])
    print(f"\n>>> 개발셋 CV 기준 선택된 아키텍처: {best_arch}")

    holdout = evaluate_on_holdout(cfg, best_arch, cv_results[best_arch]["ckpts"], device)

    result = {
        "roi": list(cfg.model.roi),
        "n_splits": cfg.data.n_splits,
        "holdout_frac": cfg.data.holdout_frac,
        "holdout_seed": cfg.data.holdout_seed,
        "cv_results": cv_results,
        "selected_arch": best_arch,
        "holdout_result": holdout,
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[저장] {out_json}")
    return result
