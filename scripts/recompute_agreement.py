"""저장된 부피에서 일치도 지표만 다시 계산함.

    python scripts/recompute_agreement.py [--dry-run] [--case-ids-from-data]

## 왜 필요한가

벤치마크는 홀드아웃 52건의 예측 부피와 정답 부피를 그대로 저장함
(`outputs/benchmark_results.json` 의 `holdout_result.volumes`). 일치도 지표는
그 두 배열만 있으면 전부 다시 나옴. 모델도 영상도 다시 볼 필요가 없음.

지표를 하나 더 보고할 때(예: LoA 폭이 참조 평균의 몇 %인가) 재학습하면 기존 값이 미세하게 달라짐.
계산만 다시 하면 기존 수치는 그대로 재현됨.

## 피험자 단위 부트스트랩

홀드아웃에는 같은 사람의 좌우 해마가 함께 들어 있어, 앙상블 일치도에 피험자 단위 부트스트랩 구간
(`ensemble_agreement.subject_bootstrap`)을 함께 냄. 부피 배열이 어느 케이스인지는 `volumes.case_ids` 에서
읽음. 이 항목이 없는 예전 결과는 `--case-ids-from-data` 로 코드의 홀드아웃 순서를 다시 만들고, 라벨에서 정답
부피를 다시 재어 저장된 정답 부피와 케이스마다 같은지 확인한 뒤에만 씀(원자료 필요, GPU 불필요).

## 무엇을 하지 않는가

예측을 다시 만들지 않고 저장된 예측을 읽기만 함. 기존 값이 달라지면 지표 계산 코드가 바뀌었다는 뜻이므로
화면에 그대로 보여 줌.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULT = ROOT / "outputs" / "benchmark_results.json"


def main() -> int:
    import numpy as np

    from hippo.data.datamodule import subject_key
    from hippo.metrics.volume_agreement import subject_bootstrap_ci, volume_agreement_report

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="비교만 하고 저장하지 않는다")
    ap.add_argument("--result", default=str(RESULT))
    ap.add_argument("--case-ids-from-data", action="store_true",
                    help="volumes.case_ids 가 없을 때 원자료에서 홀드아웃 순서를 다시 만들고 정답 부피로 확인한다")
    args = ap.parse_args()

    path = Path(args.result)
    if not path.exists():
        print(f"  {path} 가 없다. run_benchmark.py 를 먼저 실행할 것.")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    h = data.get("holdout_result")
    if not h or "volumes" not in h:
        print("  저장된 부피가 없다. 이 결과는 부피를 남기기 전 판이다.")
        return 1

    ref = np.asarray(h["volumes"]["reference"], dtype=np.float64)
    ens = np.asarray(h["volumes"]["predicted_ensemble"], dtype=np.float64)
    per = [np.asarray(v, dtype=np.float64) for v in h["volumes"]["predicted_per_ckpt"]]
    print(f"홀드아웃 {len(ref)}건, 체크포인트 {len(per)}개")

    changed = []

    def compare(label: str, old: dict, new: dict, prefix: str = "") -> None:
        for k, v in new.items():
            if isinstance(v, dict):
                compare(label, old.get(k, {}), v, f"{prefix}{k}.")
                continue
            o = old.get(k)
            if o is None:
                changed.append(f"  + {label}.{prefix}{k} = {v}  (새 항목)")
            elif isinstance(v, (list, tuple)) or isinstance(o, (list, tuple)):
                # 구간(LoA CI, 부트스트랩 구간)도 원소마다 봄.
                if not (isinstance(v, (list, tuple)) and isinstance(o, (list, tuple)) and len(v) == len(o)
                        and all(abs(float(a) - float(b)) <= 1e-9 for a, b in zip(v, o, strict=True))):
                    changed.append(f"  ! {label}.{prefix}{k}: {o} -> {v}")
            elif isinstance(v, bool) or isinstance(o, bool):
                if v != o:
                    changed.append(f"  ! {label}.{prefix}{k}: {o} -> {v}")
            elif isinstance(v, float) and isinstance(o, (int, float)):
                if abs(v - o) > 1e-9:
                    changed.append(f"  ! {label}.{prefix}{k}: {o} -> {v}")
        # 새 결과에 없는 옛 키. 통째로 바꿔 쓰므로 여기서 알리지 않으면 경고 없이 사라짐.
        for k in old:
            if k not in new:
                changed.append(f"  - {label}.{prefix}{k}  (빠짐)")

    case_ids = h["volumes"].get("case_ids")
    if case_ids is None:
        if not args.case_ids_from_data:
            print("  volumes.case_ids 가 없다. --case-ids-from-data 로 원자료에서 순서를 확인할 것.")
            return 1
        case_ids = _case_ids_from_data(ref)
        if case_ids is None:
            return 1
        h["volumes"]["case_ids"] = case_ids
    if len(case_ids) != len(ref):
        print(f"  case_ids {len(case_ids)}개와 부피 {len(ref)}개의 수가 다르다.")
        return 1

    new_ens = volume_agreement_report(ens, ref)
    new_ens["subject_bootstrap"] = subject_bootstrap_ci(ens, ref, [subject_key({"image": c}) for c in case_ids])
    compare("ensemble_agreement", h.get("ensemble_agreement", {}), new_ens)
    h["ensemble_agreement"] = new_ens

    # 체크포인트별은 평균, 표준편차로 요약해 둔 형태라 같은 형태로 다시 만듦.
    reports = [volume_agreement_report(p, ref) for p in per]
    icc = np.array([r["icc_2_1"] for r in reports])
    bias = np.array([r["bland_altman"]["bias"] for r in reports])
    # 표준편차 규약을 벤치마크 본체(`hippo/evaluation/benchmark.py`)와 맞춤(np.std 기본값, ddof=0).
    # 여기서만 ddof=1 을 쓰면 재계산만 했는데 값이 바뀐 것처럼 보임.
    # (Bland-Altman 의 sd_diff 는 별개임: LoA 는 표본표준편차로 정의됨.)
    new_per = {
        "icc_2_1": {"mean": float(icc.mean()), "std": float(icc.std())},
        "bias": {"mean": float(bias.mean()), "std": float(bias.std())},
    }
    compare("per_ckpt_agreement", h.get("per_ckpt_agreement", {}), new_per)
    h["per_ckpt_agreement"] = new_per

    if changed:
        print("\n달라진 값")
        for line in changed:
            print(line)
    else:
        print("\n달라진 값 없음")

    print("\n지금 값")
    ba = new_ens["bland_altman"]
    print(f"  ICC(2,1)          {new_ens['icc_2_1']:.4f}")
    print(f"  Pearson r         {new_ens['pearson_r']:.4f}")
    print(f"  참조 평균 부피    {new_ens['reference_mean']:.1f} mm3")
    print(f"  bias              {ba['bias']:+.1f} mm3")
    print(f"  95% LoA           [{ba['loa_lower']:.1f}, {ba['loa_upper']:.1f}]  "
          f"폭 {ba['loa_upper'] - ba['loa_lower']:.1f}")
    print(f"  LoA 반폭          참조 평균의 {new_ens['loa_halfwidth_pct_of_reference']:.1f}%")
    print(f"  비례편향          기울기 {ba['prop_bias_slope']:.3f}  p={ba['prop_bias_p']:.3f}  "
          f"유의 {ba['has_proportional_bias']}")
    sb = new_ens["subject_bootstrap"]
    print(f"  피험자 부트스트랩  {sb['n_subjects']}명 {sb['n_cases']}건, {sb['n_boot']}회, 시드 {sb['seed']}")
    for k in ("icc_2_1", "bias", "loa_lower", "loa_upper", "prop_bias_slope"):
        print(f"    {k:<16} [{sb[k][0]:.4f}, {sb[k][1]:.4f}]")

    if args.dry_run:
        print("\n--dry-run 이라 저장하지 않았다.")
        return 0
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {path}")
    return 0


def _case_ids_from_data(ref) -> list[str] | None:
    """코드의 홀드아웃 순서(benchmark 가 평가한 순서)를 다시 만들고, 라벨의 정답 부피가 저장값과 같은지 봄."""
    import os

    import numpy as np

    from hippo.config import load_config
    from hippo.data.datamodule import HippoDataModule
    from hippo.volumetry.volume import compute_volumes, spacing_from_affine

    os.chdir(ROOT)  # 설정의 data.root(./data)를 저장소 기준으로 품
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    dm = HippoDataModule(cfg).setup_holdout()
    names = [Path(str(d["image"])).name for d in dm.holdout_items]
    vols = []
    for batch in dm.holdout_loader():
        vols.append(compute_volumes(batch["label"].numpy()[0, 0], spacing_from_affine(batch["image"].affine)).total_mm3)
    vols = np.asarray(vols, dtype=np.float64)
    if len(vols) != len(ref) or not np.allclose(vols, ref, rtol=0, atol=1e-6):
        bad = int(np.sum(~np.isclose(vols, ref, rtol=0, atol=1e-6))) if len(vols) == len(ref) else -1
        print(f"  원자료에서 다시 잰 정답 부피가 저장값과 다르다(다른 케이스 {bad}건). 순서를 믿을 수 없어 멈춘다.")
        return None
    print(f"  원자료로 확인: 홀드아웃 {len(names)}건의 정답 부피가 저장값과 케이스마다 같다")
    return names


if __name__ == "__main__":
    raise SystemExit(main())
