"""아키텍처 벤치마크 실행 진입점.

    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --archs unet,segresnet --epochs 60
    python scripts/run_benchmark.py --holdout-only   # 학습 없이 저장된 fold 체크포인트로 홀드아웃만 다시 평가

출력은 콘솔과 로그 파일에 동시에 기록됨(실시간 확인 + 사후 추적).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class Tee:
    """stdout 과 로그파일에 동시 출력. 진행상황을 실시간으로 보기 위해 매번 flush."""

    def __init__(self, path: Path):
        self.file = open(path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, data: str) -> int:
        self.stdout.write(data)
        self.stdout.flush()
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def isatty(self) -> bool:
        return False  # tqdm 이 캐리지리턴 도배를 하지 않도록


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--archs", default="unet,segresnet,swinunetr")
    ap.add_argument("--folds", default="", help="비우면 전체 fold. 예: 0,1")
    ap.add_argument("--epochs", type=int, default=None, help="config 값을 덮어씀")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "benchmark_results.json"))
    ap.add_argument("--log", default=None)
    # 평가 코드만 고쳤을 때 재학습 없이 홀드아웃 평가만 다시 함. 선택된 아키텍처와
    # fold 체크포인트는 --out 의 기존 결과에서 읽고, holdout_result 만 바꿔 씀. CV 결과(모델 선택)는
    # 그대로이므로 홀드아웃이 선택에 관여하지 않는다는 원칙이 유지됨.
    ap.add_argument("--holdout-only", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log) if args.log else ROOT / "outputs" / f"benchmark_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

    import warnings

    warnings.filterwarnings("ignore")
    import torch

    from hippo.config import load_config
    from hippo.evaluation.benchmark import run_benchmark

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    folds = [int(f) for f in args.folds.split(",") if f.strip()] or None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()

    print("=" * 74)
    print("  아키텍처 벤치마크: 홀드아웃 분리 + 3종 비교")
    print("=" * 74)
    print(f"  시작       : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  로그       : {log_path}")
    print(f"  device     : {device} "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else '-'})")
    print(f"  아키텍처   : {archs}")
    print(f"  roi        : {tuple(cfg.model.roi)}   batch={cfg.train.batch}   "
          f"epochs={cfg.train.epochs}")
    print(f"  홀드아웃   : {cfg.data.holdout_frac:.0%} (seed={cfg.data.holdout_seed}, "
          f"체크포인트 선택에 미사용)")
    print(f"  CV         : {cfg.data.n_splits}-fold on 개발셋")
    print("=" * 74)

    try:
        if args.holdout_only:
            result = rerun_holdout(cfg, device, Path(args.out))
        else:
            result = run_benchmark(cfg, device, archs=archs, folds=folds, out_json=args.out)
    except Exception:
        import traceback

        print("\n[실패] 예외 발생")
        traceback.print_exc()
        return 1

    elapsed = time.time() - started
    print("\n" + "=" * 74)
    print("  결과 요약")
    print("=" * 74)
    print(f"  {'아키텍처':<12} {'CV Dice':>16} {'HD95':>10} {'ASD':>10}")
    print("  " + "-" * 52)
    for arch, r in result["cv_results"].items():
        s = r["cv_summary"]
        mark = "  <= 선택" if arch == result["selected_arch"] else ""
        print(f"  {arch:<12} {s['dice']['mean']:.4f} +- {s['dice']['std']:.4f}"
              f" {s['hd95']['mean']:>9.2f} {s['asd']['mean']:>9.3f}{mark}")

    h = result["holdout_result"]
    hs = h["holdout_summary"]
    ens = h["ensemble_agreement"]
    eba = ens["bland_altman"]
    pca = h["per_ckpt_agreement"]
    print()
    print(f"  홀드아웃 최종 평가 ({h['arch']}, n={h['n_holdout']}): 모델 선택에 미사용")
    print(f"    Dice        {hs['dice']['mean']:.4f} +- {hs['dice']['std']:.4f}"
          f"   (std 는 체크포인트 {h['n_checkpoints']}개 간 변동)")
    print(f"    HD95        {hs['hd95']['mean']:.2f}")
    print(f"    ASD         {hs['asd']['mean']:.3f}")
    print()
    print(f"    부피 일치도: 앙상블 단일 예측 기준 (n={ens['n']}, 케이스 독립 식)")
    print(f"      ICC(2,1)  {ens['icc_2_1']:.4f}")
    print(f"      Pearson r {ens['pearson_r']:.4f}")
    print(f"      bias      {eba['bias']:+.1f} mm3")
    print(f"      95% LoA   [{eba['loa_lower']:.1f}, {eba['loa_upper']:.1f}]"
          f"  (폭 {eba['loa_upper'] - eba['loa_lower']:.1f})")
    print(f"      평균절대오차 {eba['mean_pct_error']:.2f}%")
    print(f"    참고: 체크포인트별 ICC {pca['icc_2_1']['mean']:.4f} "
          f"+- {pca['icc_2_1']['std']:.4f}")
    sb = ens.get("subject_bootstrap")
    if sb:
        print(f"    피험자 단위 부트스트랩 95% 구간 ({sb['n_subjects']}명 {sb['n_cases']}건, {sb['n_boot']}회): "
              f"ICC [{sb['icc_2_1'][0]:.3f}, {sb['icc_2_1'][1]:.3f}], "
              f"bias [{sb['bias'][0]:.1f}, {sb['bias'][1]:.1f}]")
    print()
    print(f"  총 소요 {elapsed/3600:.2f}시간   종료 {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  결과 JSON: {args.out}")
    print("=" * 74)
    return 0


def rerun_holdout(cfg, device: str, out: Path) -> dict:
    """기존 결과의 선택 아키텍처와 fold 체크포인트로 홀드아웃 평가만 다시 함."""
    import json

    from hippo.evaluation.benchmark import evaluate_on_holdout

    result = json.loads(out.read_text(encoding="utf-8"))
    arch = result["selected_arch"]
    ckpts = [str(ROOT / c) if not Path(c).is_absolute() else c
             for c in result["cv_results"][arch]["ckpts"]]
    missing = [c for c in ckpts if not Path(c).is_file()]
    if missing:
        raise FileNotFoundError(f"fold 체크포인트가 없다: {missing}")
    print(f"  홀드아웃만 다시 평가: {arch}, 체크포인트 {len(ckpts)}개 (CV 결과와 모델 선택은 그대로)")
    result["holdout_result"] = evaluate_on_holdout(cfg, arch, ckpts, device)
    result["holdout_rerun"] = {
        "at": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "why": "부피 계산에 Dice 와 같은 후처리를 걸도록 평가 코드를 고쳐 홀드아웃만 다시 평가",
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"[저장] {out}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
