"""절제실험 러너: 설정을 하나씩 바꿔가며 돌리고 그 경위를 기록함.

    python scripts/run_ablation.py --id 1 --name baseline --hypothesis "기준선"
    python scripts/run_ablation.py --id 2 --name lr-3e-4 --lr 3e-4 \\
        --hypothesis "#1 이 100에폭 한도에 걸려 끊김 -> LR 을 낮추면 더 일찍 수렴할 것" \\
        --change "lr 1e-3 -> 3e-4" --parent 1

설정을 한 번에 하나씩 바꾸고 그 변화를 추적함.

## 왜 fold 전체를 도는가

fold 하나만 재면 두 설정의 차이가 설정 때문인지 그 fold 의 운 때문인지 가릴 수 없음. 회차마다 k-fold 전부를
돌려 평균과 표준편차를 함께 내고, 개선 판정은 평균 차이를 fold 표준편차와 견줘서 함(아래 MEANINGFUL).
`--folds 0` 처럼 하나만 줄 수도 있음.

## 개선의 기준선을 먼저 정함

`MEANINGFUL = 0.0039`: 처음 벤치마크에서 잰 fold 간 표준편차임. 평균이 이보다 덜 움직이면 "차이 없음"으로 적음.
결과를 보고 선을 옮기지 않으려고 미리 정한 값이지, 이 선을 넘었다고 통계적으로 유의하다는 뜻은 아님.

## 홀드아웃은 건드리지 않음

`HippoDataModule.setup_fold` 는 개발셋 안에서만 나눔. 홀드아웃 52건은 이 스크립트가 불러오지도 않음.
탐색 결과를 최종 성능으로 보고해서는 안 됨.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ABL_DIR = ROOT / "outputs" / "ablation"

# 개선으로 볼 최소 폭. 기존 벤치마크의 fold 간 표준편차(0.0039)를 그대로 씀.
# 결과를 보고 옮기지 않으려고 실험 전에 박아둔 값임.
MEANINGFUL = 0.0039


# 이어받기 조건에서 뺄 설정. 결과가 아니라 어디에 두느냐만 정함.
KEY_EXCLUDE = {"out_dir", "root"}


def _config_fields(obj) -> dict:
    """설정 객체를 {이름: 값} 으로 폄. dataclass 와 SimpleNamespace 둘 다 받음."""
    out: dict = {}
    for name, val in sorted(vars(obj).items()):
        if name in KEY_EXCLUDE or name.startswith("_"):
            continue
        if hasattr(val, "__dict__"):
            out[name] = _config_fields(val)
        elif isinstance(val, (list, tuple)):
            out[name] = list(val)
        else:
            out[name] = val
    return out


def fold_config_key(cfg) -> str:
    """이어받기를 허용할 조건. 이 값이 다르면 앞선 fold 결과를 쓰지 않음.

    같은 회차 번호로 조건만 바꿔 다시 돌릴 때 옛 fold 를 이어받으면 서로 다른 조건의 결과가 한 회차에 섞임.

    설정 전체를 해시함. 필드를 손으로 고르면 `val_interval`, `grad_accum`, `amp`, `lr_patience`, `lr_factor`,
    `num_workers` 처럼 결과를 바꾸는 값이 빠짐.

    학습에 관여하는 소스 파일의 해시도 함께 넣음. 설정이 같아도 학습 루프, 변환, 모델, 지표 코드가 바뀌면
    옛 fold 는 다른 코드의 결과임.
    """
    return hashlib.sha256(
        json.dumps({"config": _config_fields(cfg), "code": _training_code_hash()},
                   sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


# fold 결과를 바꾸는 소스. 경로는 저장소 루트 기준.
TRAINING_SOURCES = (
    "src/hippo/training/trainer.py", "src/hippo/training/callbacks.py",
    "src/hippo/data/transforms.py", "src/hippo/data/datamodule.py",
    "src/hippo/models/factory.py", "src/hippo/metrics/segmentation.py",
    "src/hippo/inference/predictor.py", "src/hippo/inference/qc.py",
)


def _training_code_hash() -> str:
    h = hashlib.sha256()
    for rel in TRAINING_SOURCES:
        h.update(rel.encode())
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()[:16]


def same_id_other_name(abl_dir: Path, ablation_id: int, name: str) -> Path | None:
    """같은 회차 번호를 다른 이름으로 적어 둔 기록이 있으면 그 경로."""
    want = f"{ablation_id:02d}_{name}.json"
    for f in sorted(abl_dir.glob(f"{ablation_id:02d}_*.json")):
        if f.name != want:
            return f
    return None


def diverged_folds(per_fold: list[dict]) -> list[int]:
    """loss 가 nan 이 된 fold 번호. 한 번이라도 나오면 그 fold 는 발산한 것임."""
    out = []
    for f in per_fold:
        for h in f.get("history") or []:
            loss = h.get("loss")
            if isinstance(loss, float) and loss != loss:   # nan
                out.append(f.get("fold"))
                break
    return out


def diagnose(per_fold: list[dict], val_interval: int) -> str:
    """수렴 여부를 진단해 기록에 남김(에폭 한도에 걸려 끊긴 학습이 산출물에 드러나게, 트러블슈팅 7).

    발산을 맨 앞에서 봄. loss 가 nan 인 fold 를 "정상 수렴"으로 두면 "lr 을 올리면 가끔 발산한다"가
    "lr 을 올리면 나빠진다"로 읽힘.
    """
    bad = diverged_folds(per_fold)
    n_ep = max(f["epochs_run"] for f in per_fold)
    best_ep = max(f["best_epoch"] for f in per_fold)
    stopped_early = all(f["stopped_early"] for f in per_fold)
    if bad:
        return (f"학습 발산: fold {bad} 에서 loss 가 nan "
                f"({len(bad)}/{len(per_fold)} fold)")
    if best_ep <= 2:
        return "학습 실패 의심: best 가 1~2에폭"
    if not stopped_early:
        return "에폭 한도에 걸려 끊김: 수렴 아님"
    if best_ep >= n_ep - val_interval:
        return "마지막까지 개선 중이었음"
    return "정상 수렴(정점 후 하락으로 조기종료)"


FOLD_FIELDS = ("fold", "best_val_dice", "best_epoch", "epochs_run", "stopped_early", "history")


def load_cached_fold(fold_dir: Path, k: int, cfg_key: str) -> dict | None:
    """끝난 fold 결과를 돌려줌. 없거나 조건이 다르면 None."""
    f = fold_dir / f"fold_{k}.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # 쓰다 만 파일. 다시 돌림.
        return None
    if d.get("config_key") != cfg_key or not all(x in d for x in FOLD_FIELDS):
        return None
    return {x: d[x] for x in FOLD_FIELDS}


def save_fold(fold_dir: Path, k: int, rec: dict, cfg_key: str) -> None:
    """fold 가 끝나자마자 남김. 다음 fold 에서 기계가 꺼져도 이건 살아남음."""
    f = fold_dir / f"fold_{k}.json"
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({**rec, "config_key": cfg_key}, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    # ── 기록용 ────────────────────────────────────────────────────────
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--parent", type=int, default=0)
    # ── 바꿔가며 실험할 축 ────────────────────────────────────────────
    ap.add_argument("--arch", default="segresnet",
                    help="탐색은 한 종으로. 승자 설정을 3종에 적용해 확인한다")
    ap.add_argument("--folds", default="0,1,2,3,4",
                    help="쉼표로 구분. 기본은 5-fold 전부. 하나만 주면 예전 동작")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    # 400 인 이유: 조기종료 patience 가 멈출 지점을 정해야 하는데 한도가 먼저 걸리면
    # 수렴 전 상태끼리 비교하게 됨(트러블슈팅 7번).
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--roi", default="", help="예: 64,64,64")
    ap.add_argument("--crop-samples", type=int, default=None)
    ap.add_argument("--aug", default=None,
                    help="증강 단계. transforms.AUG_LEVELS 참고")
    # 후처리는 켜기와 끄기를 둘 다 명시할 수 있어야 함. 켜는 플래그만 있으면 플래그를 뺀 회차는
    # configs/default.yaml 의 값을 물려받아, 기준선이 후처리 실험과 같은 설정이 될 수 있음.
    ap.add_argument("--postproc", dest="postproc", action="store_true", default=None,
                    help="평가에 최대 연결성분 후처리를 적용한다")
    ap.add_argument("--no-postproc", dest="postproc", action="store_false",
                    help="후처리를 끈다 (yaml 기본값이 켜져 있어도)")
    # 초기값, 증강 난수 시드. fold 분할은 data.fold_seed 로 고정돼 바뀌지 않음.
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch

    from hippo.config import load_config
    from hippo.data.datamodule import HippoDataModule
    from hippo.models import build_model
    from hippo.training.trainer import Trainer

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    cfg.model.name = args.arch
    cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.batch is not None:
        cfg.train.batch = args.batch
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay
    if args.patience is not None:
        cfg.train.early_stop_patience = args.patience
    if args.roi:
        cfg.model.roi = tuple(int(x) for x in args.roi.split(","))
    if args.crop_samples is not None:
        cfg.data.crop_samples = args.crop_samples
    if args.aug is not None:
        cfg.data.aug = args.aug
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.postproc is None:
        warnings.warn(
            f"--postproc/--no-postproc 를 주지 않아 yaml 기본값 "
            f"({cfg.train.postproc_largest_component})을 쓴다. 기본값은 승자 반영으로 "
            "바뀔 수 있으므로 절제실험에서는 명시하는 편이 안전하다", stacklevel=1)
    else:
        cfg.train.postproc_largest_component = bool(args.postproc)
    cfg.out_dir = str(ROOT / "outputs" / "ablation_runs" / f"{args.id:02d}_{args.name}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("=" * 74)
    print(f"  절제실험 #{args.id}: {args.name}")
    print("=" * 74)
    if args.hypothesis:
        print(f"  가설    : {args.hypothesis}")
    if args.change:
        print(f"  바꾼 것 : {args.change}")
    if args.parent:
        print(f"  근거    : 실험 #{args.parent} 결과")
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    print(f"  설정    : arch={cfg.model.name} folds={folds} lr={cfg.train.lr:g} "
          f"batch={cfg.train.batch} wd={cfg.train.weight_decay:g} "
          f"roi={tuple(cfg.model.roi)} crop_samples={cfg.data.crop_samples} "
          f"aug={cfg.data.aug} postproc={cfg.train.postproc_largest_component} "
          f"epochs={cfg.train.epochs} patience={cfg.train.early_stop_patience}")
    if args.check:
        print("\n준비 완료.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # fold 단위 이어받기. 크래시로 중간에 끊겨도 끝낸 fold 는 다시 돌리지 않음.
    #
    # fold 마다 Trainer.fit() 이 set_determinism(train.seed) 로 난수를 다시 고정하므로
    # fold 는 서로 독립임. 이어 돌린 결과와 처음부터 돌린 결과가 같음.
    # 설정이 다르면(해시 불일치) 남은 파일을 무시하고 처음부터 감.
    fold_dir = Path(cfg.out_dir) / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    cfg_key = fold_config_key(cfg)

    per_fold = []
    for k in folds:
        done = load_cached_fold(fold_dir, k, cfg_key)
        if done is not None:
            per_fold.append(done)
            print(f"    fold {k}: 이어받음 dice {done['best_val_dice']:.4f} "
                  f"@ {done['best_epoch']}에폭 (앞선 실행에서 끝난 fold)")
            continue
        # fold 마다 모델을 새로 만듦. 앞 fold 의 가중치가 남으면 뒤 fold 는
        # 그 fold 의 val 을 이미 학습한 셈이 됨.
        dm = HippoDataModule(cfg).setup_fold(k)
        model = build_model(cfg.model.name, cfg.model.in_channels,
                            cfg.model.out_channels, tuple(cfg.model.roi))
        trainer = Trainer(cfg, model, device)
        r = trainer.fit(dm.train_loader(), dm.val_loader())
        rec = {
            "fold": k,
            "best_val_dice": float(r.best_metric),
            "best_epoch": r.best_epoch,
            "epochs_run": len(r.history),
            "stopped_early": len(r.history) < cfg.train.epochs,
            "history": r.history,
        }
        per_fold.append(rec)
        save_fold(fold_dir, k, rec, cfg_key)
        print(f"    fold {k}: dice {r.best_metric:.4f} @ {r.best_epoch}에폭 "
              f"(실행 {len(r.history)}/{cfg.train.epochs})")
    elapsed = time.time() - t0

    scores = [f["best_val_dice"] for f in per_fold]
    mean = statistics.fmean(scores)
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    n_ep = max(f["epochs_run"] for f in per_fold)
    best_ep = max(f["best_epoch"] for f in per_fold)
    stopped_early = all(f["stopped_early"] for f in per_fold)
    diag = diagnose(per_fold, cfg.train.val_interval)

    print(f"\n  val dice {mean:.4f} +/- {sd:.4f}  "
          f"(fold {len(scores)}개: {', '.join(f'{x:.4f}' for x in scores)})")
    print(f"  진단: {diag}")

    ABL_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "id": args.id, "name": args.name, "parent": args.parent,
        "hypothesis": args.hypothesis, "change": args.change,
        "config": {
            "arch": cfg.model.name, "folds": folds, "lr": cfg.train.lr,
            "batch": cfg.train.batch, "weight_decay": cfg.train.weight_decay,
            "roi": list(cfg.model.roi), "crop_samples": cfg.data.crop_samples,
            "aug": cfg.data.aug,
            "epochs": cfg.train.epochs,
            "patience": cfg.train.early_stop_patience, "seed": cfg.train.seed,
            "fold_seed": cfg.data.fold_seed,
            "postproc_largest_component": cfg.train.postproc_largest_component,
        },
        "result": {
            # 회차 간 비교는 이 두 값으로 함. fold 하나짜리 숫자를 쓰면
            # 설정 효과와 fold 운을 못 가름.
            "mean_val_dice": mean,
            "std_val_dice": sd,
            "n_folds": len(scores),
            "fold_scores": scores,
            "best_epoch": best_ep, "epochs_run": n_ep,
            "stopped_early": stopped_early,
        },
        "meaningful_threshold": MEANINGFUL,
        # 판정선이 낡았는지 확인할 수 있게 이번 회차의 실측 흩어짐을 같이 남김.
        # MEANINGFUL 은 사전등록 값이라 결과를 보고 옮기지 않음. 대신 실측과
        # 크게 벌어지면 그 사실이 기록에 남아야 다음 사전등록 때 고칠 수 있음.
        "threshold_vs_measured": {
            "preregistered": MEANINGFUL,
            "measured_fold_std": sd,
            "ratio": round(sd / MEANINGFUL, 2) if MEANINGFUL else None,
        },
        "diagnosis": diag,
        "per_fold": per_fold,
        "elapsed_sec": round(elapsed, 1),
        "note": (f"개발셋 {len(scores)}개 fold 의 val 만 잰 값이다. 홀드아웃은 "
                 "불러오지 않았다. 최종 성능은 승자 설정으로 "
                 "scripts/run_benchmark.py 를 돌려 보고한다. 회차 간 비교는 "
                 f"평균 차이가 {MEANINGFUL} 이상일 때만 개선으로 적는다."),
    }
    path = ABL_DIR / f"{args.id:02d}_{args.name}.json"
    clash = same_id_other_name(ABL_DIR, args.id, args.name)
    if clash:
        print(f"\n[중단] 회차 #{args.id} 기록이 다른 이름으로 이미 있다: {clash.name}")
        print(f"        지금 쓰려는 이름: {path.name}")
        print("        둘 다 남으면 실험일지에 같은 번호가 두 번 실린다. 옛 기록을")
        print("        지우거나 다른 번호를 쓸 것.")
        return 1
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")
    if MEANINGFUL and not (0.5 <= sd / MEANINGFUL <= 2.0):
        print(f"  [주의] 판정선 {MEANINGFUL} 과 이번 회차 fold 흩어짐 {sd:.4f} 이 "
              f"{sd / MEANINGFUL:.1f}배 차이난다. 사전등록 값이 낡았을 수 있다: "
              "이번 실험에서는 그대로 쓰고, 다음 사전등록 때 다시 정할 것.")
    print(f"  기록: {path}  ({elapsed / 60:.1f}분)")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
