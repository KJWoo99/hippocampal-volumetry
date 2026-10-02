"""홀드아웃과 개발셋 5-fold 의 실제 케이스 목록을 outputs/splits.json 에 남김.

    python scripts/export_splits.py

분할은 코드가 실행 때마다 다시 계산함(get_datalist -> holdout_split -> kfold_datalist). 그런데
DecathlonDataset 이 목록을 먼저 seed 0 으로 섞어 돌려주므로, 문서의 "20%, seed 1234" 만 보고 다시
짜면 다른 52건이 나옴(트러블슈팅 13). 누가 어떤 케이스로 평가했는지 파일로 남겨,
다른 도구로 재현하거나 대조할 때 이 목록을 기준으로 쓰게 함. 케이스 이름은 MSD Task04 공개 데이터의
파일 이름임.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hippo.config import load_config  # noqa: E402
from hippo.data.datamodule import get_datalist, holdout_split, kfold_datalist  # noqa: E402

OUT = ROOT / "outputs" / "splits.json"


def _names(items: list[dict]) -> list[str]:
    return [Path(d["image"]).name for d in items]


def build(cfg) -> dict:
    full = get_datalist(cfg, download=False)
    dev, hold = holdout_split(full, cfg.data.holdout_frac, cfg.data.holdout_seed)
    folds = []
    for k in range(cfg.data.n_splits):
        _train, val = kfold_datalist(dev, cfg.data.n_splits, k, cfg.data.fold_seed)
        folds.append(_names(val))
    return {
        "how": "DecathlonDataset(section=training, val_frac=0, seed=0) 순서 -> holdout_split(frac, holdout_seed) "
               "-> kfold_datalist(개발셋, n_splits, seed 42) 의 val 목록",
        "holdout_frac": cfg.data.holdout_frac,
        "holdout_seed": cfg.data.holdout_seed,
        "n_splits": cfg.data.n_splits,
        "n_total": len(full),
        "holdout": sorted(_names(hold)),
        "dev_folds_val": [sorted(f) for f in folds],
    }


def main() -> int:
    import os

    os.chdir(ROOT)  # 설정의 data.root(./data)를 저장소 기준으로 품
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    out = build(cfg)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[저장] {OUT}  전체 {out['n_total']}, 홀드아웃 {len(out['holdout'])}, "
          f"fold {[len(f) for f in out['dev_folds_val']]}")
    return 0


if __name__ == "__main__":
    import argparse

    # 인자는 없음. 읽어 두어야 --help 가 설명만 찍고 끝남.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    raise SystemExit(main())
