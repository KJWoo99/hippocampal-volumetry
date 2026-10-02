"""벤치마크 산출물에서 README 가 약속한 그림 두 장을 그림.

    python scripts/render_figures.py            # outputs/benchmark_results.json -> outputs/*.png

`dice_curve.png`(아키텍처별 fold 성능 + 홀드아웃 기준선)와 `bland_altman.png`(부피 일치도)를 그림.
두 그림 모두 `benchmark_results.json` 에 이미 들어 있는 값만 씀. 다시 학습하거나 추론하지 않으므로 수치가
JSON 과 어긋날 일이 없음.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hippo.evaluation.plots import plot_bland_altman  # noqa: E402

OUT = ROOT / "outputs"
SRC = OUT / "benchmark_results.json"


def plot_fold_dice(d: dict, out_png: Path) -> Path:
    """아키텍처별 fold Dice 를 점으로, 홀드아웃 Dice 를 가로선으로 그림.

    개발셋 CV 로 고른 아키텍처가 홀드아웃에서 어디에 놓이는지 한눈에 보게 함.
    글자는 영어로 둠: 서버에 한글 글꼴이 없어 네모로 깨짐.
    """
    cv = d["cv_results"]
    archs = list(cv)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, arch in enumerate(archs):
        folds = cv[arch]["per_fold"]
        ys = [f["dice"] for f in folds]
        xs = np.full(len(ys), i) + np.linspace(-0.12, 0.12, len(ys))
        ax.scatter(xs, ys, s=28, zorder=3, label=None)
        m, s = float(np.mean(ys)), float(np.std(ys))
        ax.errorbar(i, m, yerr=s, fmt="_", color="black", capsize=6, markersize=18, zorder=4)
        ax.annotate(f"{m:.4f}", (i, m), textcoords="offset points", xytext=(14, -4), fontsize=8)
    h = d["holdout_result"]
    hm = h["holdout_summary"]["dice"]["mean"]
    hs = h["holdout_summary"]["dice"].get("std", 0.0)
    ax.axhline(hm, color="tab:red", lw=1.2, ls="--", zorder=2,
               label=f"holdout ({h['arch']}, n={h['n_holdout']}): {hm:.4f}")
    ax.axhspan(hm - hs, hm + hs, color="tab:red", alpha=0.08, zorder=1)
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs)
    ax.set_ylabel("Dice")
    ax.set_title("Dev-set CV per fold (dots, mean +/- sd) vs held-out (dashed)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def main() -> int:
    if not SRC.exists():
        print(f"[없음] {SRC}: run_benchmark.py 를 먼저 돌릴 것")
        return 1
    d = json.loads(SRC.read_text(encoding="utf-8"))
    h = d["holdout_result"]
    v = h["volumes"]

    p1 = plot_fold_dice(d, OUT / "dice_curve.png")
    p2 = plot_bland_altman(
        v["predicted_ensemble"], v["reference"], str(OUT / "bland_altman.png"),
        title=f"Bland-Altman, hippocampal volume ({h['arch']} ensemble, n={h['n_holdout']})",
    )
    print(f"[저장] {p1}")
    print(f"[저장] {p2}")
    return 0


if __name__ == "__main__":
    import argparse

    # 인자는 없음. 읽어 두어야 --help 가 설명만 찍고 끝남.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    raise SystemExit(main())
