"""평가 플롯: Dice 곡선, Bland-Altman (부피 일치도)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_dice_curve(history: list[dict], out_png: str) -> str:
    ep = [h["epoch"] for h in history if "dice" in h]
    dc = [h["dice"] for h in history if "dice" in h]
    plt.figure(figsize=(6, 4))
    plt.plot(ep, dc, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("val Dice")
    plt.title("Hippocampus 3D segmentation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    return out_png


def plot_bland_altman(predicted, reference, out_png: str, title: str | None = None) -> str:
    """부피 Bland-Altman 플롯 (표준 절차 전체).

    bias 와 95% LoA 만 그리는 것은 절차의 절반임. 두 가지를 함께 그림.

    1) LoA 신뢰구간: LoA 는 표본 추정치라 그 자체로 불확실함(음영 띠).
    2) 비례편향 회귀선: 차이가 측정값 크기에 따라 달라지면
       "전 구간에서 bias 가 일정하다"는 LoA 의 전제가 깨짐.
       유의하면(p<0.05) 제목에 경고를 넣음.
    """
    from hippo.metrics.volume_agreement import bland_altman

    pred = np.asarray(predicted, dtype=float)
    ref = np.asarray(reference, dtype=float)
    mean = (pred + ref) / 2
    diff = pred - ref
    ba = bland_altman(pred, ref)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(mean, diff, alpha=0.6, s=28, edgecolors="none")

    # LoA 신뢰구간 음영
    for lo, hi in (ba.loa_upper_ci, ba.loa_lower_ci):
        if lo == lo and hi == hi:  # not NaN
            plt.axhspan(lo, hi, color="r", alpha=0.10)

    plt.axhline(ba.bias, color="k", lw=1.5, label=f"bias = {ba.bias:+.1f}")
    plt.axhline(ba.loa_upper, color="r", ls="--", lw=1.2,
                label=f"95% LoA = [{ba.loa_lower:.0f}, {ba.loa_upper:.0f}]")
    plt.axhline(ba.loa_lower, color="r", ls="--", lw=1.2)
    plt.axhline(0, color="gray", lw=0.8, alpha=0.5)

    # 비례편향 회귀선
    if ba.prop_bias_slope == ba.prop_bias_slope:
        xs = np.linspace(mean.min(), mean.max(), 100)
        intercept = ba.bias - ba.prop_bias_slope * mean.mean()
        sig = "significant" if ba.has_proportional_bias else "n.s."
        plt.plot(xs, ba.prop_bias_slope * xs + intercept, color="tab:blue", lw=1.6,
                 label=f"prop. bias: slope={ba.prop_bias_slope:.3f} (p={ba.prop_bias_p:.3f}, {sig})")

    base = title or "Bland-Altman: hippocampal volume"
    if ba.has_proportional_bias:
        base += "\n(proportional bias present - single LoA understates error at large volumes)"
    plt.title(base, fontsize=10)
    plt.xlabel("mean volume (mm$^3$)")
    plt.ylabel("pred - ref (mm$^3$)")
    plt.legend(fontsize=8, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
    return out_png
