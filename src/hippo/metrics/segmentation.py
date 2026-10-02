"""분할 메트릭 묶음: Dice + 표면 거리(HD95, ASD).

Dice만 보면 경계 품질을 못 봄. 의료 분할 논문은 HD95/ASD(표면 거리)를 함께 봄.
MONAI의 누적형 Metric을 감싸 한 번에 reset/aggregate 함.
"""
from __future__ import annotations

import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric


class SegMetrics:
    """배치별로 update하고, 에폭/폴드 끝에 aggregate한다 (background 제외)."""

    def __init__(self, percentile: float = 95.0):
        self.dice = DiceMetric(include_background=False, reduction="mean")
        self.hd95 = HausdorffDistanceMetric(
            include_background=False, percentile=percentile, reduction="mean"
        )
        self.asd = SurfaceDistanceMetric(
            include_background=False, symmetric=True, reduction="mean"
        )

    def reset(self) -> None:
        self.dice.reset()
        self.hd95.reset()
        self.asd.reset()

    def update(self, y_pred: list[torch.Tensor], y: list[torch.Tensor]) -> None:
        """y_pred, y는 one-hot (C,H,W,D) 텐서 리스트 (decollate 후 post-transform 결과)."""
        self.dice(y_pred=y_pred, y=y)
        self.hd95(y_pred=y_pred, y=y)
        self.asd(y_pred=y_pred, y=y)

    @staticmethod
    def _agg(metric) -> float:
        val = metric.aggregate()
        val = val[0] if isinstance(val, (list, tuple)) else val
        return float(val.item())

    def aggregate(self) -> dict[str, float]:
        return {
            "dice": self._agg(self.dice),
            "hd95": self._agg(self.hd95),
            "asd": self._agg(self.asd),
        }
