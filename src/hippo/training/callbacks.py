"""학습 콜백: EarlyStopping, ModelCheckpoint.

지표가 클수록 좋은 경우(mode="max")와 작을수록 좋은 경우(mode="min") 모두 지원.
"""
from __future__ import annotations

import torch


class EarlyStopping:
    def __init__(self, patience: int = 20, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.num_bad = 0
        self.should_stop = False

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """개선되면 True. patience 초과 시 should_stop=True."""
        if self._improved(value):
            self.best = value
            self.num_bad = 0
            return True
        self.num_bad += 1
        if self.num_bad >= self.patience:
            self.should_stop = True
        return False


class ModelCheckpoint:
    def __init__(self, path: str, mode: str = "max"):
        self.path = path
        self.mode = mode
        self.best: float | None = None

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        return value > self.best if self.mode == "max" else value < self.best

    def maybe_save(self, value: float, model: torch.nn.Module) -> bool:
        if self._improved(value):
            self.best = value
            torch.save(model.state_dict(), self.path)
            return True
        return False
