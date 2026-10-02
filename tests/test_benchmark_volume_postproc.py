"""홀드아웃 부피는 Dice 와 같은 마스크(같은 후처리)로 재야 함(트러블슈팅 12).

Trainer._validate 는 cfg.train.postproc_largest_component 가 켜져 있으면 최대 연결성분만 남긴 마스크로
Dice 를 측정함. 부피를 argmax 그대로의 마스크로 재면 보고한 Dice 와 ICC, Bland-Altman 이 서로 다른
마스크의 값이 되고 배포(`hippo predict`)와도 달라짐.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import hippo.evaluation.benchmark as bm
import hippo.evaluation.crossval as cv


class _Img:
    affine = torch.eye(4)

    def to(self, _device):
        return self


class _StubPredictor:
    """큰 덩어리(27 복셀) + 떨어진 섬(1 복셀)을 가진 라벨 1 마스크를 내놓음."""

    def __init__(self, *a, **k):
        pass

    def predict_mask(self, _x):
        m = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        m[0, 1:4, 1:4, 1:4] = 1
        m[0, 7, 7, 7] = 1
        return m


def _cfg(postproc: bool):
    return SimpleNamespace(model=SimpleNamespace(roi=(8, 8, 8)),
                           train=SimpleNamespace(amp=False, postproc_largest_component=postproc))


def test_collect_volumes_applies_the_same_postproc_as_dice(monkeypatch):
    monkeypatch.setattr(bm, "Predictor", _StubPredictor)
    label = torch.zeros(1, 1, 8, 8, 8)
    label[0, 0, 1:4, 1:4, 1:4] = 1
    loader = [{"image": _Img(), "label": label}]

    on, ref_on = bm._collect_volumes(None, loader, _cfg(True), "cpu")
    off, _ = bm._collect_volumes(None, loader, _cfg(False), "cpu")
    assert np.isclose(on[0], 27.0) and np.isclose(ref_on[0], 27.0)   # 섬이 지워져 정답과 같음
    assert np.isclose(off[0], 28.0)                                   # 후처리를 끄면 섬까지 셈


def test_crossval_collect_volumes_applies_the_same_postproc(monkeypatch):
    """구버전 crossval 경로도 같은 후처리를 적용함."""
    monkeypatch.setattr(cv, "Predictor", _StubPredictor)
    label = torch.zeros(1, 1, 8, 8, 8)
    label[0, 0, 1:4, 1:4, 1:4] = 1
    loader = [{"image": _Img(), "label": label}]

    on, _ = cv._collect_volumes(None, loader, _cfg(True), "cpu")
    off, _ = cv._collect_volumes(None, loader, _cfg(False), "cpu")
    assert np.isclose(on[0], 27.0)
    assert np.isclose(off[0], 28.0)
