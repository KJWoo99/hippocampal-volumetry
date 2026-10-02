import numpy as np

from hippo.inference.qc import keep_largest_component, qc_mask


def test_keep_largest_removes_islands():
    mask = np.zeros((10, 10, 10), dtype=np.int16)
    mask[1:5, 1:5, 1:5] = 1     # 큰 덩어리
    mask[8, 8, 8] = 1            # 떨어진 노이즈 섬
    cleaned = keep_largest_component(mask, labels=(1,))
    assert cleaned[8, 8, 8] == 0
    assert (cleaned == 1).sum() == 64


def test_qc_flags_empty():
    mask = np.zeros((8, 8, 8), dtype=np.int16)
    rep = qc_mask(mask, total_mm3=0.0)
    assert rep["passed"] is False
    assert any("빈 분할" in f for f in rep["flags"])


def test_qc_flags_implausible_volume():
    mask = np.zeros((8, 8, 8), dtype=np.int16)
    mask[0, 0, 0] = 1
    rep = qc_mask(mask, total_mm3=100.0)  # 너무 작음
    assert rep["passed"] is False
    assert any("과소" in f for f in rep["flags"])


def test_qc_passes_plausible():
    mask = np.zeros((20, 20, 20), dtype=np.int16)
    mask[2:12, 2:12, 2:12] = 1  # 단일 덩어리
    rep = qc_mask(mask, total_mm3=3000.0)
    assert rep["passed"] is True
    assert rep["flags"] == []
