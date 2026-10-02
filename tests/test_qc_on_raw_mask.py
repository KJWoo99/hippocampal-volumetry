"""QC 는 정제 전 마스크로 판정해야 한다는 불변식.

`keep_largest_component()` 로 노이즈 섬을 지운 뒤 `qc_mask()` 를 부르면
라벨당 연결성분이 항상 1개라 파편화 검사가 절대 발동하지 않음. 그러면
원본 출력이 아무리 노이즈투성이여도 리포트는 "파편화 없음"이라고 말함.
cli.predict 의 순서(QC 뒤 정제)를 회귀 테스트로 고정함.
"""
from __future__ import annotations

import numpy as np

from hippo.inference.qc import keep_largest_component, qc_mask


def _noisy_mask() -> np.ndarray:
    """라벨 1이 큰 덩어리 + 떨어진 노이즈 섬 3개로 파편화된 마스크."""
    m = np.zeros((24, 24, 24), np.int16)
    m[2:10, 2:10, 2:10] = 1          # 본체 512복셀
    m[20, 20, 20] = 1                # 노이즈 섬
    m[20, 2, 20] = 1
    m[2, 20, 20] = 1
    m[12:18, 12:18, 12:18] = 2       # 라벨 2 는 단일 덩어리
    return m


def test_cleaning_before_qc_hides_fragmentation():
    """정제 후 QC 는 파편화를 못 잡음."""
    raw = _noisy_mask()
    cleaned = keep_largest_component(raw)
    after = qc_mask(cleaned, 3000.0)
    assert not any("파편화" in f for f in after["flags"]), (
        "정제 후에는 연결성분이 1개뿐이라 파편화 플래그가 나올 수 없다"
    )


def test_qc_on_raw_mask_detects_fragmentation():
    """정제 전 마스크로 보면 파편화가 잡힘."""
    raw = _noisy_mask()
    before = qc_mask(raw, 3000.0)
    frag = [f for f in before["flags"] if "파편화" in f]
    assert frag, "노이즈 섬 3개가 있는데 파편화를 못 잡았다"
    assert "라벨 1" in frag[0]
    assert not before["passed"]


def test_cleanup_removed_voxels_is_reported():
    """정제로 몇 복셀이 사라졌는지 세는 계산이 맞는지 확인.

    cli.predict 가 리포트에 넣는 `cleanup_removed_voxels` 와 같은 식임.
    """
    raw = _noisy_mask()
    cleaned = keep_largest_component(raw)
    removed = int((raw > 0).sum() - (cleaned > 0).sum())
    assert removed == 3, f"노이즈 섬 3복셀이 제거돼야 하는데 {removed}"


def test_clean_mask_passes_qc():
    """노이즈가 없으면 파편화 플래그도 없음: 거짓 경보를 내지 않음."""
    m = np.zeros((24, 24, 24), np.int16)
    m[2:10, 2:10, 2:10] = 1
    m[12:18, 12:18, 12:18] = 2
    rep = qc_mask(m, 3000.0)
    assert rep["passed"], rep["flags"]
    assert int((m > 0).sum() - (keep_largest_component(m) > 0).sum()) == 0
