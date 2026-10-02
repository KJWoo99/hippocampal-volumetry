"""출력 QC: 분할 마스크의 자동 품질 점검 + 정제.

해마는 좌우 각 1개의 연결 덩어리가 기본. 분할이 작은 섬(노이즈)을 만들면 부피가 오염됨.
가장 큰 연결성분만 남기고, 비정상 부피/공백을 플래그함. (틀린 결과가 경고 없이 나가지 않게)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def keep_largest_component(mask: np.ndarray, labels=(1, 2)) -> np.ndarray:
    """각 전경 라벨에 대해 가장 큰 연결성분만 남긴다 (노이즈 섬 제거)."""
    out = np.zeros_like(mask)
    for lab in labels:
        binary = mask == lab
        if not binary.any():
            continue
        comp, n = ndimage.label(binary)
        if n <= 1:
            out[binary] = lab
            continue
        sizes = ndimage.sum(binary, comp, index=range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        out[comp == largest] = lab
    return out


# 타당 부피 범위(mm³). 정상 범위가 아니라 "명백한 실패"를 잡는 선임.
#
# 문헌의 성인 해마 부피는 한쪽당 2,780~4,285mm³ 근처이고, 기관, 경계 기준에 따라
# 1,730~5,680mm³ 까지 보고됨. 이 데이터(MSD Task04) 홀드아웃의 정답 평균은 3,152mm³ 로
# 그 안에 들어감. 여기서 쓰는 500~12,000 은 그보다 훨씬 넓은데, 좁히면 정상 변이를
# 실패로 부르게 되기 때문임. 이 선이 잡으려는 것은 빈 마스크, 배경 통째 선택
# 같은 자릿수가 다른 실패임.
#   정상 범위: https://pubmed.ncbi.nlm.nih.gov/7718948/ ,
#             https://www.sciencedirect.com/science/article/pii/S2213158219302542
PLAUSIBLE_RANGE_MM3 = (500.0, 12000.0)


def qc_mask(
    mask: np.ndarray,
    total_mm3: float,
    plausible_range_mm3: tuple[float, float] = PLAUSIBLE_RANGE_MM3,
    labels=(1, 2),
) -> dict:
    """마스크 QC 리포트. 통과 여부 + 플래그 사유."""
    flags: list[str] = []
    mask = np.asarray(mask)

    if not (mask > 0).any():
        flags.append("빈 분할 (전경 없음)")

    for lab in labels:
        binary = mask == lab
        if binary.any():
            _, n = ndimage.label(binary)
            if n > 1:
                flags.append(f"라벨 {lab}: 연결성분 {n}개 (파편화: 노이즈 가능)")

    lo, hi = plausible_range_mm3
    if total_mm3 < lo:
        flags.append(f"부피 과소 {total_mm3:.0f} < {lo:.0f} mm^3")
    elif total_mm3 > hi:
        flags.append(f"부피 과대 {total_mm3:.0f} > {hi:.0f} mm^3")

    return {"passed": len(flags) == 0, "flags": flags, "total_mm3": float(total_mm3)}
