"""임상 확장층: ICV 정규화 / 좌우 비대칭 / 연령 정상군 대비.

데이터 한계 명시:
Task04 Hippocampus는 해마 주변 crop이라 좌/우(L/R)도 ICV(두개내용적)도 없음.
따라서 이 모듈은 '전뇌 데이터(예: OASIS)가 들어오면 동작하는 인터페이스'로 설계됐고,
정상군 파라미터는 문헌 기반 예시값(illustrative)임. 실데이터 연결 시 fit 으로 교체함.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


def normalize_by_icv_ratio(volume_mm3: float, icv_mm3: float) -> float:
    """단순 비율 정규화: volume / ICV. 직관적이나 통계적으로 잔차법보다 약함."""
    if icv_mm3 <= 0:
        raise ValueError("ICV는 양수여야 함")
    return volume_mm3 / icv_mm3


def normalize_by_icv_residual(
    volume_mm3: float, icv_mm3: float, slope: float, intercept: float
) -> float:
    """회귀 잔차 정규화 (Jack et al.): 정상군에서 적합한 volume~ICV 회귀의 잔차.

    residual = observed - (slope * ICV + intercept).
    단순 나눗셈보다 머리 크기 효과를 통계적으로 더 옳게 제거함.
    slope/intercept 는 정상군에서 ``numpy.polyfit(icv, volume, 1)`` 로 적합해 전달.
    """
    expected = slope * icv_mm3 + intercept
    return float(volume_mm3 - expected)


def asymmetry_index(left_mm3: float, right_mm3: float) -> float:
    """좌우 비대칭 지수(%) = (L - R) / ((L+R)/2) * 100. AD/뇌전증 바이오마커."""
    denom = (left_mm3 + right_mm3) / 2.0
    if denom == 0:
        return 0.0
    return (left_mm3 - right_mm3) / denom * 100.0


@dataclass
class NormativeModel:
    """연령별 정상 해마 부피 정상곡선 (mean, sd as linear functions of age).

    문헌 기반 예시 파라미터. 실데이터로 교체 가능하도록 from_fit 제공.
    z = (value - mean(age)) / sd(age), percentile = Phi(z).
    """

    mean_intercept: float        # age=0 가상 절편 (mm^3)
    mean_slope: float            # 연령당 부피 변화(mm^3/year), 보통 음수(위축)
    sd_value: float              # 정상군 표준편차(mm^3)

    @classmethod
    def illustrative(cls) -> NormativeModel:
        """문헌 범위 안에서 고른 예시 파라미터.

        이 값들이 실제로 만드는 곡선(직접 계산해 확인한 값):

            40세 3000mm^3 -> 연 0.50% 감소
            60세 2700mm^3 -> 연 0.56% 감소
            80세 2400mm^3 -> 연 0.62% 감소

        정상 노화의 해마 위축률은 보고마다 편차가 큼: 0.49%/년을 보고한 코호트가
        있는가 하면 1.3~1.8%/년을 보고한 고령 코호트도 있고, 나이가 들수록 가속됨.
        여기 값은 그 범위의 낮은 쪽(≈0.5%/년, 완만한 선형)에 해당함.

        데모용 값: 실제 임상 판정에 쓰면 안 됨. OASIS 등 실데이터로 ``from_fit``
        적합해 교체하는 것이 전제임.
        """
        return cls(mean_intercept=3600.0, mean_slope=-15.0, sd_value=350.0)

    @classmethod
    def from_fit(
        cls, ages: Sequence[float], volumes: Sequence[float]
    ) -> NormativeModel:
        """정상군 실데이터로 정상곡선을 적합한다 (OASIS 등 연결 시 사용).

        부피를 연령에 1차 회귀해 mean(age) 을 얻고, 그 잔차의 표준편차를 sd 로 씀.
        sd 를 원자료의 표준편차가 아니라 잔차의 표준편차로 잡는 이유는, 연령
        효과를 제거한 뒤 남은 개인차만이 z 점수의 기준이 되어야 하기 때문임.
        원자료 표준편차를 쓰면 연령 분포가 넓은 코호트에서 sd 가 부풀려져 z 가
        일괄적으로 0 쪽으로 눌림.

        표본이 3개 미만이면 잔차 표준편차를 정의할 수 없어 거부함.
        """
        age_arr = np.asarray(ages, dtype=float)
        vol_arr = np.asarray(volumes, dtype=float)
        if age_arr.shape != vol_arr.shape or age_arr.ndim != 1:
            raise ValueError("ages 와 volumes 는 같은 길이의 1D 배열이어야 함")
        if len(age_arr) < 3:
            raise ValueError("정상곡선 적합에는 최소 3개 표본이 필요함")
        if np.ptp(age_arr) == 0:
            raise ValueError("연령이 모두 같으면 기울기를 적합할 수 없음")

        slope, intercept = np.polyfit(age_arr, vol_arr, 1)
        residual = vol_arr - (slope * age_arr + intercept)
        sd = float(residual.std(ddof=2))  # 기울기, 절편 2개를 추정했으므로 자유도 n-2
        # 완벽한 직선이어도 BLAS 에 따라 잔차가 0 대신 1e-13 쯤으로 남음. 부피 크기 대비 수치 오차 수준이면 0 으로 봄.
        if sd <= 1e-9 * max(1.0, float(np.abs(vol_arr).max())):
            raise ValueError("잔차 표준편차가 0 이다: z 점수를 정의할 수 없음")
        return cls(mean_intercept=float(intercept), mean_slope=float(slope), sd_value=sd)

    def mean_at(self, age: float) -> float:
        return self.mean_intercept + self.mean_slope * age

    def zscore(self, value_mm3: float, age: float) -> float:
        if self.sd_value <= 0:
            raise ValueError("sd_value는 양수여야 함")
        return (value_mm3 - self.mean_at(age)) / self.sd_value

    def percentile(self, value_mm3: float, age: float) -> float:
        """정상군 대비 백분위수(0~100)."""
        return float(stats.norm.cdf(self.zscore(value_mm3, age)) * 100.0)
