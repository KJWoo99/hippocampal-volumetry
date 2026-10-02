import numpy as np
import pytest

from hippo.volumetry.normative import (
    NormativeModel,
    asymmetry_index,
    normalize_by_icv_ratio,
    normalize_by_icv_residual,
)


def test_asymmetry_symmetric_is_zero():
    assert asymmetry_index(3000.0, 3000.0) == pytest.approx(0.0)


def test_asymmetry_sign_and_scale():
    ai = asymmetry_index(3300.0, 2700.0)  # (600)/(3000)*100 = 20%
    assert ai == pytest.approx(20.0)


def test_icv_ratio():
    assert normalize_by_icv_ratio(3000.0, 1.5e6) == pytest.approx(0.002)


def test_icv_ratio_rejects_nonpositive():
    with pytest.raises(ValueError):
        normalize_by_icv_ratio(3000.0, 0.0)


def test_icv_residual():
    # volume = 2*ICV + 100 정확히 맞으면 잔차 0
    r = normalize_by_icv_residual(volume_mm3=2 * 1000 + 100, icv_mm3=1000, slope=2.0, intercept=100.0)
    assert r == pytest.approx(0.0)


def test_normative_zscore_and_percentile():
    m = NormativeModel.illustrative()
    age = 70.0
    mean = m.mean_at(age)
    assert m.zscore(mean, age) == pytest.approx(0.0)
    assert m.percentile(mean, age) == pytest.approx(50.0, abs=1e-6)
    # 평균보다 작으면 percentile < 50 (위축 방향)
    assert m.percentile(mean - m.sd_value, age) < 50.0


# --- from_fit: 정상군 실데이터로 정상곡선 적합 ---

def test_from_fit_recovers_known_line():
    """거의 선형인 데이터를 넣으면 원래 기울기, 절편을 되찾아야 함.

    잡음을 완전히 0 으로 두면 잔차 sd 가 0 이 되어 from_fit 이 (의도대로) 거부함:
    z 점수를 정의할 수 없기 때문임. 실제 정상군에는 개인차가 반드시 있으므로
    아주 작은 잡음을 넣어 현실적인 입력으로 만듦.
    """
    rng = np.random.default_rng(7)
    ages = np.arange(40, 81, 2, dtype=float)
    volumes = 3600.0 - 15.0 * ages + rng.normal(0, 1e-3, ages.size)
    m = NormativeModel.from_fit(ages, volumes)
    assert m.mean_slope == pytest.approx(-15.0, abs=1e-3)
    assert m.mean_intercept == pytest.approx(3600.0, abs=1e-1)


def test_from_fit_rejects_zero_residual():
    """잔차가 정확히 0 이면 z 점수를 정의할 수 없으므로 거부해야 함."""
    ages = np.arange(40, 81, 5, dtype=float)
    volumes = 3600.0 - 15.0 * ages          # 완벽한 직선 = 개인차 0
    with pytest.raises(ValueError, match="잔차 표준편차가 0"):
        NormativeModel.from_fit(ages, volumes)


def test_from_fit_uses_residual_sd_not_raw_sd():
    """sd 는 원자료가 아니라 잔차의 표준편차여야 함.

    연령 범위가 넓으면 원자료 표준편차는 연령 효과 때문에 크게 부풀려짐.
    잔차 표준편차를 써야 연령을 보정한 개인차만 남음.
    """
    rng = np.random.default_rng(0)
    ages = np.linspace(40, 80, 200)
    noise = rng.normal(0, 100.0, ages.size)      # 실제 개인차 sd = 100
    volumes = 3600.0 - 15.0 * ages + noise

    m = NormativeModel.from_fit(ages, volumes)
    raw_sd = float(volumes.std(ddof=1))

    assert m.sd_value == pytest.approx(100.0, rel=0.15), "잔차 sd 를 되찾아야 한다"
    # 연령효과(40~80세 x 15mm3/년 = 폭 600 -> sd 약 173)와 개인차(100)가 합쳐져
    # 원자료 sd 는 sqrt(173^2 + 100^2) ~= 200, 즉 잔차 sd 의 약 2배가 됨.
    assert raw_sd > 1.8 * m.sd_value, "원자료 sd 는 연령 효과로 부풀려져 있다"


def test_from_fit_zscore_is_calibrated():
    """적합한 모델의 z 점수는 정상군에서 평균 0, 표준편차 1 근처여야 함."""
    rng = np.random.default_rng(1)
    ages = np.linspace(45, 85, 300)
    volumes = 3600.0 - 15.0 * ages + rng.normal(0, 120.0, ages.size)
    m = NormativeModel.from_fit(ages, volumes)

    z = np.array([m.zscore(v, a) for v, a in zip(volumes, ages, strict=True)])
    assert abs(z.mean()) < 0.1
    assert z.std() == pytest.approx(1.0, rel=0.1)


@pytest.mark.parametrize(
    "ages, volumes, msg",
    [
        ([50.0, 60.0], [3000.0, 2900.0], "최소 3개"),
        ([50.0, 50.0, 50.0], [3000.0, 2900.0, 3100.0], "연령이 모두 같으면"),
        ([50.0, 60.0, 70.0], [3000.0, 2900.0], "같은 길이"),
    ],
)
def test_from_fit_rejects_bad_input(ages, volumes, msg):
    with pytest.raises(ValueError, match=msg):
        NormativeModel.from_fit(ages, volumes)
