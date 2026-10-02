import numpy as np
import pytest

from hippo.metrics.volume_agreement import bland_altman, icc_2_1, volume_agreement_report


def test_icc_perfect_agreement():
    x = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    assert icc_2_1(x, x) == pytest.approx(1.0, abs=1e-9)


def test_icc_drops_with_systematic_bias():
    ref = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    pred = ref + 50.0  # 체계적 +50 bias -> 절대일치도 하락
    assert icc_2_1(pred, ref) < 1.0


def test_bland_altman_zero_bias_when_equal():
    x = np.array([10.0, 20.0, 30.0, 40.0])
    ba = bland_altman(x, x)
    assert ba.bias == pytest.approx(0.0)
    assert ba.sd_diff == pytest.approx(0.0)


def test_bland_altman_bias_sign():
    ref = np.array([10.0, 20.0, 30.0])
    pred = ref + 5.0
    ba = bland_altman(pred, ref)
    assert ba.bias == pytest.approx(5.0)
    assert ba.loa_upper >= ba.bias >= ba.loa_lower


def test_volume_agreement_report_keys():
    rng = np.random.default_rng(0)
    ref = rng.uniform(1000, 4000, 20)
    pred = ref + rng.normal(0, 50, 20)
    rep = volume_agreement_report(pred, ref)
    assert rep["n"] == 20
    assert 0.0 <= rep["icc_2_1"] <= 1.0
    assert "bland_altman" in rep


def test_icc_requires_min_samples():
    with pytest.raises(ValueError):
        icc_2_1(np.array([1.0]), np.array([1.0]))


def test_loa_confidence_interval_present():
    """LoA 는 표본 추정치라 그 자체로 불확실함. CI 를 함께 내야 함."""
    import numpy as np

    from hippo.metrics.volume_agreement import bland_altman

    rng = np.random.default_rng(0)
    ref = rng.normal(3000, 400, 60)
    pred = ref + rng.normal(0, 120, 60)
    ba = bland_altman(pred, ref)
    assert ba.loa_upper_ci[0] < ba.loa_upper < ba.loa_upper_ci[1]
    assert ba.loa_lower_ci[0] < ba.loa_lower < ba.loa_lower_ci[1]


def test_loa_ci_narrows_with_larger_sample():
    import numpy as np

    from hippo.metrics.volume_agreement import bland_altman

    rng = np.random.default_rng(1)
    widths = []
    for n in (30, 3000):
        ref = rng.normal(3000, 400, n)
        pred = ref + rng.normal(0, 120, n)
        ba = bland_altman(pred, ref)
        widths.append(ba.loa_upper_ci[1] - ba.loa_upper_ci[0])
    assert widths[1] < widths[0], "표본이 커지면 LoA 신뢰구간이 좁아져야 한다"


def test_loa_ci_uses_t_with_n_minus_1_df():
    """Bland & Altman(1986)은 LoA 신뢰구간에 1.96 이 아니라 자유도 n-1 의 t 값을 씀.

    원문 예시: 17쌍(자유도 16)에서 t = 2.12. 1.96 을 쓰면 작은 표본에서 구간이 좁게 나옴.
    """
    import numpy as np
    from scipy import stats

    from hippo.metrics.volume_agreement import bland_altman

    rng = np.random.default_rng(2)
    ref = rng.normal(3000, 400, 17)
    pred = ref + rng.normal(0, 120, 17)
    ba = bland_altman(pred, ref)
    sd = float(np.std(pred - ref, ddof=1))
    half = stats.t.ppf(0.975, 16) * np.sqrt(3.0 * sd**2 / 17)
    assert stats.t.ppf(0.975, 16) == pytest.approx(2.12, abs=0.005)
    assert (ba.loa_upper_ci[1] - ba.loa_upper_ci[0]) / 2 == pytest.approx(half, rel=1e-9)
    assert (ba.loa_lower_ci[1] - ba.loa_lower_ci[0]) / 2 == pytest.approx(half, rel=1e-9)


def test_proportional_bias_detected():
    """차이가 측정값 크기에 비례하면 단일 LoA 로 요약하면 안 됨."""
    import numpy as np

    from hippo.metrics.volume_agreement import bland_altman

    rng = np.random.default_rng(2)
    ref = rng.uniform(2000, 4500, 200)
    # 부피가 클수록 과소추정 (실제 홀드아웃에서 관측된 방향)
    pred = ref - 0.10 * (ref - ref.mean()) + rng.normal(0, 50, 200)
    ba = bland_altman(pred, ref)
    assert ba.has_proportional_bias
    assert ba.prop_bias_slope < 0


def test_no_proportional_bias_when_absent():
    import numpy as np

    from hippo.metrics.volume_agreement import bland_altman

    rng = np.random.default_rng(3)
    ref = rng.uniform(2000, 4500, 300)
    pred = ref + rng.normal(0, 80, 300)      # 크기와 무관한 오차
    ba = bland_altman(pred, ref)
    assert not ba.has_proportional_bias


def test_bland_altman_handles_tiny_sample():
    import numpy as np

    from hippo.metrics.volume_agreement import bland_altman

    ba = bland_altman(np.array([100.0]), np.array([110.0]))
    assert ba.bias == -10.0
    assert np.isnan(ba.prop_bias_p)


def test_subject_bootstrap_draws_whole_subjects():
    """같은 피험자의 케이스를 한 번에 뽑는지 가짜 데이터로 봄(docs/prereg_pair_split.md 두 번째 추가 절 5번).

    30명의 좌우가 똑같은 값이면 실제 독립 단위는 30 임. 피험자 단위로 뽑으면 60건을 따로 뽑을 때보다 구간이
    약 1.4배(루트 2) 넓어야 함.
    """
    from hippo.metrics.volume_agreement import subject_bootstrap_ci

    rng = np.random.default_rng(5)
    ref1 = rng.uniform(2500, 4000, 30)
    pred1 = ref1 + rng.normal(100, 150, 30)
    ref, pred = np.repeat(ref1, 2), np.repeat(pred1, 2)
    by_subject = subject_bootstrap_ci(pred, ref, [i // 2 for i in range(60)])
    by_case = subject_bootstrap_ci(pred, ref, list(range(60)))
    assert by_subject["n_subjects"] == 30 and by_case["n_subjects"] == 60
    w_subj = by_subject["bias"][1] - by_subject["bias"][0]
    w_case = by_case["bias"][1] - by_case["bias"][0]
    assert 1.2 < w_subj / w_case < 1.7
    bias = float(np.mean(pred - ref))
    assert by_subject["bias"][0] < bias < by_subject["bias"][1]
    assert by_subject["icc_2_1"][0] < icc_2_1(pred, ref) < by_subject["icc_2_1"][1]
    assert by_subject == subject_bootstrap_ci(pred, ref, [i // 2 for i in range(60)])
    with pytest.raises(ValueError):
        subject_bootstrap_ci(pred, ref, [0] * 59)


def test_icc_3_1_ignores_a_constant_bias():
    """편향만 있고 흔들림이 없으면 ICC(3,1)은 1 이고 ICC(2,1)은 그보다 낮음."""
    from hippo.metrics.volume_agreement import icc_3_1

    ref = np.linspace(2500, 4000, 30)
    pred = ref + 150.0
    assert icc_3_1(pred, ref) == pytest.approx(1.0)
    assert icc_2_1(pred, ref) < 0.95
    rep = volume_agreement_report(pred, ref)
    assert rep["icc_3_1_consistency"] == pytest.approx(1.0)


def test_icc_3_1_matches_the_two_rater_closed_form():
    """두 측정일 때 ICC(3,1) = 2 cov(x, y) / (var x + var y). 흔들림이 있는 자료로 식을 고정함.

    상관계수나 ICC(3,2)로 잘못 짜면 이 값과 달라짐.
    """
    from hippo.metrics.volume_agreement import icc_3_1

    rng = np.random.default_rng(11)
    ref = rng.uniform(2500, 4000, 40)
    pred = 0.8 * ref + 700 + rng.normal(0, 200, 40)
    c = np.cov(pred, ref, ddof=1)
    want = 2 * c[0, 1] / (c[0, 0] + c[1, 1])
    got = icc_3_1(pred, ref)
    assert got == pytest.approx(want, rel=1e-12)
    assert abs(got - np.corrcoef(pred, ref)[0, 1]) > 1e-3
    assert abs(got - 2 * got / (1 + got)) > 1e-3   # ICC(3,2) 와 구별됨
