"""부피 일치도: 예측 부피와 정답 부피가 얼마나 같은지.

높은 Dice != 정확한 부피. 분할이 체계적으로 과/소추정하면 Dice가 좋아도 부피에 bias가 생김.
그래서 예측부피 vs 정답부피를 ICC(급내상관)와 Bland-Altman으로 별도 검증함.
의료기기 측정 비교에서 흔히 쓰는 방법임.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def icc_2_1(predicted: np.ndarray, reference: np.ndarray) -> float:
    """ICC(2,1): two-way random, single measure, absolute agreement.

    두 측정(예측/정답)을 같은 대상에 대한 반복측정으로 보고 절대 일치도를 계산함.
    1에 가까울수록 부피 측정이 정답과 절대적으로 일치.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape or pred.ndim != 1:
        raise ValueError("predicted/reference는 같은 길이의 1D 배열이어야 함")
    n = len(pred)
    if n < 2:
        raise ValueError("ICC 계산에 최소 2개 샘플 필요")

    data = np.stack([pred, ref], axis=1)  # (n, k=2)
    k = 2
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    # 평균제곱 분해
    ss_total = ((data - grand_mean) ** 2).sum()
    ss_row = k * ((row_means - grand_mean) ** 2).sum()
    ss_col = n * ((col_means - grand_mean) ** 2).sum()
    ss_err = ss_total - ss_row - ss_col

    ms_row = ss_row / (n - 1)
    ms_col = ss_col / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom = ms_row + (k - 1) * ms_err + k * (ms_col - ms_err) / n
    if denom == 0:
        return 1.0
    return float((ms_row - ms_err) / denom)


def icc_3_1(predicted: np.ndarray, reference: np.ndarray) -> float:
    """ICC(3,1): two-way mixed, single measure, consistency.

    두 측정 사이의 평균 차이(편향)를 빼고 본 일치도. ICC(2,1)과 나란히 두면 ICC 가 낮은 이유가 전체를 한쪽으로
    미는 편향인지, 케이스마다 흔들리는 오차인지 가를 수 있음. 판정에는 ICC(2,1)을 쓰고 이 값은 해석에만 씀.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape or pred.ndim != 1:
        raise ValueError("predicted/reference는 같은 길이의 1D 배열이어야 함")
    n = len(pred)
    if n < 2:
        raise ValueError("ICC 계산에 최소 2개 샘플 필요")
    data = np.stack([pred, ref], axis=1)
    k = 2
    grand_mean = data.mean()
    ss_total = ((data - grand_mean) ** 2).sum()
    ss_row = k * ((data.mean(axis=1) - grand_mean) ** 2).sum()
    ss_col = n * ((data.mean(axis=0) - grand_mean) ** 2).sum()
    ms_row = ss_row / (n - 1)
    ms_err = (ss_total - ss_row - ss_col) / ((n - 1) * (k - 1))
    denom = ms_row + (k - 1) * ms_err
    return 1.0 if denom == 0 else float((ms_row - ms_err) / denom)


@dataclass
class BlandAltman:
    bias: float                # 평균 차이 (예측 - 정답)
    sd_diff: float             # 차이의 표준편차
    loa_lower: float           # 95% 일치 한계 하한 (bias - 1.96*sd)
    loa_upper: float           # 95% 일치 한계 상한 (bias + 1.96*sd)
    mean_pct_error: float      # 평균 절대 백분율 오차

    # LoA 자체의 불확실성. 표본이 작으면 LoA 는 넓게 흔들림.
    loa_upper_ci: tuple[float, float] = (float("nan"), float("nan"))
    loa_lower_ci: tuple[float, float] = (float("nan"), float("nan"))
    # 비례편향: 차이가 측정값 크기에 따라 달라지는가.
    # 기울기가 유의하면 "전 구간에 걸쳐 동일한 bias" 라는 LoA 의 전제가 깨짐.
    prop_bias_slope: float = float("nan")
    prop_bias_p: float = float("nan")
    prop_bias_r: float = float("nan")

    @property
    def has_proportional_bias(self) -> bool:
        return bool(self.prop_bias_p == self.prop_bias_p and self.prop_bias_p < 0.05)


def bland_altman(predicted: np.ndarray, reference: np.ndarray) -> BlandAltman:
    """Bland-Altman: bias, 95% 일치 한계, LoA 신뢰구간, 비례편향.

    LoA 만 보고하는 것은 절차의 절반임. 두 가지가 더 필요함.

    1) LoA 신뢰구간: LoA 는 표본에서 추정한 값이라 그 자체로 불확실함.
       n 이 작으면 LoA 가 넓게 흔들림. SE(LoA) ≈ sqrt(3 * s²/n).

    2) 비례편향(proportional bias): 차이(diff)를 평균(mean)에 회귀시켜
       기울기가 0 과 다른지 봄. 유의하면 "측정 범위 전체에서 bias 가 일정하다"는
       LoA 의 전제가 깨지므로, 단일 LoA 로 요약하면 안 됨.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    diff = pred - ref
    avg = (pred + ref) / 2.0
    n = len(diff)

    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.abs(diff) / np.where(ref == 0, np.nan, ref) * 100.0

    # LoA 의 표준오차와 95% 신뢰구간 (Bland & Altman 1986). 원문은 구간의 배수로
    # 1.96 이 아니라 자유도 n-1 인 t 분포의 값을 씀(예시: 자유도 16 에서 t = 2.12).
    # n=52 에서 t 는 2.008 이라 1.96 을 쓰면 구간이 약 2.4% 좁게 나옴.
    if n > 1:
        from scipy import stats

        se_loa = float(np.sqrt(3.0 * sd**2 / n))
        t_crit = float(stats.t.ppf(0.975, n - 1))
        hi_ci = (hi - t_crit * se_loa, hi + t_crit * se_loa)
        lo_ci = (lo - t_crit * se_loa, lo + t_crit * se_loa)
    else:
        hi_ci = lo_ci = (float("nan"), float("nan"))

    # 비례편향 회귀
    slope = p_val = r_val = float("nan")
    if n > 2 and np.ptp(avg) > 0:
        from scipy import stats

        res = stats.linregress(avg, diff)
        slope, p_val, r_val = float(res.slope), float(res.pvalue), float(res.rvalue)

    return BlandAltman(
        bias=bias,
        sd_diff=sd,
        loa_lower=lo,
        loa_upper=hi,
        mean_pct_error=float(np.nanmean(pct)) if n else 0.0,
        loa_upper_ci=hi_ci,
        loa_lower_ci=lo_ci,
        prop_bias_slope=slope,
        prop_bias_p=p_val,
        prop_bias_r=r_val,
    )


def volume_agreement_report(predicted: np.ndarray, reference: np.ndarray) -> dict:
    """ICC + Bland-Altman + Pearson r 한 번에."""
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    pearson = float(np.corrcoef(pred, ref)[0, 1]) if len(pred) > 1 else 1.0
    ba = bland_altman(pred, ref)
    # `has_proportional_bias` 는 property 라 asdict 가 빼먹음. 판정 결과가
    # JSON 에 없으면 읽는 쪽이 p<0.05 임계값을 각자 다시 써야 하고, 그러면
    # 임계값이 두 곳에 흩어짐. 여기서 한 번만 정하고 결과에 박아 넣음.
    ba_dict = asdict(ba)
    ba_dict["has_proportional_bias"] = ba.has_proportional_bias
    # LoA 를 mm³ 로만 적으면 그 폭이 큰지 작은지 판단할 근거가 없음. 해마
    # 부피는 개체마다 2,400~4,400mm³ 범위라(이 데이터), 같은 300mm³ 라도 참조 평균 대비
    # 몇 %인지가 임상에서 읽히는 숫자임. 읽는 쪽이 매번 나누지 않도록 여기서
    # 한 번 계산해 둠.
    ref_mean = float(np.mean(ref)) if len(ref) else float("nan")
    half = (ba.loa_upper - ba.loa_lower) / 2
    return {
        "n": int(len(pred)),
        "icc_2_1": icc_2_1(pred, ref),
        "icc_3_1_consistency": icc_3_1(pred, ref),
        "pearson_r": pearson,
        "reference_mean": ref_mean,
        "loa_halfwidth_pct_of_reference": (
            float(half / ref_mean * 100) if ref_mean else float("nan")),
        "bland_altman": ba_dict,
    }


def subject_bootstrap_ci(predicted: np.ndarray, reference: np.ndarray, groups: list,
                         n_boot: int = 1000, seed: int = 42) -> dict:
    """ICC, Bland-Altman 편향과 한계, 비례편향 기울기의 피험자 단위 부트스트랩 95% 구간(백분위).

    홀드아웃 52건에는 같은 사람의 좌우 해마가 함께 든 짝이 있어 케이스끼리 독립이 아님. 위의 식(LoA 신뢰구간,
    비례편향 p)은 케이스를 독립으로 보므로, 같은 피험자(`groups` 값이 같은 케이스)를 한 번에 뽑는 부트스트랩
    구간을 함께 냄(docs/prereg_pair_split.md 두 번째 추가 절 5번).
    """
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if not (len(pred) == len(ref) == len(groups)):
        raise ValueError("predicted, reference, groups 는 길이가 같아야 함")
    units: dict = {}
    for i, g in enumerate(groups):
        units.setdefault(g, []).append(i)
    unit_list = list(units.values())
    rng = np.random.default_rng(seed)
    keys = ("icc_2_1", "bias", "loa_lower", "loa_upper", "prop_bias_slope")
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(unit_list), len(unit_list))
        ix = np.array([i for u in pick for i in unit_list[u]])
        ba = bland_altman(pred[ix], ref[ix])
        draws["icc_2_1"].append(icc_2_1(pred[ix], ref[ix]))
        draws["bias"].append(ba.bias)
        draws["loa_lower"].append(ba.loa_lower)
        draws["loa_upper"].append(ba.loa_upper)
        draws["prop_bias_slope"].append(ba.prop_bias_slope)
    out: dict = {"n_cases": int(len(pred)), "n_subjects": len(unit_list), "n_boot": n_boot, "seed": seed,
                 "n_nan": {}}
    for k in keys:
        a = np.asarray(draws[k])
        # 뽑힌 표본이 한 값으로 몰리면 기울기가 NaN 이 될 수 있음. 뺀 개수를 남기고, 전부 NaN 이면 멈춤.
        nan = int(np.isnan(a).sum())
        out["n_nan"][k] = nan
        a = a[~np.isnan(a)]
        if a.size == 0:
            raise ValueError(f"{k} 의 부트스트랩 값이 전부 NaN 임")
        out[k] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return out
