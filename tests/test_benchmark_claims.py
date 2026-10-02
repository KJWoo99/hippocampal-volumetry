"""벤치마크 산출물이 방법론 주장을 실제로 지키는지 검증.

`evaluation/benchmark.py` 는 학습, GPU 가 필요해 단위 테스트로 직접 돌릴 수 없음.
대신 그것이 만든 `outputs/benchmark_results.json` 을 검사해, README 가 내세우는
방법론 주장이 산출물에서 실제로 성립하는지 확인함.

여기서 지키려는 것 세 가지.

1. ICC 독립성: fold 체크포인트 5개를 홀드아웃 52건에 적용하면 예측이 260개가
   됨. 이걸 한 배열로 합쳐 ICC 를 계산하면 같은 개체가 5번 들어가 n 이 부풀려지고
   신뢰도가 과대평가됨. 앙상블 ICC 는 반드시 n=52 여야 함.
2. winner's curse 없음: 아키텍처 선택은 개발셋 CV 로만 해야 함. 홀드아웃을
   보고 고르면 홀드아웃이 더 이상 홀드아웃이 아님.
3. 수치 재현성: 저장된 원자료(부피 배열)로 다시 계산한 ICC, Bland-Altman 이
   저장된 요약값과 일치해야 함. 어긋나면 리포트 수치의 출처를 믿을 수 없음.

산출물이 없으면(다른 사람이 클론한 직후) 건너뜀.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from hippo.metrics.volume_agreement import volume_agreement_report

RESULTS = Path(__file__).resolve().parents[1] / "outputs" / "benchmark_results.json"
pytestmark = pytest.mark.skipif(
    not RESULTS.exists(), reason="benchmark_results.json 없음: run_benchmark.py 먼저 실행"
)


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_ensemble_icc_uses_independent_cases_not_stacked(result):
    """앙상블 ICC 의 n 이 홀드아웃 케이스 수와 같아야 한다 (체크포인트 수만큼 부풀지 않음)."""
    h = result["holdout_result"]
    n_case = h["n_holdout"]
    n_ckpt = h["n_checkpoints"]

    assert h["ensemble_agreement"]["n"] == n_case, (
        f"앙상블 ICC 의 n 이 {h['ensemble_agreement']['n']}: "
        f"케이스 수({n_case})여야 한다. {n_case * n_ckpt} 이면 예측을 쌓아 독립성이 깨진 것이다."
    )
    assert h["ensemble_agreement"]["n"] != n_case * n_ckpt


def test_per_ckpt_agreement_is_distribution_over_checkpoints(result):
    """체크포인트별 지표는 각각 n=52 로 계산한 값들의 분포여야 함."""
    h = result["holdout_result"]
    assert len(h["per_ckpt"]) == h["n_checkpoints"]

    for key in ("icc_2_1", "bias"):
        vals = [c[key] for c in h["per_ckpt"]]
        agg = h["per_ckpt_agreement"][key]
        assert agg["mean"] == pytest.approx(float(np.mean(vals)), rel=1e-9)
        assert agg["std"] == pytest.approx(float(np.std(vals)), rel=1e-9)


def test_architecture_selected_by_cv_not_holdout(result):
    """선택된 아키텍처가 개발셋 CV 최고여야 함: 홀드아웃을 보고 고르면 안 됨."""
    cv = result["cv_results"]
    cv_best = max(cv, key=lambda a: cv[a]["cv_summary"]["dice"]["mean"])
    assert result["selected_arch"] == cv_best, (
        f"선택 {result['selected_arch']} 인데 CV 최고는 {cv_best}: "
        "홀드아웃을 보고 고른 흔적이면 winner's curse 다."
    )


def test_stored_agreement_recomputes_from_raw_volumes(result):
    """저장된 부피 배열로 다시 계산한 값이 저장된 요약과 일치해야 함."""
    h = result["holdout_result"]
    vols = h["volumes"]
    pred = np.array(vols["predicted_ensemble"], dtype=float)
    ref = np.array(vols["reference"], dtype=float)

    recomputed = volume_agreement_report(pred, ref)
    stored = h["ensemble_agreement"]

    assert recomputed["n"] == stored["n"]
    assert recomputed["icc_2_1"] == pytest.approx(stored["icc_2_1"], rel=1e-9)
    assert recomputed["pearson_r"] == pytest.approx(stored["pearson_r"], rel=1e-9)
    for key in ("bias", "sd_diff", "loa_lower", "loa_upper", "prop_bias_slope", "prop_bias_p"):
        assert recomputed["bland_altman"][key] == pytest.approx(
            stored["bland_altman"][key], rel=1e-9
        ), f"{key} 가 원자료에서 재현되지 않는다"


def test_ensemble_is_mean_of_per_checkpoint_predictions(result):
    """앙상블 예측이 체크포인트별 예측의 케이스별 평균이어야 함."""
    vols = result["holdout_result"]["volumes"]
    per_ckpt = np.array(vols["predicted_per_ckpt"], dtype=float)   # [n_ckpt, n_case]
    ensemble = np.array(vols["predicted_ensemble"], dtype=float)   # [n_case]

    assert per_ckpt.shape[1] == ensemble.shape[0]
    np.testing.assert_allclose(per_ckpt.mean(axis=0), ensemble, rtol=1e-9)


def test_proportional_bias_claim_matches_outputs(result):
    """비례편향에 대한 README 서술이 산출물과 같은 방향이어야 함.

    재학습하면 유의성은 바뀔 수 있고, 그때 고쳐야 할 것은 산출물이 아니라 문서임.
    산출물이 기준이고, 문서가 거기에 맞는지 봄.
    """
    ba = result["holdout_result"]["ensemble_agreement"]["bland_altman"]
    text = _docs_text()
    # 문장에서 주장을 추출하려 하면 부정문을 긍정으로 읽음.
    # 그래서 README 가 판정을 한 줄로 명시하고, 여기서는 그 줄만 봄.
    m = re.search(r"(?m)^(?:\*\*)?비례편향 판정: (유의함|유의하지 않음)(?:\*\*)?$", text)
    assert m, "문서에 '비례편향 판정: …' 줄이 없다"
    claims_bias = m.group(1) == "유의함"
    assert claims_bias == ba["has_proportional_bias"], (
        f"산출물의 비례편향 유의성은 {ba['has_proportional_bias']} "
        f"(기울기 {ba['prop_bias_slope']:.3f}, p={ba['prop_bias_p']:.3f}) 인데 "
        f"README 의 서술은 {'있다' if claims_bias else '없다/언급 없음'} 쪽이다")
    if ba["has_proportional_bias"]:
        assert ba["prop_bias_slope"] < 0, (
            "README 는 '부피가 클수록 과소추정' 이라고 적었다. 기울기가 양수면 반대다")


_ROOT = Path(__file__).resolve().parents[1]
_README = _ROOT / "README.md"
_BENCH = _ROOT / "outputs" / "benchmark_results.json"
_BACKEND = _ROOT / "outputs" / "backend_benchmark.json"
# 결과 절은 docs/report.md 에 있음. README 가 인용하던 수치는 README 와 report 를 합친 글에서 찾음.
# 트러블슈팅은 옛 값과 새 값을 함께 적어 두므로 넣지 않음(넣으면 README, report 에서 빠진 수치를 놓침).
_REPORT = _ROOT / "docs" / "report.md"


def _docs_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in (_README, _REPORT) if p.exists())


@pytest.mark.skipif(not (_README.exists() and _BENCH.exists()),
                    reason="README 또는 벤치 산출물 없음")
def test_readme_headline_numbers_match_outputs():
    """README 와 결과 보고서의 결론 수치가 산출물과 같아야 함.

    첫 화면은 가장 먼저 읽히는 자리임. 손으로 적는 자리이므로
    대조를 코드로 둠.
    """
    text = _docs_text()
    d = json.loads(_BENCH.read_text(encoding="utf-8"))
    h = d["holdout_result"]
    ea = h["ensemble_agreement"]
    ba = ea["bland_altman"]
    want = {
        "홀드아웃 Dice": f"{h['holdout_summary']['dice']['mean']:.4f}",
        "ICC": f"{ea['icc_2_1']:.4f}",
        "LoA 반폭%": f"{ea['loa_halfwidth_pct_of_reference']:.1f}%",
        "LoA 폭": f"{ba['loa_upper'] - ba['loa_lower']:.1f}",
        "ICC(3,1)": f"{ea['icc_3_1_consistency']:.4f}",
        "비례편향 기울기": f"{ba['prop_bias_slope']:.3f}".replace("-", "−"),
        "비례편향 p": f"{ba['prop_bias_p']:.3f}",
    }
    # 1, 2위 아키텍처의 차이. 어느 것이 1위인지는 재학습에 따라 바뀌므로
    # 이름을 박지 않고 순위로 잡음. README 는 크기를 적으므로 절댓값으로 봄.
    cv = d["cv_results"]
    ranked = sorted(cv.items(), key=lambda kv: -kv[1]["cv_summary"]["dice"]["mean"])
    (first, f_res), (second, s_res) = ranked[0], ranked[1]
    gap = f_res["cv_summary"]["dice"]["mean"] - s_res["cv_summary"]["dice"]["mean"]
    want[f"{first}↔{second} 차이"] = f"{gap:.4f}"
    want["1위 fold 표준편차"] = f"{f_res['cv_summary']['dice']['std']:.4f}"
    if _BACKEND.exists():
        b = json.loads(_BACKEND.read_text(encoding="utf-8"))["backends"]
        # 백엔드는 환경에 따라 없을 수 있음. TensorRT 는 설치된 장비에서만 나오고,
        # 없으면 그 자리에 설치 안내 문자열이 들어감. 나온 것만 대조하고, 안 나온 것은 건너뜀.
        for name, label in (("pytorch_fp32", "PyTorch"), ("onnxruntime", "ONNXRuntime"),
                            ("tensorrt_fp16", "TensorRT")):
            entry = b.get(name)
            if not isinstance(entry, dict) or "latency_ms" not in entry:
                continue
            want[f"{label} latency"] = f"{entry['latency_ms']:.2f}ms"
            if isinstance(entry.get("speedup_vs_pytorch"), (int, float)) and name != "pytorch_fp32":
                want[f"{label} 가속비"] = f"{entry['speedup_vs_pytorch']:.2f}x"
    missing = [f"{k}={v}" for k, v in want.items() if v not in text]
    assert not missing, "README 에 없는 산출물 수치: " + ", ".join(missing)


_CARD = _ROOT / "docs" / "MODEL_CARD.md"


@pytest.mark.skipif(not (_CARD.exists() and _BENCH.exists()),
                    reason="모델카드 또는 벤치 산출물 없음")
def test_model_card_numbers_match_outputs():
    """모델카드의 수치도 산출물과 같아야 함.

    README 만 대조하면 재벤치 뒤 모델카드만 옛 Dice, ICC 로 남을 수 있음.
    모델카드는 남에게 건네는 요약본이라 오히려 이쪽이 읽힘.

    비례편향은 숫자보다 결론이 중요함. p 가 판정선을 넘나들면 "유의했다"와
    "단정하지 않는다"가 뒤집히므로, 값과 서술을 함께 봄.
    """
    text = _CARD.read_text(encoding="utf-8")
    d = json.loads(_BENCH.read_text(encoding="utf-8"))
    h = d["holdout_result"]
    ea = h["ensemble_agreement"]
    ba = ea["bland_altman"]
    want = {
        "홀드아웃 Dice": f"{h['holdout_summary']['dice']['mean']:.4f}",
        "ICC": f"{ea['icc_2_1']:.4f}",
        "LoA 반폭%": f"{ea['loa_halfwidth_pct_of_reference']:.1f}%",
        "LoA 폭": f"{ba['loa_upper'] - ba['loa_lower']:.1f}",
    }
    missing = [f"{k}={v}" for k, v in want.items() if v not in text]
    assert not missing, (
        "모델카드에 없는 산출물 수치: " + ", ".join(missing)
        + ": README 만 고치고 모델카드를 두면 두 문서가 서로 다른 말을 한다")

    # 유의성 서술이 실제 p 와 같은 편이어야 함.
    p_val = float(ba["prop_bias_p"])
    says_significant = "유의했다" in text or "유의하다" in text
    if p_val >= 0.05:
        assert not says_significant, (
            f"p={p_val:.3f} 인데 모델카드가 비례편향을 유의하다고 적었다")


@pytest.mark.skipif(not _README.exists(), reason="README 없음")
def test_readme_leads_with_the_finding_not_a_feature_list():
    """첫 화면이 기능 나열이 아니라 발견(수치)이어야 함."""
    text = _README.read_text(encoding="utf-8")
    i_answer = text.find("## 요약")
    i_struct = text.find("## 구조")
    assert i_answer != -1, "README 에 결론 절('## 요약')이 없다"
    assert i_struct == -1 or i_answer < i_struct, "결론이 구조 설명보다 뒤에 있다"
    assert "종단" in text[i_answer:i_answer + 2000], (
        "첫 화면에 종단 추적 불가 결론이 없다: 이 프로젝트의 핵심이다")


_DEG = _ROOT / "outputs" / "degradation_results.json"


@pytest.mark.skipif(not (_README.exists() and _DEG.exists()), reason="README 또는 저하 분석 산출물 없음")
def test_readme_degradation_table_matches_outputs():
    """결과 보고서 부가 분석 절의 표와 판정이 degradation_results.json 에서 그대로 나오는가."""
    text = _docs_text()
    start = text.index("## 부가 분석. 화질 저하와 복원이 부피 일치도에 주는 영향")
    end = text.find("\n## ", start + 10)
    sec = text[start:end if end != -1 else len(text)]   # report 의 마지막 절이면 글 끝까지
    r = json.loads(_DEG.read_text(encoding="utf-8"))
    assert r["path_check"]["same"]
    classic = {"noise": "nlm", "bias": "n4", "thick": "cubic"}
    labels = {"noise": "잡음 f ", "bias": "바이어스 c ", "thick": "절편 "}
    for tag, v in r["levels"].items():
        kind = next(k for k in classic if tag.startswith(k))
        lv = tag[len(kind):]
        name = labels[kind] + (f"{lv} mm" if kind == "thick" else ("0.10" if lv == "0.1" and kind == "noise" else lv))
        f4 = lambda x: f"{x:.4f}"  # noqa: E731
        row = (f"| {name} | {f4(v['icc_degraded'])} | {f4(v['drop'])} [{f4(v['drop_ci'][0])}, {f4(v['drop_ci'][1])}] | "
               f"{'예' if v['drop_confirmed'] else '아니오'} | {f4(v['methods']['unet']['icc'])} | "
               f"{f4(v['methods'][classic[kind]]['icc'])} | {f4(v['icc_finetuned'])} |")
        assert row in sec, f"README 표에 없는 줄: {row}"
    assert r["primary"]["verdict"].startswith("잡음 f=0.06 에서는 저하가 부피 일치도를 해치지 않았다")
    assert "잡음 f=0.06 에서는 저하가 부피 일치도를 해치지 않았다" in sec
    rec = r["levels"]["thick4"]["methods"]["unet"]
    assert f"{round(rec['recovery'] * 100)}% 를 되찾고 [{round(rec['recovery_ci'][0] * 100)}%, {round(rec['recovery_ci'][1] * 100)}%]" in sec
    assert f"{r['psnr_up_bias_up']['count']}건" in sec
    ratio = _ROOT / "outputs" / "degradation_bias_ratio.json"
    if ratio.exists():  # 사전등록 3절: 바이어스 장의 최대/최소 비 실측
        b = json.loads(ratio.read_text(encoding="utf-8"))["holdout"]
        assert ", ".join(f"{b[c]['median']:.2f}" for c in ("0.025", "0.05", "0.1")) in sec

