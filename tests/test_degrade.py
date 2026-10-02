"""화질 저하 모사와 복원 검증(사전등록 `docs/prereg_degradation.md` 3, 4절).

저하가 뜻한 대로 걸리지 않으면 뒤의 부피 비교가 엉뚱한 영상 위에서 나옴. 합성 볼륨으로 크기, 방향, 결정성을 고정함.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hippo.degrade import (
    LEVELS,
    bias_field,
    case_seed,
    cubic_from_blocks,
    degrade,
    psnr3d,
    rician,
    thick_blocks,
)


def _phantom(shape=(32, 40, 36)):
    """가운데가 밝은 매끈한 볼륨(원래 세기 100~300)."""
    g = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    return 100.0 + 200.0 * np.exp(-(g[0] ** 2 + g[1] ** 2 + g[2] ** 2))


def test_rician_zero_level_is_identity_and_noise_scales_with_f():
    x = _phantom()
    assert np.allclose(rician(x, 0.0, np.random.default_rng(0)), x)
    hi = np.full((40, 40, 40), 500.0)  # 신호가 크면 Rician 은 가우시안에 가까움
    for f in LEVELS["noise"]:
        y = rician(hi, f, np.random.default_rng(1))
        assert y.std() == pytest.approx(f * 500.0, rel=0.05)


def test_bias_field_is_positive_smooth_and_grows_with_c():
    ratios = []
    for c in LEVELS["bias"]:
        f = bias_field((20, 24, 22), c, np.random.default_rng(3))
        assert (f > 0).all()
        assert np.abs(np.diff(f, axis=0)).max() < 0.5 * f.mean()  # 매끈함
        ratios.append(f.max() / f.min())
    assert ratios[0] < ratios[1] < ratios[2]


def test_thick_blocks_average_and_keep_constant_volumes():
    x = np.arange(8, dtype=float)[None, None, :].repeat(2, 0).repeat(3, 1)
    means, centers = thick_blocks(x, 2)
    assert np.allclose(means[0, 0], [0.5, 2.5, 4.5, 6.5])
    assert np.allclose(centers, [0.5, 2.5, 4.5, 6.5])
    means3, centers3 = thick_blocks(x, 3)          # 마지막 블록은 남은 두 장만 평균
    assert np.allclose(means3[0, 0], [1.0, 4.0, 6.5]) and np.allclose(centers3, [1.0, 4.0, 6.5])
    flat = np.full((10, 12, 30), 7.0)
    for t in LEVELS["thick"]:
        assert np.allclose(degrade(flat, "thick", t, 0)["image"], 7.0)


def test_thick_loses_detail_and_cubic_keeps_shape():
    x = _phantom()
    for t in LEVELS["thick"]:
        dg = degrade(x, "thick", t, 0)
        assert dg["image"].shape == x.shape
        assert psnr3d(dg["image"], x) < float("inf")
        assert cubic_from_blocks(*dg["blocks"], x.shape[2]).shape == x.shape


def test_degrade_is_deterministic_per_case_and_rejects_unknown_kind():
    x = _phantom()
    s = case_seed("hippocampus_001.nii.gz", "noise", 0.06)
    assert s == case_seed("hippocampus_001.nii.gz", "noise", 0.06)
    assert s != case_seed("hippocampus_002.nii.gz", "noise", 0.06)
    a = degrade(x, "noise", 0.06, s)["image"]
    b = degrade(x, "noise", 0.06, s)["image"]
    assert np.array_equal(a, b)
    assert degrade(x, "bias", 0.4, 5)["field_ratio"] > 1
    with pytest.raises(ValueError):
        degrade(x, "motion", 1, 0)


def test_psnr_is_infinite_for_identical_and_drops_with_noise():
    x = _phantom()
    assert psnr3d(x, x) == float("inf")
    lo = psnr3d(degrade(x, "noise", 0.10, 1)["image"], x)
    hi = psnr3d(degrade(x, "noise", 0.03, 1)["image"], x)
    assert lo < hi


def test_nlm_reduces_noise_error():
    pytest.importorskip("skimage")
    from hippo.degrade import nlm3d

    x = _phantom()
    y = degrade(x, "noise", 0.06, 2)["image"]
    assert psnr3d(nlm3d(y, 1.0), x) > psnr3d(y, x)


def test_n4_flattens_a_multiplicative_field():
    pytest.importorskip("SimpleITK")
    from hippo.degrade import n4

    x = np.full((24, 28, 26), 200.0)
    x[:, :, :4] = 0.0  # 배경이 있어야 Otsu 마스크가 뇌 영역을 고름
    y = degrade(x, "bias", 0.6, 4)["image"]
    fg = x > 0
    cv = lambda a: a[fg].std() / a[fg].mean()  # noqa: E731
    assert cv(n4(y)) < 0.5 * cv(y)


def test_unet_lr_is_reduced_before_early_stop():
    # 3D U-Net 복원기: 학습률을 줄일 기회가 조기종료보다 먼저 와야 함.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_degradation.py"
    spec = importlib.util.spec_from_file_location("run_degradation", path)
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    cfg = rd.UNET
    assert 0 < cfg["lr_factor"] < 1 and cfg["lr_min"] < cfg["lr"]
    assert cfg["lr_patience"] < cfg["patience"] < cfg["max_epochs"]


def test_nlm_wide_grid_and_json_numpy(tmp_path):
    # 넓힌 NLM 격자(사전등록 이탈, 보조)는 원안 값을 모두 포함하고 양쪽이 더 넓어야 하며,
    # 결과 저장은 numpy 숫자가 섞여도 멈추지 않아야 함.
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_degradation.py"
    spec = importlib.util.spec_from_file_location("run_degradation", path)
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    assert set(rd.NLM_GRID) <= set(rd.NLM_GRID_WIDE)
    assert min(rd.NLM_GRID_WIDE) < min(rd.NLM_GRID) and max(rd.NLM_GRID_WIDE) > max(rd.NLM_GRID)
    assert rd.NLM_GRIDS["prereg"] == ("", rd.NLM_GRID)
    out = tmp_path / "r.json"
    rd._save_json(out, {"a": np.float32(0.5), "b": np.int64(2)})
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 0.5, "b": 2}


# ── 짝 단위 분할 뒤의 부가 분석(docs/prereg_pair_split.md, 결과 보기 전 추가) ──────────
_DEG_SRC = (Path(__file__).resolve().parents[1] / "scripts" / "run_degradation.py").read_text(encoding="utf-8")


def test_degradation_uses_the_benchmark_selected_arch():
    # 기준 모델 경로에 구조 이름을 박아 두면 벤치마크 선택이 바뀌었을 때 배포와 다른 모델을 측정함.
    code = "\n".join(ln for ln in _DEG_SRC.splitlines() if not ln.strip().startswith(("#", '"', "예전에는")))
    assert '"segresnet"' not in code
    assert '["selected_arch"]' in _DEG_SRC


def _deg_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_degradation",
                                                  Path(__file__).resolve().parents[1] / "scripts" / "run_degradation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy(ids):
    return [{"image": f"imagesTr/hippocampus_{i:03d}.nii.gz"} for i in ids]


def test_degradation_bootstrap_resamples_subjects_not_cases():
    # 홀드아웃에 같은 사람의 좌우가 함께 들어 있어(16쌍) 케이스 단위로 뽑으면 구간이 좁아짐.
    pytest.importorskip("monai")
    mod = _deg_module()
    items = _toy([1, 2, 3, 4, 5, 9, 10, 13])            # 짝 (1,2), (3,4), (9,10) 과 혼자 5, 13
    units = mod._subject_units(items)
    assert sorted(map(sorted, units)) == [[0, 1], [2, 3], [4], [5, 6], [7]]
    rng = np.random.default_rng(0)
    for _ in range(200):
        ix = list(mod._subject_resample(units, rng))
        for a, b in ([0, 1], [2, 3], [5, 6]):
            assert ix.count(a) == ix.count(b)            # 짝은 늘 함께 뽑힘
    assert "_subject_resample(unit_list, rng)" in _DEG_SRC


def test_unet_validation_pick_keeps_subjects_whole():
    pytest.importorskip("monai")
    mod = _deg_module()
    items = _toy([i for i in range(1, 90) if i % 7])      # 7 의 배수를 빼 혼자인 피험자를 여럿 만듦(실제 개발셋처럼)
    pick = mod._subject_pick(items, 20, 42)
    assert len(pick) == 20
    keys = lambda idx: {(int(items[i]["image"][-10:-7]) + 1) // 2 for i in idx}  # noqa: E731
    rest = [i for i in range(len(items)) if i not in set(pick)]
    assert not (keys(pick) & keys(rest))
    assert pick == mod._subject_pick(items, 20, 42)
