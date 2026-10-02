"""transform 회귀 테스트.

핵심 1: ROI보다 작은 볼륨도 SpatialPadd 로 크롭이 성공해야 함.
        (ROI 보다 작은 볼륨이 있으면 RandCropByPosNegLabeld 가 멈춤)
핵심 2: 증강은 train_t 에만 있고 val_t 에는 없어야 함.
        분할 이후 Dataset 단계에서만 적용되므로 근친 누수가 구조적으로 불가능함.
"""
import nibabel as nib
import numpy as np
import pytest

from hippo.data.transforms import get_transforms


def _save(tmp_path, name, data):
    p = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
    return str(p)


def _sample(tmp_path, shape=(34, 56, 31)):
    img = np.random.rand(*shape).astype(np.float32)
    lab = np.zeros(shape, dtype=np.float32)
    lab[10:20, 10:30, 5:15] = 1
    return {"image": _save(tmp_path, "img.nii.gz", img),
            "label": _save(tmp_path, "lab.nii.gz", lab)}


@pytest.mark.parametrize("roi", [(32, 32, 32), (64, 64, 64)])
def test_train_transform_handles_volume_smaller_than_roi(tmp_path, roi):
    """ROI 보다 작은 축이 있어도 SpatialPadd 로 처리되어 ROI 크기 패치가 나와야 함.

    64^3 도 함께 검사함. Task04 볼륨의 최대 단일 축이 59 라 64^3 에서는
    모든 볼륨이 패딩 대상이 되기 때문임.
    """
    train_t, _ = get_transforms(roi)
    out = train_t(_sample(tmp_path))
    assert isinstance(out, list)
    for patch in out:
        assert tuple(patch["image"].shape[1:]) == roi
        assert tuple(patch["label"].shape[1:]) == roi


@pytest.mark.parametrize("crop_samples", [1, 2, 4])
def test_crop_samples_controls_patch_count(tmp_path, crop_samples):
    """crop_samples 가 볼륨당 생성 패치 수를 결정함.

    ROI 가 볼륨 전체를 덮는 경우(현 Task04 + 64^3) 크롭은 no-op 이므로
    기본값을 1 로 두어 동일 샘플 중복 생성을 막음.
    """
    train_t, _ = get_transforms((32, 32, 32), crop_samples=crop_samples)
    out = train_t(_sample(tmp_path))
    assert len(out) == crop_samples


def test_val_transform_keeps_channel_first(tmp_path):
    _, val_t = get_transforms((32, 32, 32))
    out = val_t(_sample(tmp_path, shape=(40, 40, 40)))
    assert out["image"].shape[0] == 1  # channel-first


def test_val_transform_pads_to_roi(tmp_path):
    """검증셋도 ROI 로 패딩되어야 sliding-window 추론이 단일 패스로 끝남."""
    roi = (64, 64, 64)
    _, val_t = get_transforms(roi)
    out = val_t(_sample(tmp_path, shape=(34, 56, 31)))
    for axis, size in enumerate(roi):
        assert out["image"].shape[1 + axis] >= size


def test_val_transform_has_no_random_augmentation(tmp_path):
    """누수 방지: val/test 에 증강이 들어가면 평가가 오염됨.

    같은 입력을 두 번 통과시켜 결과가 동일한지(결정적인지) 확인함.
    """
    _, val_t = get_transforms((64, 64, 64))
    s = _sample(tmp_path, shape=(40, 40, 40))
    a = val_t(s)["image"]
    b = val_t(s)["image"]
    assert np.allclose(np.asarray(a), np.asarray(b))


# --- monai 업그레이드 안전장치 ---

def test_orientation_labels_default_change_is_behaviour_preserving():
    """monai 1.6.0 에서 Orientation 의 labels 기본값이 None 으로 바뀜.

    경고만 뜨고 넘어가면 전처리가 그대로 달라져 학습된 가중치와 어긋날 수 있음.
    axcodes="RAS" 로 쓰는 이 프로젝트에서 옛 기본값(명시)과 새 기본값(생략)의
    결과가 같은지 고정해, 이후 업그레이드에서 실제로 동작이 바뀌면 잡아냄.
    """
    import nibabel as nib
    import numpy as np
    import torch
    from monai.data import MetaTensor
    from monai.transforms import Orientation

    labels_before_1_6 = (("L", "R"), ("P", "A"), ("I", "S"))
    rng = np.random.default_rng(0)

    for src in ("RAS", "LPS", "LAS", "RPI", "PSR", "AIL"):
        arr = rng.normal(size=(1, 8, 9, 10)).astype(np.float32)
        aff = np.linalg.inv(
            nib.orientations.inv_ornt_aff(
                nib.orientations.axcodes2ornt(src), (8, 9, 10)
            )
        )
        vol = MetaTensor(torch.from_numpy(arr),
                         affine=torch.as_tensor(aff, dtype=torch.float64))

        old = Orientation(axcodes="RAS", labels=labels_before_1_6)(vol.clone())
        new = Orientation(axcodes="RAS")(vol.clone())

        assert torch.allclose(torch.as_tensor(np.asarray(old)),
                              torch.as_tensor(np.asarray(new))), f"{src}: 데이터가 달라짐"
        assert torch.allclose(old.affine.double(), new.affine.double()), f"{src}: affine 이 달라짐"


def _affine(aug):
    from monai.transforms import RandAffined
    train_t, _ = get_transforms((64, 64, 64), aug=aug)
    found = [x for x in train_t.transforms if isinstance(x, RandAffined)]
    return found[0] if found else None


def _names(aug):
    train_t, _ = get_transforms((64, 64, 64), aug=aug)
    return [type(x).__name__ for x in train_t.transforms]


def test_spatial_levels_keep_their_ranges():
    """절제 #7, #11 의 증강이 새 단계를 더한 뒤에도 그대로인지 봄.

    기존 단계의 범위가 바뀌면 이미 보고한 회차를 다시 돌려도 같은 설정이 아니게 됨.
    """
    ranges = {aug: (_affine(aug).rand_affine.rand_affine_grid.rotate_range[0],
                    _affine(aug).rand_affine.rand_affine_grid.scale_range[0])
              for aug in ("spatial", "spatial-wide", "spatial-narrow")}
    assert ranges["spatial"] == (0.26, 0.1)
    assert ranges["spatial-wide"] == (0.52, 0.2)
    assert ranges["spatial-narrow"] == (0.13, 0.05)


def test_spatial_narrow_differs_from_spatial_only_in_range():
    # 한 번에 한 축만 바꾼다는 절제 규칙: 회전, 스케일 범위 말고는 spatial 과 같아야 함.
    assert _names("spatial-narrow") == _names("spatial")
    assert "RandRotate90d" not in _names("spatial-narrow")


def test_spatial_always_differs_from_spatial_only_in_probability():
    # 절제 #15: 범위(+/- 15도, +/- 10%)와 나머지 변환은 spatial 과 같고, 기하 변형 확률만 1.0 임.
    a, s = _affine("spatial-always"), _affine("spatial")
    assert _names("spatial-always") == _names("spatial")
    assert (a.prob, s.prob) == (1.0, 0.5)
    grid_a, grid_s = a.rand_affine.rand_affine_grid, s.rand_affine.rand_affine_grid
    assert (grid_a.rotate_range, grid_a.scale_range) == (grid_s.rotate_range, grid_s.scale_range)


def test_spatial_geo_is_spatial_without_intensity():
    # 절제 #16: 기하 변형은 spatial 그대로 두고 강도 증강 여섯 가지만 뺌.
    intensity = {"RandGaussianNoised", "RandGaussianSmoothd", "RandScaleIntensityd",
                 "RandShiftIntensityd", "RandAdjustContrastd", "RandBiasFieldd"}
    assert [n for n in _names("spatial") if n not in intensity] == _names("spatial-geo")
    assert intensity <= set(_names("spatial"))
    g, s = _affine("spatial-geo"), _affine("spatial")
    assert g.prob == s.prob == 0.5
    assert g.rand_affine.rand_affine_grid.rotate_range == s.rand_affine.rand_affine_grid.rotate_range


def test_spatial_wide_always_differs_from_spatial_wide_only_in_probability():
    # 절제 #18(부모 #11): 범위(+/- 30도, +/- 20%)와 나머지 변환은 spatial-wide 그대로, 기하 변형 확률만 1.0 임.
    a, w = _affine("spatial-wide-always"), _affine("spatial-wide")
    assert _names("spatial-wide-always") == _names("spatial-wide")
    assert (a.prob, w.prob) == (1.0, 0.5)
    grid_a, grid_w = a.rand_affine.rand_affine_grid, w.rand_affine.rand_affine_grid
    assert (grid_a.rotate_range, grid_a.scale_range) == (grid_w.rotate_range, grid_w.scale_range)


def test_spatial_wide_geo_is_spatial_wide_without_intensity():
    # 절제 #19(부모 #11): 기하 변형은 spatial-wide 그대로 두고 강도 증강 여섯 가지만 뺌.
    intensity = {"RandGaussianNoised", "RandGaussianSmoothd", "RandScaleIntensityd",
                 "RandShiftIntensityd", "RandAdjustContrastd", "RandBiasFieldd"}
    assert [n for n in _names("spatial-wide") if n not in intensity] == _names("spatial-wide-geo")
    g, w = _affine("spatial-wide-geo"), _affine("spatial-wide")
    assert g.prob == w.prob == 0.5
    grid_g, grid_w = g.rand_affine.rand_affine_grid, w.rand_affine.rand_affine_grid
    assert (grid_g.rotate_range, grid_g.scale_range) == (grid_w.rotate_range, grid_w.scale_range)
