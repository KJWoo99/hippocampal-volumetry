import numpy as np
import pytest

from hippo.volumetry.volume import compute_volumes, spacing_from_affine, voxel_volume_mm3


def test_spacing_from_affine_identity():
    assert spacing_from_affine(np.eye(4)) == pytest.approx((1.0, 1.0, 1.0))


def test_spacing_from_affine_anisotropic():
    aff = np.diag([2.0, 1.5, 3.0, 1.0])
    assert spacing_from_affine(aff) == pytest.approx((2.0, 1.5, 3.0))


def test_spacing_from_affine_batched():
    aff = np.stack([np.diag([1.0, 1.0, 1.0, 1.0])])  # (1,4,4)
    assert spacing_from_affine(aff) == pytest.approx((1.0, 1.0, 1.0))


def test_voxel_volume_product():
    assert voxel_volume_mm3((1.0, 1.0, 1.0)) == 1.0
    assert voxel_volume_mm3((2.0, 1.0, 0.5)) == 1.0
    assert voxel_volume_mm3((1.5, 1.5, 1.5)) == pytest.approx(3.375)


def test_voxel_volume_rejects_nonpositive():
    with pytest.raises(ValueError):
        voxel_volume_mm3((1.0, 0.0, 1.0))


def test_compute_volumes_counts_known_voxels():
    mask = np.zeros((10, 10, 10), dtype=np.int16)
    mask[:2, :, :] = 1   # anterior: 2*10*10 = 200 voxels
    mask[3:5, :, :] = 2  # posterior: 200 voxels
    res = compute_volumes(mask, (1.0, 1.0, 1.0))
    assert res.per_label_voxels["anterior"] == 200
    assert res.per_label_voxels["posterior"] == 200
    assert res.total_voxels == 400
    assert res.total_mm3 == 400.0


def test_compute_volumes_scales_with_spacing():
    mask = np.zeros((4, 4, 4), dtype=np.int16)
    mask[0, 0, 0] = 1  # 1 voxel
    res = compute_volumes(mask, (2.0, 2.0, 2.0))
    assert res.voxel_volume_mm3 == 8.0
    assert res.total_mm3 == 8.0


def test_compute_volumes_records_space():
    mask = np.zeros((4, 4, 4), dtype=np.int16)
    res = compute_volumes(mask, (1.0, 1.0, 1.0), space="native")
    assert res.space == "native"
    assert res.to_dict()["space"] == "native"


def test_compute_volumes_rejects_non3d():
    with pytest.raises(ValueError):
        compute_volumes(np.zeros((4, 4)), (1.0, 1.0, 1.0))
