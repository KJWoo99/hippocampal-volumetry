import nibabel as nib
import numpy as np
import pytest

from hippo.data.validation import ValidationError, validate_nifti


def _save(tmp_path, data, affine=None):
    affine = np.eye(4) if affine is None else affine
    p = tmp_path / "x.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(p))
    return str(p)


def test_validate_good_volume(tmp_path):
    data = np.random.rand(20, 20, 20).astype(np.float32)
    info = validate_nifti(_save(tmp_path, data))
    assert info.shape == (20, 20, 20)
    assert info.orientation == "RAS"
    assert info.warnings == []


def test_validate_reports_the_stored_dtype(tmp_path):
    """검증기는 float32 로 바꾼 뒤의 자료형이 아니라 파일에 저장된 자료형을 보고해야 함
    (트러블슈팅 5-b). float32 볼륨만 쓰는 테스트로는 변환 뒤 자료형을 보고해도 구분되지 않음."""
    data = (np.random.rand(20, 20, 20) * 1000).astype(np.int16)
    info = validate_nifti(_save(tmp_path, data))
    assert info.dtype == "int16"


def test_validate_rejects_2d(tmp_path):
    with pytest.raises(ValidationError):
        validate_nifti(_save(tmp_path, np.zeros((20, 20), dtype=np.float32)))


def test_validate_warns_small_volume(tmp_path):
    data = np.random.rand(4, 4, 4).astype(np.float32)
    info = validate_nifti(_save(tmp_path, data), min_size=8)
    assert any("작은 볼륨" in w for w in info.warnings)


def test_validate_warns_constant(tmp_path):
    data = np.ones((20, 20, 20), dtype=np.float32)
    info = validate_nifti(_save(tmp_path, data))
    assert any("단일 상수" in w for w in info.warnings)
