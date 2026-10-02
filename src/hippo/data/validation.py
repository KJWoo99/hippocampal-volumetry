"""입력 NIfTI 검증: 틀린 입력이 오류 없이 결과까지 가지 않게 막음.

추론 전에 입력이 가정과 맞는지 확인하고, 어긋나면 명확히 실패시킴.
(개발 환경 != 운영 환경에서 생기는 문제를 입구에서 차단)
"""
from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np


class ValidationError(ValueError):
    """입력이 추론 가정을 위반할 때."""


@dataclass
class NiftiInfo:
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    dtype: str
    orientation: str
    intensity_min: float
    intensity_max: float
    warnings: list[str]


def validate_nifti(path: str, min_size: int = 8, max_spacing_mm: float = 5.0) -> NiftiInfo:
    """NIfTI를 로드해 기본 정합성을 검사하고 메타정보를 반환.

    하드 실패(ValidationError): 3D 아님, 비유한 값, 0/음수 spacing.
    소프트 경고(warnings 리스트): 너무 작은 볼륨, 비정상적으로 큰 spacing, 단일 강도값.
    """
    try:
        img = nib.load(path)
    except Exception as e:  # noqa: BLE001
        raise ValidationError(f"NIfTI 로드 실패: {path} ({e})") from e

    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim not in (3, 4):
        raise ValidationError(f"3D(또는 4D) 볼륨이 아님: ndim={data.ndim}")
    vol = data[..., 0] if data.ndim == 4 else data

    if not np.isfinite(vol).all():
        raise ValidationError("볼륨에 NaN/Inf가 있음: 손상된 입력")

    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    if any(s <= 0 for s in spacing):
        raise ValidationError(f"spacing에 0/음수 값: {spacing}")

    orientation = "".join(nib.aff2axcodes(img.affine))

    warnings: list[str] = []
    if min(vol.shape) < min_size:
        warnings.append(f"매우 작은 볼륨 (min dim {min(vol.shape)} < {min_size})")
    if max(spacing) > max_spacing_mm:
        warnings.append(f"비정상적으로 큰 spacing {spacing} (> {max_spacing_mm}mm)")
    if float(vol.min()) == float(vol.max()):
        warnings.append("강도값이 단일 상수: 빈/손상 가능성")

    return NiftiInfo(
        shape=tuple(vol.shape),
        spacing=spacing,
        # 파일에 저장된 dtype 을 보고함. 위에서 float32 로 강제 변환했으므로
        # `data.dtype` 을 쓰면 무엇을 넣든 항상 'float32' 가 나와, 검증기가
        # 정작 확인해야 할 원본 자료형을 감춤.
        dtype=str(img.get_data_dtype()),
        orientation=orientation,
        intensity_min=float(vol.min()),
        intensity_max=float(vol.max()),
        warnings=warnings,
    )
