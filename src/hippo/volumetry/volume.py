"""분할 마스크 -> 해마 부피(mm^3).

주의: 부피를 어느 공간에서 세느냐가 중요함.
- 1mm 등방 resampling 공간에서 세면 voxel 1개 = 1mm^3 로 편하지만,
- 정확히는 native space로 역변환 후 native spacing으로 세거나 resampling 보정을 해야
  체계적 bias가 없음.
이 함수는 spacing을 명시적으로 받아 '어느 공간에서 측정했는지'를 결과에 박아둠.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

from hippo import LABELS


def voxel_volume_mm3(spacing: tuple[float, ...]) -> float:
    """voxel 1개의 부피(mm^3) = spacing 성분들의 곱."""
    if len(spacing) < 3 or any(s <= 0 for s in spacing[:3]):
        raise ValueError(f"3D spacing(>0)이 필요: {spacing}")
    return float(prod(spacing[:3]))


def spacing_from_affine(affine) -> tuple[float, float, float]:
    """affine 행렬에서 voxel spacing(mm)을 유도함. (매직 상수 대신 메타데이터 사용)

    각 축의 spacing = affine 열벡터의 L2 norm. (B,4,4) 배치도 첫 항목을 씀.
    Spacingd로 1mm resampling한 뒤라면 (1,1,1)이 나오지만, 하드코딩이 아니라
    실제 격자에서 유도하므로 비-1mm 데이터에도 옳음.
    """
    a = np.asarray(affine, dtype=float)
    if a.ndim == 3:  # (B,4,4)
        a = a[0]
    if a.shape[-2:] != (4, 4):
        raise ValueError(f"4x4 affine이 필요: shape={a.shape}")
    return tuple(float(np.linalg.norm(a[:3, i])) for i in range(3))  # type: ignore[return-value]


@dataclass
class VolumeResult:
    voxel_volume_mm3: float
    spacing: tuple[float, float, float]
    space: str                              # 측정 공간 표기 (예: "resampled_1mm" | "native")
    per_label_mm3: dict[str, float]         # {"anterior":.., "posterior":..}
    per_label_voxels: dict[str, int]
    total_mm3: float                        # 전경(해마) 전체 부피
    total_voxels: int

    def to_dict(self) -> dict:
        return {
            "space": self.space,
            "spacing_mm": list(self.spacing),
            "voxel_volume_mm3": self.voxel_volume_mm3,
            "per_label_mm3": self.per_label_mm3,
            "per_label_voxels": self.per_label_voxels,
            "total_mm3": self.total_mm3,
            "total_voxels": self.total_voxels,
        }


def compute_volumes(
    mask: np.ndarray,
    spacing: tuple[float, ...],
    space: str = "resampled_1mm",
    foreground_labels: tuple[int, ...] = (1, 2),
) -> VolumeResult:
    """정수 라벨 마스크(H,W,D)에서 라벨별/전체 부피를 계산.

    Parameters
    ----------
    mask : 정수 라벨 볼륨 (0=배경, 1=anterior, 2=posterior)
    spacing : 이 mask가 놓인 공간의 voxel 간격(mm)
    space : 측정 공간 표기 (리포트에 기록: 재현성/감사용)
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"3D 마스크가 필요: ndim={mask.ndim}")
    vox = voxel_volume_mm3(spacing)

    per_label_voxels: dict[str, int] = {}
    per_label_mm3: dict[str, float] = {}
    total_voxels = 0
    for lab in foreground_labels:
        name = LABELS.get(lab, str(lab))
        cnt = int((mask == lab).sum())
        per_label_voxels[name] = cnt
        per_label_mm3[name] = cnt * vox
        total_voxels += cnt

    return VolumeResult(
        voxel_volume_mm3=vox,
        spacing=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
        space=space,
        per_label_mm3=per_label_mm3,
        per_label_voxels=per_label_voxels,
        total_mm3=total_voxels * vox,
        total_voxels=total_voxels,
    )
