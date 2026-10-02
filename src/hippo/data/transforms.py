"""MONAI 전처리/증강 파이프라인.

흐름: NIfTI 로드 -> 채널정리 -> RAS 방향통일 -> 1mm resampling -> z-score 정규화
      -> (학습) ROI 패딩 + 양/음성 균형 patch 크롭 + flip/rotate 증강.
"""
from __future__ import annotations

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    SpatialPadd,
)

KEYS = ["image", "label"]

# 증강 단계. 약한 것부터 센 것 순.
#
#   basic     flip(축0) + rotate90
#   intensity basic + 강도 증강(잡음, 평활, 밝기, 대비, 바이어스 필드)
#   spatial      intensity 에서 rotate90 을 빼고 작은 각도 회전, 스케일로 교체
#   spatial-wide spatial 의 범위를 두 배로 (회전 +/- 30도, 스케일 +/- 20%)
#   spatial-narrow spatial 의 범위를 절반으로 (회전 +/- 7.5도, 스케일 +/- 5%). 절제 #14
#   spatial-always spatial 과 범위는 같고 회전, 스케일을 매 샘플에 적용함(확률 0.5 -> 1.0). 절제 #15
#   spatial-geo  spatial 에서 강도 증강(잡음, 평활, 밝기, 대비, 바이어스필드)을 뺀 것. 절제 #16
#   spatial-wide-always spatial-wide 와 범위는 같고 회전, 스케일을 매 샘플에 적용함. 절제 #18(부모 #11)
#   spatial-wide-geo    spatial-wide 에서 강도 증강을 뺀 것. 절제 #19(부모 #11)
#
# RandRotate90 은 이 도메인에서 나올 수 없는 영상을 만듦. 라벨이 해마의 전방, 후방이라 90도 회전은 그 축을
# 다른 축으로 옮김. 그래서 spatial 계열은 실제로 생길 수 있는 정도(+/- 15도, +/- 10% 스케일)로 바꿈.
#
# 강도 증강은 MRI 에서 스캐너, 프로토콜 차이가 주로 강도로 나타나기 때문에 따로 둠.
AUG_LEVELS = ("basic", "intensity", "spatial", "spatial-wide", "spatial-narrow", "spatial-always",
              "spatial-geo", "spatial-wide-always", "spatial-wide-geo")


def get_transforms(roi: tuple[int, int, int], crop_samples: int = 1,
                   aug: str = "basic"):
    """(train_transform, val_transform) 반환.

    Parameters
    ----------
    roi : 학습 패치 크기.
    crop_samples : RandCropByPosNegLabeld 가 볼륨당 만들 샘플 수.
        roi 가 볼륨 전체를 덮는 경우(현 Task04 + 64^3) 크롭은 no-op 이므로 1 로 둠.
        2 이상은 동일 샘플을 중복 생성해 계산만 낭비함.
    aug : `AUG_LEVELS` 중 하나. 절제실험에서 바꿔가며 봄.

    Notes
    -----
    val_t 에는 증강을 넣지 않음. 증강은 분할 이후 Dataset 단계에서만 적용되므로
    같은 원본의 변형본이 train/val 양쪽에 존재하는 근친 누수는 구조적으로 발생하지 않음.
    """
    base = [
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        Orientationd(keys=KEYS, axcodes="RAS"),
        Spacingd(keys=KEYS, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    spatial = [
        *base,
        # 해마 볼륨은 작아 대부분 ROI보다 작다 -> 크롭 전 ROI 크기로 패딩
        SpatialPadd(keys=KEYS, spatial_size=roi),
        RandCropByPosNegLabeld(
            keys=KEYS, label_key="label",
            spatial_size=roi, pos=1, neg=1, num_samples=max(1, crop_samples),
            image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
    ]
    if aug not in AUG_LEVELS:
        raise ValueError(f"모르는 증강 단계: {aug!r} (가능: {AUG_LEVELS})")
    if aug in ("basic", "intensity"):
        spatial.append(RandRotate90d(keys=KEYS, prob=0.5, max_k=3))
    else:  # spatial 계열: 도메인에서 실제로 생길 수 있는 범위로만 흔들어 줌
        # 범위를 두 배(wide)와 절반(narrow)으로 바꿔 정점이 어디인지 봄.
        rot, sc = {"spatial-wide": (0.52, 0.2),     # 약 +/- 30도, +/- 20%
                   "spatial": (0.26, 0.1),          # 약 +/- 15도, +/- 10%
                   "spatial-narrow": (0.13, 0.05),  # 약 +/- 7.5도, +/- 5%
                   "spatial-always": (0.26, 0.1),
                   "spatial-geo": (0.26, 0.1),
                   # 쓰는 설정 #11(spatial-wide) 위에서 #15, #16 과 같은 변경을 적용함(docs/prereg_pair_split.md).
                   "spatial-wide-always": (0.52, 0.2),
                   "spatial-wide-geo": (0.52, 0.2),
                   }[aug]
        spatial.append(RandAffined(
            # 범위를 좁히면 조금 낮고 넓혀도 같아(#14, #11) 범위가 아니라 빈도를 봄(#15).
            keys=KEYS, prob=1.0 if aug in ("spatial-always", "spatial-wide-always") else 0.5,
            rotate_range=(rot, rot, rot),
            scale_range=(sc, sc, sc),
            mode=("bilinear", "nearest"), padding_mode="zeros",
        ))
    if aug in ("intensity", "spatial", "spatial-wide", "spatial-narrow", "spatial-always", "spatial-wide-always"):
        # 라벨에는 걸지 않음: 강도 변형은 영상에만 해당함.
        spatial += [
            RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.05),
            RandGaussianSmoothd(keys="image", prob=0.15,
                                sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            RandScaleIntensityd(keys="image", factors=0.2, prob=0.3),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
            RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.7, 1.5)),
            # MRI 의 대표적 아티팩트. 스캐너가 바뀌면 실제로 나타남.
            RandBiasFieldd(keys="image", prob=0.15, coeff_range=(0.0, 0.1)),
        ]
    train_t = Compose([*spatial, EnsureTyped(keys=KEYS)])
    # 검증/평가도 동일 ROI 로 패딩해 sliding-window 추론이 단일 패스로 끝나게 함.
    val_t = Compose([*base, SpatialPadd(keys=KEYS, spatial_size=roi), EnsureTyped(keys=KEYS)])
    return train_t, val_t


def post_transforms(num_classes: int):
    """추론 후처리: (pred 후처리, label 후처리)."""
    post_pred = Compose([
        Activations(softmax=True),
        AsDiscrete(argmax=True, to_onehot=num_classes),
    ])
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])
    return post_pred, post_label
