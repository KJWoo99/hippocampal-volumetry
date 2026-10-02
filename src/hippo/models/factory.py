"""모델 팩토리: 이름으로 3D 분할 네트워크를 생성함.

아키텍처 벤치마크(UNet vs SegResNet vs SwinUNETR)를 위해 단일 진입점으로 통일.

입력 크기는 세 아키텍처 공통으로 64^3 을 씀. 메모리 때문이 아니라 데이터와
구조 때문임: 학습 볼륨 260개의 최대 단일 축이 59 라 64^3 이면 전 볼륨이
잘림 없이 들어가고, SwinUNETR 는 32배 축소 구조여서 32^3 에서는 1x1x1 로 붕괴함.
더 키워도 패딩만 늘고 정보는 늘지 않음.
"""
from __future__ import annotations

import torch.nn as nn
from monai.networks.nets import UNet

_MODELS = ("unet", "segresnet", "swinunetr")


def list_models() -> tuple[str, ...]:
    return _MODELS


def build_model(
    name: str = "unet",
    in_channels: int = 1,
    out_channels: int = 3,
    roi: tuple[int, int, int] = (32, 32, 32),
) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
        )
    if name == "segresnet":
        from monai.networks.nets import SegResNet
        return SegResNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            init_filters=16, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
        )
    if name == "swinunetr":
        from monai.networks.nets import SwinUNETR
        # monai >=1.5 에서 img_size 인자가 제거됨(입력 크기에 동적으로 적응).
        # roi는 더 이상 필요 없지만 다른 모델과 호출부를 통일하기 위해 인자로는 유지.
        return SwinUNETR(
            in_channels=in_channels, out_channels=out_channels,
            feature_size=24, use_checkpoint=True,
        )
    raise ValueError(f"알 수 없는 모델 '{name}'. 지원: {_MODELS}")
