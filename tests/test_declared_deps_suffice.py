"""선언한 의존성만으로 기본 설정이 실제로 도는지 검증.

왜 필요한가
-----------
SwinUNETR 은 einops 를 쓰는데, monai 가 optional import 로 감싸고 있어 선언이 빠져도
모델을 실제로 만드는 순간에야 터짐. 모델 이름을 문자열로만 넘기는 테스트는 이걸 못 잡으므로,
여기서는 기본 설정으로 세 아키텍처를 실제로 만들고 순전파까지 시킴.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from hippo.config import load_config
from hippo.models import build_model, list_models

REPO = Path(__file__).resolve().parents[1]


# 세 아키텍처를 함께 돌릴 수 있는 최소 입력. SwinUNETR 는 32배 축소 구조라
# 32^3 을 넣으면 1x1x1 로 붕괴해 InstanceNorm 이 실패함(config.py ROI 주석).
MIN_ROI = (64, 64, 64)


@pytest.mark.parametrize("name", list_models())
def test_every_model_can_actually_be_built_and_run(name):
    """이름만 아는 것으로는 부족함: 만들어서 돌려봐야 의존성 누락이 드러남.

    `einops` 누락이 여기서 걸림. monai 가 optional import 로 감싸고 있어
    임포트 시점에는 조용하고, 모델을 생성하는 순간에야 터지기 때문임.
    """
    net = build_model(name, in_channels=1, out_channels=3, roi=MIN_ROI).eval()
    with torch.no_grad():
        out = net(torch.randn(1, 1, *MIN_ROI))
    assert out.shape == (1, 3, *MIN_ROI), out.shape


def test_default_config_model_is_buildable():
    """기본 설정(configs/default.yaml)의 모델이 실제로 만들어져야 함."""
    cfg = load_config(str(REPO / "configs" / "default.yaml"))
    roi = tuple(cfg.model.roi)
    net = build_model(cfg.model.name, cfg.model.in_channels,
                      cfg.model.out_channels, roi).eval()
    with torch.no_grad():
        out = net(torch.randn(1, cfg.model.in_channels, *roi))
    assert out.shape[1] == cfg.model.out_channels


def test_swinunetr_needs_at_least_64_cubed():
    """32^3 에서 붕괴한다는 사실을 고정함.

    `config.py` 의 ROI 주석이 이 근거로 64^3 을 정했다고 적고 있음. 주석이
    맞는지 실제로 확인해 두지 않으면, 나중에 누군가 ROI 를 줄였을 때 왜
    깨지는지 알 수 없음.
    """
    net = build_model("swinunetr", 1, 3, (32, 32, 32)).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="spatial element"):
        net(torch.randn(1, 1, 32, 32, 32))


def test_einops_is_declared():
    """SwinUNETR 의 숨은 의존성이 선언돼 있어야 함.

    monai 가 optional import 로 감싸므로 설치가 빠져도 임포트 시점에는
    조용함. 선언에서 빠지면 남이 clone 해서 기본 설정으로 돌릴 수 없음.
    """
    req = (REPO / "requirements.txt").read_text(encoding="utf-8")
    declared = {
        line.split("==")[0].split(">=")[0].strip().lower()
        for line in req.splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert "einops" in declared, "requirements.txt 에 einops 가 없다"

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'"einops"', pyproject), "pyproject.toml 에 einops 가 없다"
