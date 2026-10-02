"""설정이 두 곳에 있으므로 둘이 어긋나지 않는지 봄.

`src/hippo/config.py` 의 데이터클래스 기본값과 `configs/default.yaml` 은 같은
설정을 두 번 적은 것임. 스크립트는 YAML 을 명시적으로 넘기고, CLI 는 YAML 이
없으면 데이터클래스 값을 씀. 그래서 둘이 달라지면 같은 저장소에서 같은
명령을 다르게 부르면 다른 모델이 도는 상태가 됨.

한쪽으로 합치는 것이 더 깔끔하지만, 데이터클래스는 타입과 기본값을 주는 자리라
없앨 수 없고 YAML 은 사람이 고치는 자리라 없앨 수 없음. 그래서 남겨두고 대조를
테스트로 강제함.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "configs" / "default.yaml"


@pytest.mark.skipif(not YAML.exists(), reason="configs/default.yaml 없음")
def test_dataclass_defaults_match_default_yaml():
    from omegaconf import OmegaConf

    from hippo.config import Config

    base = OmegaConf.structured(Config)
    merged = OmegaConf.merge(base, OmegaConf.load(YAML))

    # YAML 이 값을 덮어쓴 자리를 찾음. 덮어쓴 것이 있다는 말은 두 기본값이
    # 다르다는 뜻임.
    diffs = []

    def walk(a, b, path=""):
        # 사전만 따라 내려감. 리스트(roi 등)는 값으로 비교함: 리스트를 키로 돌면
        # 원소 값으로 색인하게 돼 엉뚱한 오류가 남.
        for k in b:
            va, vb = a[k], b[k]
            if OmegaConf.is_dict(vb):
                walk(va, vb, f"{path}{k}.")
                continue
            if OmegaConf.is_list(vb):
                va, vb = list(va), list(vb)
            if va != vb:
                diffs.append(f"{path}{k}: config.py={va!r} yaml={vb!r}")

    walk(base, merged)
    assert not diffs, (
        "config.py 기본값과 configs/default.yaml 이 다르다. CLI 를 --config 없이 "
        "부르면 앞의 값이, 스크립트에서는 뒤의 값이 쓰인다:\n  " + "\n  ".join(diffs))
