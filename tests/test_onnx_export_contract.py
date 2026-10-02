"""ONNX 내보내기가 약속한 것을 실제로 지키는지 고정.

세 가지를 잡음.

1. opset 이 문서와 같은가: torch 2.14 의 dynamo exporter 는 17 을 요구해도 18 로 만든 뒤
   17 변환에 실패하고(`No Adapter To Version 17 for Pad`) 경고만 남긴 채 18 로 둠.

2. parity 판정이 재현되는가: 매 호출마다 새 난수 입력을 쓰면 CUDA 에서 실행마다 판정이
   뒤집힘(max_abs_diff 0.0088~0.0113).

3. parity 실패가 실제로 실패로 처리되는가: 출력만 하고 넘어가면 깨진 ONNX 로 TensorRT
   엔진까지 만들어 exit 0 으로 끝남.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from hippo.config import load_config
from hippo.export.to_onnx import OPSET, export_onnx, verify_onnx_parity
from hippo.models import build_model

REPO = Path(__file__).resolve().parents[1]
ROI = (64, 64, 64)


@pytest.fixture(scope="module")
def onnx_file(tmp_path_factory):
    net = build_model("unet", 1, 3, ROI).eval()
    path = tmp_path_factory.mktemp("onnx") / "m.onnx"
    export_onnx(net, str(path), ROI, 1, "cpu")
    return net, str(path)


def test_exported_opset_matches_the_declared_one(onnx_file):
    """실제 산출물의 opset 이 코드가 요구한 값과 같아야 함."""
    onnx = pytest.importorskip("onnx")
    _, path = onnx_file
    model = onnx.load(path, load_external_data=False)
    actual = {i.domain or "ai.onnx": i.version for i in model.opset_import}
    assert actual["ai.onnx"] == OPSET, (
        f"요구 opset {OPSET} 인데 산출물은 {actual['ai.onnx']} 다: "
        "torch 가 경고만 남기고 다른 버전으로 내보냈다"
    )


def test_docs_state_the_actual_opset():
    """문서가 실제 opset 을 적고 있어야 함."""
    wrong = []
    for name in ("README.md", "docs/report.md", "docs/TROUBLESHOOTING.md", "docs/DESIGN.md", "docs/MODEL_CARD.md"):
        p = REPO / name
        if not p.exists():
            continue
        for m in re.finditer(r"opset\s*(\d+)", p.read_text(encoding="utf-8"), re.I):
            if int(m.group(1)) != OPSET:
                wrong.append(f"{name}: opset{m.group(1)} (실제 {OPSET})")
    assert not wrong, "문서가 틀린 opset 을 적고 있다:\n  " + "\n  ".join(wrong)


def test_parity_is_reproducible(onnx_file):
    """같은 입력이면 같은 판정이 나와야 함: 판정이 흔들리면 검증이 아님."""
    net, path = onnx_file
    runs = [verify_onnx_parity(net, path, ROI, 1, "cpu") for _ in range(3)]
    diffs = {round(r["max_abs_diff"], 10) for r in runs}
    assert len(diffs) == 1, f"실행마다 값이 달라진다: {diffs}"
    assert {r["ok"] for r in runs} == {True}


def test_parity_uses_multiple_inputs(onnx_file):
    """입력 하나로 판정하면 운에 좌우됨."""
    _, path = onnx_file
    net, _ = onnx_file
    assert verify_onnx_parity(net, path, ROI, 1, "cpu")["n_trials"] >= 2


def test_export_command_aborts_when_parity_fails(monkeypatch, tmp_path):
    """parity 실패를 출력만 하고 넘어가면 안 됨."""
    import typer

    import hippo.export.to_onnx as to_onnx
    from hippo import cli

    monkeypatch.setattr(
        to_onnx, "verify_onnx_parity",
        lambda *a, **k: {"max_abs_diff": 9.9, "seg_match": 0.5, "ok": False},
    )
    monkeypatch.setattr(to_onnx, "export_onnx", lambda *a, **k: str(tmp_path / "x.onnx"))
    monkeypatch.setattr(cli, "_device", lambda: torch.device("cpu"))

    cfg = load_config(str(REPO / "configs" / "default.yaml"))
    cfg.out_dir = str(tmp_path)
    monkeypatch.setattr(cli, "_cfg", lambda *a, **k: cfg)

    with pytest.raises(typer.Exit) as exc:
        cli.export(config=None, onnx=True, tensorrt=False)
    assert exc.value.exit_code == 1
