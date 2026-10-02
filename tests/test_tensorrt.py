"""TensorRT FP16 엔진 빌드 + 런타임 추론 검증.

tensorrt 미설치 또는 CUDA 없으면 자동 skip (CI/일반 PC). GPU+tensorrt 환경에서만 실행.
"""
import numpy as np
import pytest
import torch

from hippo.export.to_tensorrt import tensorrt_available

pytestmark = pytest.mark.skipif(
    not (tensorrt_available() and torch.cuda.is_available()),
    reason="tensorrt 미설치 또는 CUDA 불가",
)


def test_trt_fp16_engine_build_and_infer(tmp_path):
    from hippo.export.to_onnx import export_onnx
    from hippo.export.to_tensorrt import build_tensorrt_engine, load_trt_engine
    from hippo.models import build_model

    net = build_model("unet", 1, 3, (32, 32, 32)).cuda().eval()
    onnx16 = str(tmp_path / "m_fp16.onnx")
    engine = str(tmp_path / "m_fp16.plan")

    export_onnx(net, onnx16, (32, 32, 32), 1, "cuda", half=True)
    res = build_tensorrt_engine(onnx16, engine, fp16=True)
    assert res["ok"] is True

    eng = load_trt_engine(engine, "cuda")
    x = torch.randn(1, 1, 32, 32, 32, device="cuda")
    out = eng(x)
    assert tuple(out.shape) == (1, 3, 32, 32, 32)

    # FP16 엔진이지만 argmax 분할은 PyTorch FP32와 거의 동일해야 함
    with torch.no_grad():
        ref = net(x)
    seg_match = (out.argmax(1) == ref.argmax(1)).float().mean().item()
    assert seg_match > 0.99
    assert np.isfinite(out.cpu().numpy()).all()
