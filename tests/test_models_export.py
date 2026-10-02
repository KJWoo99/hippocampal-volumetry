import torch

from hippo.export.to_onnx import export_onnx, verify_onnx_parity
from hippo.models import build_model, list_models


def test_list_models():
    assert "unet" in list_models()


def test_unet_forward_shape():
    net = build_model("unet", 1, 3, (32, 32, 32))
    x = torch.randn(1, 1, 32, 32, 32)
    out = net(x)
    assert out.shape == (1, 3, 32, 32, 32)


def test_onnx_export_parity_unet(tmp_path):
    net = build_model("unet", 1, 3, (32, 32, 32)).eval()
    onnx_path = str(tmp_path / "unet.onnx")
    export_onnx(net, onnx_path, (32, 32, 32), 1, "cpu")
    parity = verify_onnx_parity(net, onnx_path, (32, 32, 32), 1, "cpu")
    # onnxruntime 설치 환경에서는 ok=True (argmax 라벨 일치 + 작은 logits 오차)
    if parity.get("ok") is not None:
        assert parity["seg_match"] > 0.99
        assert parity["ok"] is True
