"""PyTorch -> ONNX(opset18) 변환 + parity 검증.

분할의 본질은 argmax라서 logits의 미세 FP 오차보다 '라벨이 같은가'가 본질.
그래서 parity는 logits 최대오차 + argmax 라벨 일치율 두 축으로 봄.
"""
from __future__ import annotations

import numpy as np
import torch

# torch 2.14 의 dynamo exporter 는 opset 18 이상으로만 내보냄. 17 을 요구해도 변환에 실패하면
# (`No Adapter To Version 17 for Pad`) 경고만 남기고 18 로 두므로, 요구값을 18 로 맞춤.
# TensorRT, onnxruntime 모두 18 을 지원함.
OPSET = 18


def export_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    roi: tuple[int, int, int] = (32, 32, 32),
    in_channels: int = 1,
    device: torch.device | str = "cpu",
    half: bool = False,
) -> str:
    """half=True면 FP16 ONNX로 내보낸다 (TensorRT strongly-typed FP16 엔진용).

    half일 때 호출자의 모델을 오염시키지 않도록 deepcopy 후 .half() 함.
    """
    import copy

    model = model.to(device).eval()
    dummy = torch.randn(1, in_channels, *roi, device=device)
    if half:  # FP16 export는 CUDA에서만 안정적. 부작용 방지 위해 복제본을 변환.
        model = copy.deepcopy(model).half()
        dummy = dummy.half()
    # external_data=False 로 가중치를 .onnx 안에 담음.
    #
    # torch 2.14 의 dynamo exporter 는 external_data=True 가 기본값이라 가중치를
    # `<name>.onnx.data` 로 분리함. TensorRT 파서는 그 파일을 현재 작업 디렉터리
    # 기준으로 찾으므로, onnx 파일을 다른 경로에 두면 로드에 실패함.
    #   [TRT] [E] Failed to open file: m.onnx.data
    #   [TRT] [E] UNSUPPORTED_NODE: Failed to import initializer: ...conv.weight
    # 단일 파일로 내보내면 경로 의존성이 사라져 엔진 빌드가 재현 가능해짐.
    # (해마 모델은 20MB 아래라 2GB protobuf 한계와 무관함)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=OPSET,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        external_data=False,
    )
    return onnx_path


def verify_onnx_parity(
    model: torch.nn.Module,
    onnx_path: str,
    roi: tuple[int, int, int] = (32, 32, 32),
    in_channels: int = 1,
    device: torch.device | str = "cpu",
    atol: float = 3e-2,
    n_trials: int = 3,
    seed: int = 0,
) -> dict:
    """PyTorch vs ONNXRuntime 출력 비교.

    판정은 argmax 라벨 일치율로 함. 분할의 본질은 어느 클래스가 이겼는가
    이지 로짓의 소수점이 아님. logits 최대오차는 참고값으로 함께 냄.

    ## 입력을 고정하는 이유

    매 호출마다 새 난수 입력을 쓰면 CUDA 에서 max_abs_diff 가 0.0088~0.0113 으로 흔들려 판정이 실행마다 달라짐.
    시드를 고정하고 여러 입력으로 최악값을 봄.

    ## atol 을 3e-2 로 둔 이유

    CPU 에서는 오차가 4e-5 수준이지만 CUDA 에서는 9e-3 근처임. GPU 커널의 누적 순서와 TF32 때문이며 변환 오류가
    아님. argmax 일치율은 두 경우 모두 0.99998 이상이라, CUDA 분포의 한복판인 1e-2 는 임계로 쓸 수 없음.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"ok": None, "reason": "onnxruntime 미설치"}

    model = model.to(device).eval()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    gen = torch.Generator(device="cpu").manual_seed(seed)

    worst_diff, worst_match = 0.0, 1.0
    for _ in range(n_trials):
        x = torch.randn(1, in_channels, *roi, generator=gen).to(device)
        with torch.no_grad():
            torch_out = model(x).cpu().numpy()
        ort_out = sess.run(["logits"], {"input": x.cpu().numpy()})[0]
        worst_diff = max(worst_diff, float(np.abs(torch_out - ort_out).max()))
        worst_match = min(worst_match, float((torch_out.argmax(1) == ort_out.argmax(1)).mean()))

    return {
        "max_abs_diff": worst_diff,
        "seg_match": worst_match,
        "n_trials": n_trials,
        "device": str(device),
        "ok": bool(worst_match > 0.999 and worst_diff < atol),
    }
