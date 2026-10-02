"""추론 백엔드 latency/throughput: PyTorch FP32 vs ONNXRuntime vs TensorRT FP16.

같은 입력으로 backend 별 지연, 처리량, VRAM 을 측정함.
TensorRT 가 없으면 PyTorch/ONNX 만 재고 TRT 는 'N/A' 로 남김.
"""
from __future__ import annotations

import os
import time

import torch


def _bench(fn, x, warmup: int = 10, iters: int = 50) -> dict:
    # 워밍업/반복을 충분히 줘야 GPU 클럭 부스트가 반영돼 측정이 안정적임.
    for _ in range(warmup):
        fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return {"latency_ms": dt * 1000.0, "throughput_ips": 1.0 / dt}


def benchmark_backends(
    model: torch.nn.Module,
    onnx_path: str | None = None,
    roi: tuple[int, int, int] = (32, 32, 32),
    in_channels: int = 1,
    device: torch.device | str = "cuda",
    iters: int = 50,
    engine_path: str | None = None,
) -> dict:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, in_channels, *roi, device=device)
    results: dict[str, dict] = {}

    # 1) PyTorch FP32
    model = model.to(device).eval()

    @torch.no_grad()
    def torch_fp32(inp):
        return model(inp)

    results["pytorch_fp32"] = _bench(torch_fp32, x, iters=iters)
    if device.type == "cuda":
        results["pytorch_fp32"]["vram_mb"] = torch.cuda.max_memory_allocated() / 1e6

    # 2) ONNXRuntime
    if onnx_path:
        try:
            import onnxruntime as ort
            providers = (["CUDAExecutionProvider"] if device.type == "cuda" else []) + [
                "CPUExecutionProvider"
            ]
            sess = ort.InferenceSession(onnx_path, providers=providers)
            xn = x.cpu().numpy()

            def onnx_run(_):
                return sess.run(["logits"], {"input": xn})

            results["onnxruntime"] = _bench(onnx_run, x, iters=iters)
        except Exception as e:  # noqa: BLE001
            results["onnxruntime"] = {"error": str(e)}

    # 3) TensorRT FP16: 엔진이 있으면 실제 런타임 추론을 측정
    from hippo.export.to_tensorrt import tensorrt_available
    if not tensorrt_available():
        results["tensorrt_fp16"] = {"note": "tensorrt 미설치: pip install tensorrt-cu12"}
    elif not engine_path or not os.path.exists(engine_path):
        results["tensorrt_fp16"] = {"note": "엔진(.plan) 없음: hippo export --tensorrt 먼저"}
    elif device.type != "cuda":
        results["tensorrt_fp16"] = {"note": "TensorRT는 CUDA 필요"}
    else:
        try:
            from hippo.export.to_tensorrt import load_trt_engine
            engine = load_trt_engine(engine_path, "cuda")

            def trt_run(inp):
                return engine(inp)

            torch.cuda.reset_peak_memory_stats()
            results["tensorrt_fp16"] = _bench(trt_run, x, iters=iters)
            results["tensorrt_fp16"]["vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
        except Exception as e:  # noqa: BLE001
            results["tensorrt_fp16"] = {"error": str(e)}

    # 상대 속도 요약
    base = results["pytorch_fp32"]["latency_ms"]
    for v in results.values():
        if "latency_ms" in v:
            v["speedup_vs_pytorch"] = round(base / v["latency_ms"], 2)
    return results


def summarize(results: dict) -> str:
    lines = ["backend            latency_ms  throughput  speedup"]
    for k, v in results.items():
        if "latency_ms" in v:
            lines.append(
                f"{k:18s} {v['latency_ms']:9.3f}  {v['throughput_ips']:9.1f}  "
                f"{v.get('speedup_vs_pytorch', float('nan')):.2f}x"
            )
        else:
            lines.append(f"{k:18s} {v.get('note') or v.get('error')}")
    return "\n".join(lines)
