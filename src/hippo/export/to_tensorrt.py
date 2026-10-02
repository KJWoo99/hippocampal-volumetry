"""ONNX -> TensorRT FP16 엔진 빌드.

'PyTorch->ONNX->TensorRT FP16' 배포 경로를 3D 의료 분할에 적용함.
tensorrt 미설치 환경(CI/일반 PC)에서는 우아하게 안내만 하고, CUDA가 맞는 환경에서 실제 빌드함.
"""
from __future__ import annotations

import os


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: int = 2,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 4,
) -> dict:
    """ONNX를 TensorRT 엔진(.plan)으로 컴파일. dict(ok, engine_path|reason).

    dynamic batch(min/opt/max) 프로파일을 설정하고 FP16를 켬.
    """
    if not tensorrt_available():
        return {
            "ok": None,
            "reason": (
                "tensorrt 미설치. CUDA 매칭 환경에서 "
                "`pip install tensorrt` 후 이 함수를 실행하세요."
            ),
        }
    if not os.path.exists(onnx_path):
        return {"ok": False, "reason": f"ONNX 없음: {onnx_path}"}

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    NDCF = trt.NetworkDefinitionCreationFlag
    has_fp16_flag = hasattr(trt.BuilderFlag, "FP16")
    flags = 0
    if hasattr(NDCF, "EXPLICIT_BATCH"):  # TRT8/9
        flags |= 1 << int(NDCF.EXPLICIT_BATCH)
    # TRT11은 FP16 빌더 플래그가 없고 strongly-typed 네트워크로 정밀도를 결정함.
    # 이 경우 fp16 ONNX를 strongly-typed로 파싱해야 FP16 엔진이 됨.
    used_strongly_typed = fp16 and not has_fp16_flag and hasattr(NDCF, "STRONGLY_TYPED")
    if used_strongly_typed:
        flags |= 1 << int(NDCF.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            return {"ok": False, "reason": "ONNX 파싱 실패", "errors": errs}

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and has_fp16_flag:  # TRT<=10: 빌더 플래그로 FP16 허용
        config.set_flag(trt.BuilderFlag.FP16)

    # dynamic batch 프로파일 (입력 0번 축이 batch)
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    shape = list(inp.shape)  # [batch, C, D, H, W] (batch=-1 dynamic)
    cdhw = shape[1:]
    profile.set_shape(inp.name, [min_batch, *cdhw], [opt_batch, *cdhw], [max_batch, *cdhw])
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return {"ok": False, "reason": "엔진 빌드 실패"}
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return {
        "ok": True,
        "engine_path": engine_path,
        "fp16": bool(fp16),
        "mode": "strongly_typed(onnx dtype)" if used_strongly_typed else "builder_flag",
    }


class TRTEngine:
    """TensorRT 엔진 런타임: torch CUDA 텐서를 직접 바인딩해 추론한다 (pycuda 불필요).

    엔진을 'logits를 반환하는 callable'로 노출하므로, MONAI ``sliding_window_inference``
    나 ``Predictor`` 에 일반 모델처럼 그대로 끼워 패치 추론을 FP16로 가속할 수 있음.
    """

    def __init__(self, engine_path: str, device: str = "cuda"):
        import tensorrt as trt
        import torch

        self._trt = trt
        self._torch = torch
        self.device = torch.device(device)
        # ERROR 레벨: 작은 모델에선 기본 스트림이 더 빠른데 TRT가 매번 경고를 내므로 억제.
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"엔진 역직렬화 실패: {engine_path}")
        self.context = self.engine.create_execution_context()

        # I/O 텐서 이름 식별
        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        # 엔진의 실제 I/O dtype (FP16 strongly-typed 엔진은 half I/O)
        self.in_dtype = self._torch_dtype(self.engine.get_tensor_dtype(self.input_name))
        self.out_dtype = self._torch_dtype(self.engine.get_tensor_dtype(self.output_name))

    def _torch_dtype(self, trt_dtype):
        trt, torch = self._trt, self._torch
        return {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }.get(trt_dtype, torch.float32)

    def __call__(self, x):
        """입력 (B,C,H,W,D) -> logits (B,classes,H,W,D). 출력은 항상 float32로 반환."""
        torch = self._torch
        x = x.to(self.device, dtype=self.in_dtype).contiguous()
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out = torch.empty(out_shape, device=self.device, dtype=self.out_dtype)
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())
        # 기본(현재) 스트림에서 실행: 작은 모델에선 전용 스트림보다 동기화 오버헤드가 적음.
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return out.float()

    # Predictor/sliding_window 호환용 no-op
    def eval(self):
        return self

    def to(self, *_args, **_kwargs):
        return self


def load_trt_engine(engine_path: str, device: str = "cuda") -> TRTEngine:
    return TRTEngine(engine_path, device)
