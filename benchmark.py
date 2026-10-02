"""python benchmark.py: `hippo benchmark` 과 동일.

PyTorch FP32 vs ONNXRuntime vs TensorRT FP16 지연/처리량 비교. 코드는 src/hippo 패키지에 있음.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from hippo.cli import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "benchmark", *sys.argv[1:]]
    app()
