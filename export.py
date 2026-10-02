"""python export.py [--tensorrt]: `hippo export` 과 동일.

학습 모델 -> ONNX(+parity)[ -> TensorRT FP16 엔진]. 코드는 src/hippo 패키지에 있음.
예)
    python export.py --onnx
    python export.py --tensorrt
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from hippo.cli import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "export", *sys.argv[1:]]
    app()
