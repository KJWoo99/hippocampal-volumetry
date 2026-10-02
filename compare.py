"""python compare.py <img.nii.gz>: `hippo compare` 과 동일.

같은 케이스를 PyTorch FP32와 TensorRT FP16로 둘 다 돌려 속도, 정확도를 나란히 비교.
예)
    python compare.py data/Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from hippo.cli import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "compare", *sys.argv[1:]]
    app()
