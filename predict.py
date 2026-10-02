"""python predict.py <img.nii.gz> [--backend tensorrt]: `hippo predict` 과 동일.

코드는 src/hippo 패키지에 있고, 이 파일은 그걸 실행하는 얇은 문임.
예)
    python predict.py data/Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz
    python predict.py <img> --backend tensorrt
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from hippo.cli import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "predict", *sys.argv[1:]]
    app()
