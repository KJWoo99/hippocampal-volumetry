"""python train.py: `hippo train` 과 똑같이 학습을 실행하는 진입 파일.

코드는 src/hippo 패키지에 정리돼 있고, 이 파일은 그걸 '실행만' 하는 얇은 문임.
pip install -e . 를 안 했어도 동작하도록 src 경로를 직접 추가함.

예)
    python train.py                  # 설정 파일 값(기본 400 epoch, segresnet)
    python train.py --epochs 50
    python train.py --model segresnet
"""
import os
import sys

# src/ 를 import 경로에 추가 (설치 안 해도 hippo 패키지를 찾게)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from hippo.cli import app  # noqa: E402

if __name__ == "__main__":
    # 'train' 서브명령을 자동으로 붙여 `hippo train [옵션]` 과 동일하게 동작
    sys.argv = [sys.argv[0], "train", *sys.argv[1:]]
    app()
