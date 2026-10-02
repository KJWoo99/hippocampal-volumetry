"""python crossval.py [--folds 0,1,2,3,4]: `hippo crossval` 과 동일.

5-fold 교차검증 + 부피 일치도. 성능 보고에는 쓰지 말 것(scripts/run_benchmark.py 사용).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from hippo.cli import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "crossval", *sys.argv[1:]]
    app()
