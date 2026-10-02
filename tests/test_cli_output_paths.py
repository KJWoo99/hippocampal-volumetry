"""`hippo train`, `hippo crossval` 이 배포 산출물을 덮지 않는지 소스에서 확인함.

Trainer 는 cfg.out_dir/best_model.pth 에 체크포인트를 씀. out_dir 가 기본값(outputs/)이면
저장소에 올라가는 배포 가중치(벤치마크 SegResNet fold0 사본)와 벤치마크 그림
(dice_curve.png, bland_altman.png)을 덮음. QUICKSTART 1단계가 `python train.py` 라
문서를 따라 하는 것만으로 배포 모델이 바뀜. 두 명령은 하위 폴더에 씀.
"""
from __future__ import annotations

import ast
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / "src" / "hippo" / "cli.py"


def _func_src(name: str) -> str:
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.unparse(fn)


def test_train_writes_under_train_subfolder():
    assert "cfg.out_dir = os.path.join(cfg.out_dir, 'train', cfg.model.name)" in _func_src("train")


def test_crossval_writes_under_crossval_subfolder():
    assert "cfg.out_dir = os.path.join(cfg.out_dir, 'crossval')" in _func_src("crossval")
