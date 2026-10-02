"""홀드아웃, fold 목록 파일과 분할 코드의 약속(트러블슈팅 13).

문서의 "20%, seed 1234" 만 보고 분할을 다시 짜면 다른 52건이 나옴. DecathlonDataset 이 목록을
seed 0 으로 먼저 섞기 때문임. 그 seed 를 코드에 적었고, 실제 목록은 outputs/splits.json 에 남김.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "outputs" / "splits.json"


def test_get_datalist_passes_the_decathlon_shuffle_seed_explicitly():
    tree = ast.parse((ROOT / "src/hippo/data/datamodule.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_datalist")
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "DecathlonDataset"]
    assert calls, "get_datalist 가 DecathlonDataset 을 부르지 않는다"
    kws = {k.arg: k.value for k in calls[0].keywords}
    assert "seed" in kws and isinstance(kws["seed"], ast.Constant) and kws["seed"].value == 0


def test_splits_file_is_a_clean_partition():
    d = json.loads(SPLITS.read_text(encoding="utf-8"))
    hold = set(d["holdout"])
    folds = [set(f) for f in d["dev_folds_val"]]
    assert d["n_total"] == 260 and len(hold) == 52 and len(folds) == d["n_splits"] == 5
    dev = set().union(*folds)
    assert len(dev) == sum(len(f) for f in folds) == 208   # fold 끼리 겹치지 않음
    assert not (dev & hold)                                  # 홀드아웃이 CV 에 들어가지 않음


def test_splits_file_matches_the_code_when_data_is_present():
    if not (ROOT / "data" / "Task04_Hippocampus" / "dataset.json").is_file():
        pytest.skip("원자료 없음")
    pytest.importorskip("monai")
    import os
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import export_splits

    here = os.getcwd()
    try:
        os.chdir(ROOT)
        from hippo.config import load_config

        now = export_splits.build(load_config(str(ROOT / "configs" / "default.yaml")))
    finally:
        os.chdir(here)
    saved = json.loads(SPLITS.read_text(encoding="utf-8"))
    assert now["holdout"] == saved["holdout"]
    assert now["dev_folds_val"] == saved["dev_folds_val"]


# ── 짝 단위 분할(docs/prereg_pair_split.md, 트러블슈팅 16) ──────────────────────────
# 번호 (31,32) 처럼 홀수와 그다음 짝수가 한 사람의 좌우 해마임.
# 파일과 코드 양쪽에서 같은 사람이 갈리지 않는지 고정함.

def _key(name: str) -> int:
    import re

    return (int(re.search(r"(\d+)", name).group(1)) + 1) // 2


def test_splits_file_never_splits_a_subject():
    d = json.loads(SPLITS.read_text(encoding="utf-8"))
    hold_keys = {_key(n) for n in d["holdout"]}
    fold_keys = [{_key(n) for n in f} for f in d["dev_folds_val"]]
    assert not any(hold_keys & fk for fk in fold_keys), "같은 사람이 홀드아웃과 개발셋에 갈렸다"
    for i in range(len(fold_keys)):
        for j in range(i + 1, len(fold_keys)):
            assert not (fold_keys[i] & fold_keys[j]), f"같은 사람이 fold {i}, {j} 에 갈렸다"
    sizes = [len(f) for f in d["dev_folds_val"]]
    assert max(sizes) - min(sizes) <= 2


def _toy(ids):
    return [{"image": f"imagesTr/hippocampus_{i:03d}.nii.gz", "label": ""} for i in ids]


@pytest.fixture(scope="module")
def dm():
    pytest.importorskip("monai")
    from hippo.data import datamodule

    return datamodule


def test_holdout_split_keeps_subjects_whole_and_hits_the_target(dm):
    ids = [i for i in range(1, 140) if i not in (31, 32, 85, 86, 7, 50)]   # 결번과 혼자 남는 번호를 섞음
    data = _toy(ids)
    for seed in (0, 1, 1234):
        dev, hold = dm.holdout_split(data, 0.2, seed)
        assert len(hold) == round(len(data) * 0.2)
        hk = {dm.subject_key(x) for x in hold}
        assert not (hk & {dm.subject_key(x) for x in dev})
    assert dm.holdout_split(data, 0.2, 5) == dm.holdout_split(data, 0.2, 5)
    assert dm.holdout_split(data, 0.2, 5)[1] != dm.holdout_split(data, 0.2, 6)[1]


def test_kfold_keeps_subjects_whole_and_balanced(dm):
    ids = [i for i in range(1, 120) if i not in (31, 32, 85, 86, 9)]
    data = _toy(ids)
    vals = [dm.kfold_datalist(data, 5, k, 42)[1] for k in range(5)]
    names = [x["image"] for v in vals for x in v]
    assert sorted(names) == sorted(x["image"] for x in data)      # 빠짐, 겹침 없음
    keys = [{dm.subject_key(x) for x in v} for v in vals]
    for i in range(5):
        for j in range(i + 1, 5):
            assert not (keys[i] & keys[j])
    assert max(map(len, vals)) - min(map(len, vals)) <= 2
    tr, va = dm.kfold_datalist(data, 5, 0, 42)
    assert not ({dm.subject_key(x) for x in tr} & {dm.subject_key(x) for x in va})
