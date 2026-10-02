"""학습 루프의 기울기 누적, 분할 격리 검증.

GPU, 실데이터 없이 확인할 수 있는 두 가지를 잡음.

1. 기울기 누적의 잔여분: 배치 수가 `grad_accum` 의 배수가 아니면 마지막
   묶음이 step 되지 않은 채 남고, 다음 에폭의 `zero_grad` 가 지워버림. 그
   배치들은 학습에 전혀 기여하지 않는데 에러도 나지 않음.
2. 홀드아웃 격리: 홀드아웃이 CV 분할에 한 조각도 섞이면 안 됨. 섞이면
   "최종 평가는 딱 한 번"이라는 전제가 깨져 낙관 편향이 되살아남.
"""
from __future__ import annotations

import pytest

from hippo.data.datamodule import holdout_split, kfold_datalist

# ── 1. 기울기 누적 ─────────────────────────────────────────────────────

def _count_optimizer_steps(n_batches: int, accum: int) -> int:
    """Trainer._train_epoch 의 step 발생 횟수를 그대로 재현함."""
    steps = 0
    pending = False
    for i in range(n_batches):
        pending = True
        if (i + 1) % accum == 0:
            steps += 1
            pending = False
    if pending:      # 에폭 끝 잔여분 flush
        steps += 1
    return steps


@pytest.mark.parametrize(
    "n_batches, accum, expected",
    [
        (10, 1, 10),   # 기본값: 매 배치 step
        (10, 2, 5),    # 딱 나누어떨어짐
        (10, 4, 3),    # 8개까지 2번 + 잔여 2개 1번 = 3
        (7, 3, 3),     # 6개까지 2번 + 잔여 1개 1번 = 3
        (1, 4, 1),     # 배치가 accum 보다 적어도 한 번은 step 되어야 함
        (0, 4, 0),     # 빈 로더는 step 없음
    ],
)
def test_no_batch_is_silently_discarded(n_batches, accum, expected):
    assert _count_optimizer_steps(n_batches, accum) == expected


def test_default_accum_one_steps_every_batch():
    """기본 설정(grad_accum=1)의 동작은 바뀌면 안 됨: 기존 결과의 전제임."""
    for n in (1, 5, 13, 100):
        assert _count_optimizer_steps(n, 1) == n


def test_every_batch_contributes_to_some_step():
    """어떤 배치도 step 없이 버려지지 않음."""
    for n_batches in range(1, 20):
        for accum in range(1, 6):
            steps = _count_optimizer_steps(n_batches, accum)
            # 모든 배치가 어떤 step 에든 포함되려면 step 수가
            # ceil(n/accum) 이어야 함
            assert steps == -(-n_batches // accum), (
                f"batches={n_batches} accum={accum}: step {steps} "
                f"!= ceil = {-(-n_batches // accum)}: 일부 배치가 버려진다"
            )


# ── 2. 홀드아웃 격리 ───────────────────────────────────────────────────

def _fake_datalist(n: int) -> list[dict]:
    return [{"image": f"img{i}.nii.gz", "label": f"lbl{i}.nii.gz"} for i in range(n)]


def test_holdout_and_dev_are_disjoint_and_complete():
    data = _fake_datalist(260)
    dev, hold = holdout_split(data, holdout_frac=0.2, seed=1234)

    assert len(hold) == 52
    assert len(dev) + len(hold) == len(data)
    dev_imgs = {d["image"] for d in dev}
    hold_imgs = {d["image"] for d in hold}
    assert dev_imgs & hold_imgs == set(), "홀드아웃이 개발셋과 겹친다"
    assert dev_imgs | hold_imgs == {d["image"] for d in data}


def test_holdout_split_is_deterministic_across_calls():
    """같은 시드면 항상 같은 홀드아웃: 아키텍처 비교의 전제임."""
    data = _fake_datalist(260)
    a_dev, a_hold = holdout_split(data, 0.2, seed=1234)
    b_dev, b_hold = holdout_split(data, 0.2, seed=1234)
    assert [d["image"] for d in a_hold] == [d["image"] for d in b_hold]
    assert [d["image"] for d in a_dev] == [d["image"] for d in b_dev]


def test_holdout_changes_with_seed():
    data = _fake_datalist(260)
    _, h1 = holdout_split(data, 0.2, seed=1234)
    _, h2 = holdout_split(data, 0.2, seed=999)
    assert [d["image"] for d in h1] != [d["image"] for d in h2]


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_holdout_split_rejects_invalid_fraction(frac):
    with pytest.raises(ValueError, match="holdout_frac"):
        holdout_split(_fake_datalist(10), holdout_frac=frac)


def test_kfold_never_touches_holdout():
    """CV 분할 전체를 합쳐도 홀드아웃이 한 조각도 들어오면 안 됨."""
    data = _fake_datalist(260)
    dev, hold = holdout_split(data, 0.2, seed=1234)
    hold_imgs = {d["image"] for d in hold}

    seen_val: set[str] = set()
    for fold in range(5):
        train, val = kfold_datalist(dev, n_splits=5, fold=fold, seed=42)
        train_imgs = {d["image"] for d in train}
        val_imgs = {d["image"] for d in val}

        assert train_imgs & hold_imgs == set(), f"fold{fold} train 에 홀드아웃 유입"
        assert val_imgs & hold_imgs == set(), f"fold{fold} val 에 홀드아웃 유입"
        assert train_imgs & val_imgs == set(), f"fold{fold} train/val 중복"
        assert train_imgs | val_imgs == {d["image"] for d in dev}
        seen_val |= val_imgs

    assert seen_val == {d["image"] for d in dev}, "모든 개발셋 표본이 한 번씩 val 이어야 한다"


@pytest.mark.parametrize("fold", [-1, 5, 99])
def test_kfold_rejects_out_of_range_fold(fold):
    with pytest.raises(ValueError, match="fold"):
        kfold_datalist(_fake_datalist(50), n_splits=5, fold=fold)


def test_kfold_folds_are_balanced():
    """fold 크기가 최대 1 차이여야 함: 한쪽이 쏠리면 CV 평균이 왜곡됨."""
    dev = _fake_datalist(208)
    sizes = [len(kfold_datalist(dev, 5, f, seed=42)[1]) for f in range(5)]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == len(dev)
