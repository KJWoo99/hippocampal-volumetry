"""데이터 모듈: DecathlonDataset 래핑 + k-fold 분할.

DecathlonDataset의 기본 train/val 분할 외에, 소규모 의료 데이터의 신뢰 가능한
성능 보고를 위해 K-fold 교차검증용 datalist 분할을 별도 제공함.
"""
from __future__ import annotations

import os
import re

import numpy as np
from monai.apps import DecathlonDataset
from monai.data import CacheDataset, DataLoader

from hippo.config import Config
from hippo.data.transforms import get_transforms


def _ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.data.root, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)


def get_datalist(cfg: Config, download: bool = True) -> list[dict]:
    """전체 training 섹션의 {image, label} 경로 목록을 얻는다 (k-fold 입력용)."""
    _ensure_dirs(cfg)
    # DecathlonDataset 은 val_frac=0 이어도 training 목록을 seed 로 한 번 섞어 돌려줌
    # (MONAI _split_datalist, np.random.RandomState(seed).shuffle). 홀드아웃과 fold 는 이 섞인 순서
    # 위에서 holdout_seed, 42 로 다시 나누므로, 이 seed 가 바뀌면 홀드아웃 구성도 바뀜. 기본값(0)에
    # 기대지 않고 적어 둠. 실제 목록은 outputs/splits.json 에 있음(트러블슈팅 13).
    ds = DecathlonDataset(
        root_dir=cfg.data.root, task=cfg.data.task, section="training",
        transform=None, download=download, cache_rate=0.0, num_workers=0,
        val_frac=0.0, seed=0,
    )
    # DecathlonDataset.data 는 {image,label} dict 리스트
    return list(ds.data)


def subject_key(item: dict) -> int:
    """볼륨이 속한 피험자의 키. 번호 (31,32), (33,34) 처럼 홀수와 그다음 짝수가 한 사람의 좌, 우 해마임.

    MSD 는 피험자와 파일의 대응을 공개하지 않아 자료로 확인한 규칙임. 이 규칙의 짝 90쌍은 해마 부피 상관이
    0.915 로 무작위 짝(95% 범위 -0.21~0.20)과 확연히 다름(scripts/check_subject_pairs.py, docs/prereg_pair_split.md).
    """
    num = int(re.search(r"(\d+)", os.path.basename(str(item["image"]))).group(1))
    return (num + 1) // 2


def _units(datalist: list[dict]) -> list[list[int]]:
    """같은 피험자의 볼륨 인덱스를 한 단위로 묶음. 순서는 목록에 처음 나온 순서임."""
    groups: dict[int, list[int]] = {}
    for i, item in enumerate(datalist):
        groups.setdefault(subject_key(item), []).append(i)
    return list(groups.values())


def holdout_split(
    datalist: list[dict], holdout_frac: float = 0.2, seed: int = 1234
) -> tuple[list[dict], list[dict]]:
    """전체를 (개발셋, 홀드아웃)으로 나눔. 같은 사람의 좌우 해마는 한쪽에만 들어감.

    홀드아웃은 체크포인트 선택, 하이퍼파라미터 결정에 일절 사용하지 않고, 최종 모델을 딱 한 번 평가할 때만 씀.
    val 로 고른 체크포인트를 같은 val 에서 평가하면 낙관 편향(winner's curse)이 생기기 때문임(트러블슈팅 1).

    seed 는 train.seed 와 분리해 고정함. 아키텍처, 하이퍼파라미터를 바꿔도 홀드아웃 구성이 같아야 비교가 성립함.

    피험자 단위(`subject_key`)를 시드로 섞어 앞에서부터 더하고, 더하면 목표 수를 넘는 단위는 건너뛰어 정확히
    목표 수를 채움(트러블슈팅 16).
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac 은 (0,1) 이어야 함, got {holdout_frac}")
    units = _units(datalist)
    rng = np.random.default_rng(seed)
    n_hold = int(round(len(datalist) * holdout_frac))
    hold_idx: set[int] = set()
    for u in rng.permutation(len(units)):
        members = units[u]
        if len(hold_idx) + len(members) <= n_hold:
            hold_idx.update(members)
        if len(hold_idx) == n_hold:
            break
    if len(hold_idx) != n_hold:
        raise ValueError(f"홀드아웃을 {n_hold}건으로 채우지 못했다({len(hold_idx)}건)")
    dev = [datalist[i] for i in range(len(datalist)) if i not in hold_idx]
    holdout = [datalist[i] for i in sorted(hold_idx)]
    return dev, holdout


def kfold_datalist(datalist: list[dict], n_splits: int, fold: int, seed: int = 42):
    """datalist를 n_splits로 나눠 fold번째를 val, 나머지를 train으로 반환. 같은 사람의 좌우는 한 fold 에 들어감.

    입력은 `holdout_split()` 이 돌려준 개발셋이어야 함(홀드아웃 제외 상태).
    피험자 단위를 시드로 섞고 차례로 볼륨이 가장 적은 fold 에 넣음(같으면 번호가 작은 fold). 단위가 1~2건이라
    fold 크기 차이는 2건 이하임.
    """
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold는 [0,{n_splits}) 범위여야 함, got {fold}")
    units = _units(datalist)
    rng = np.random.default_rng(seed)
    loads = [0] * n_splits
    members_of: list[list[int]] = [[] for _ in range(n_splits)]
    for u in rng.permutation(len(units)):
        k = min(range(n_splits), key=lambda j: (loads[j], j))
        members_of[k].extend(units[u])
        loads[k] += len(units[u])
    val_idx = set(members_of[fold])
    train = [datalist[i] for i in range(len(datalist)) if i not in val_idx]
    val = [datalist[i] for i in sorted(val_idx)]
    return train, val


class HippoDataModule:
    """학습/검증 DataLoader 생성기.

    - ``setup_decathlon`` : DecathlonDataset 기본 분할 사용 (빠른 단일 실험)
    - ``setup_fold``      : 외부 datalist + k-fold 분할 사용 (교차검증)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.train_t, self.val_t = get_transforms(
            tuple(cfg.model.roi),
            crop_samples=cfg.data.crop_samples,
            aug=getattr(cfg.data, "aug", "basic"),
        )
        self.train_ds = None
        self.val_ds = None
        self.holdout_ds = None

    def setup_decathlon(self, download: bool = True):
        c = self.cfg
        _ensure_dirs(c)
        self.train_ds = DecathlonDataset(
            root_dir=c.data.root, task=c.data.task, section="training",
            transform=self.train_t, download=download,
            cache_rate=c.data.cache_rate, num_workers=c.data.num_workers,
            val_frac=c.data.val_frac,
        )
        self.val_ds = DecathlonDataset(
            root_dir=c.data.root, task=c.data.task, section="validation",
            transform=self.val_t, download=False,
            cache_rate=c.data.cache_rate, num_workers=c.data.num_workers,
            val_frac=c.data.val_frac,
        )
        return self

    def setup_fold(self, fold: int, download: bool = True):
        """개발셋(홀드아웃 제외분)을 k-fold로 나눠 train/val 구성.

        홀드아웃은 여기서 완전히 배제되므로 체크포인트 선택에 영향을 줄 수 없음.
        """
        c = self.cfg
        full = get_datalist(c, download=download)
        dev, _holdout = holdout_split(full, c.data.holdout_frac, c.data.holdout_seed)
        train_list, val_list = kfold_datalist(dev, c.data.n_splits, fold, c.data.fold_seed)
        self.train_ds = CacheDataset(
            data=train_list, transform=self.train_t,
            cache_rate=c.data.cache_rate, num_workers=c.data.num_workers,
        )
        self.val_ds = CacheDataset(
            data=val_list, transform=self.val_t,
            cache_rate=c.data.cache_rate, num_workers=c.data.num_workers,
        )
        return self

    def setup_holdout(self, download: bool = False):
        """최종 1회 평가용 홀드아웃 데이터셋. 학습 중에는 절대 호출하지 않음."""
        c = self.cfg
        full = get_datalist(c, download=download)
        _dev, holdout = holdout_split(full, c.data.holdout_frac, c.data.holdout_seed)
        self.holdout_items = holdout   # 평가 순서 그대로. 부피 배열의 케이스 이름을 남기는 데 씀
        self.holdout_ds = CacheDataset(
            data=holdout, transform=self.val_t,
            cache_rate=c.data.cache_rate, num_workers=c.data.num_workers,
        )
        return self

    def train_loader(self):
        return DataLoader(
            self.train_ds, batch_size=self.cfg.train.batch, shuffle=True,
            num_workers=self.cfg.data.num_workers,
        )

    def val_loader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=self.cfg.data.num_workers)

    def holdout_loader(self):
        return DataLoader(
            self.holdout_ds, batch_size=1, shuffle=False, num_workers=self.cfg.data.num_workers
        )
