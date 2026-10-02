"""설정 스키마 + 로더 (OmegaConf 기반).

YAML 파일을 읽고 CLI에서 ``key=value`` 형태로 오버라이드할 수 있음.
Hydra의 작업디렉토리 변경 마법 없이 재현성, 테스트를 쉽게 하려고 OmegaConf만 씀.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import OmegaConf

# 64^3 인 이유:
#   1) Task04 학습 볼륨 260개의 최대 단일 축이 59 -> 64^3 는 전 볼륨을 잘림 없이 담음.
#   2) SwinUNETR 는 32배 축소 구조라 32^3 입력이 1x1x1 로 붕괴함. 실행 가능한 최소 크기가 64^3 임.
ROI = (64, 64, 64)

# 홀드아웃 분할 전용 시드. train.seed 와 분리해 고정함.
# 아키텍처, 하이퍼파라미터를 바꿔도 홀드아웃 구성이 변하면 비교가 무의미해지기 때문임.
HOLDOUT_SEED = 1234


@dataclass
class DataConfig:
    root: str = "./data"
    task: str = "Task04_Hippocampus"
    cache_rate: float = 1.0
    num_workers: int = 2
    val_frac: float = 0.2
    n_splits: int = 5  # k-fold 교차검증 fold 수

    # 260개 중 이 비율을 떼어 체크포인트 선택에 일절 쓰지 않음(val 로 고른 값을 그대로 보고하는 낙관 편향을 막음).
    holdout_frac: float = 0.2
    holdout_seed: int = HOLDOUT_SEED
    # k-fold 분할 시드. train.seed 와 떼어 고정함. 한 시드가 둘을 함께 정하면 "시드만 바꿔 다시 돌린" 회차가
    # 초기값의 운과 fold 나누기의 운을 섞어 측정함. train.seed 는 초기값, 증강 난수만 정함.
    fold_seed: int = 42

    # ROI 가 볼륨 전체를 덮으므로 RandCropByPosNegLabeld 는 사실상 no-op 임.
    # num_samples 를 1 로 두어 동일 샘플 4중복 생성을 막음(패치 학습 -> 전체 볼륨 학습).
    crop_samples: int = 1
    # 증강 단계. transforms.AUG_LEVELS 참고.
    # 절제실험에서 처음 회차 최고인 spatial-wide(#11, +0.0084)를 시드 반복으로 확인해 씀(세 시드 평균 0.8867,
    # 기준선 계열 0.8789). #11 위의 #17~#19 는 모두 판정선(0.0039) 안임.
    aug: str = "spatial-wide"


@dataclass
class ModelConfig:
    # 아키텍처 벤치마크(개발셋 5-fold CV, n=208, 400에폭 한도, 짝 단위 분할)로 확정.
    #   unet 0.8789 +/- 0.0016 / segresnet 0.8878 +/- 0.0025 / swinunetr 0.8865 +/- 0.0034
    # 프로토콜대로 CV Dice 최고값인 segresnet 을 선택함. swinunetr 와의 차이(0.0013)는 fold 간 표준편차보다
    # 작아 정확도로는 구분되지 않음. 이 선택은 "더 낫다"가 아니라 "규칙대로 골랐다"임.
    # YAML(configs/default.yaml)과 같아야 함. 어긋나면 테스트가 막음(tests/test_config_matches_yaml.py).
    name: str = "segresnet"  # unet | segresnet | swinunetr
    in_channels: int = 1
    out_channels: int = 3
    roi: tuple[int, int, int] = ROI


@dataclass
class TrainConfig:
    # 한도에 걸려 끊긴 점수로 판정하지 않게 넉넉히 두고, patience 가 멈출 지점을 정하게 둠(트러블슈팅 7).
    epochs: int = 400
    batch: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_interval: int = 2
    grad_accum: int = 1          # gradient accumulation steps
    amp: bool = True
    # EarlyStopping / ReduceLROnPlateau
    early_stop_patience: int = 20
    # 평가 때 최대 연결성분만 남길지. `hippo predict` 가 하는 후처리와 같게 켬. 절제실험 #1(켬)과 #13(끔)의
    # 차이는 판정선 안이었지만(+0.0002), 평가와 배포가 같은 마스크를 내놓아야 함.
    postproc_largest_component: bool = True
    lr_patience: int = 8
    lr_factor: float = 0.5
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    out_dir: str = "./outputs"


def load_config(yaml_path: str | None = None, overrides: list[str] | None = None) -> Config:
    """기본 설정 -> (선택) YAML 병합 -> (선택) CLI 오버라이드 순으로 구성."""
    base = OmegaConf.structured(Config)
    if yaml_path:
        base = OmegaConf.merge(base, OmegaConf.load(yaml_path))
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(base)  # type: ignore[return-value]
