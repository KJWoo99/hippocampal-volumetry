# 실행 가이드 (USAGE)

이 프로젝트는 GUI 없는 CLI 임. 터미널에서 `hippo ...` 명령을 실행하면
터미널에 수치가 출력되고 `outputs/` 폴더에 그림, 리포트, 모델 파일이 생김.

---

## 0. 핵심 개념

- 학습(Training) = 모델을 가르침. PyTorch로 함. (`python train.py`)
- 추론(Inference) = 다 배운 모델로 예측. PyTorch 또는 TensorRT로 함.
- TensorRT는 학습용이 아니라 추론 "가속"용임. 학습이 끝난 모델을 그 GPU에 맞춰
  최적화해 더 빠르게 추론하게 해줌.

```
학습(PyTorch) -> 모델(.pth) -> 변환(.onnx) -> GPU 최적화 엔진(.plan) -> 빠른 추론
```

---

## 1. 환경 준비 (한 번만)

```powershell
conda activate hippo          # torch 2.14.0+cu132, monai 1.6.0
pip install -e ".[dev,export]"
# TensorRT 가속까지 쓰려면 (설치된 CUDA 와 매칭):
pip install tensorrt
```

---

## 2. 전체 실행 순서

> 실행 방식 3가지는 모두 동일하게 동작함 (같은 코드 + 같은 결과):
> - `python train.py` (친숙한 방식, 최상위 실행파일)
> - `python -m hippo.cli train` (모듈 방식)
> - `hippo train` (설치된 명령)
> 아래는 친숙한 `python OOO.py` 형태. (`hippo train`, `python -m hippo.cli train`도 동일)

```powershell
# 1) 학습 (기본 SegResNet, 400 epoch): outputs/train/segresnet/ 에 best_model.pth + dice_curve.png
python train.py

# 2) 아키텍처 벤치마크: benchmark_results.json (홀드아웃 분리, 최종 성능은 이걸로)
python scripts/run_benchmark.py

# 3) ONNX + TensorRT FP16 엔진 생성. 기본은 배포 가중치 outputs/best_model.pth 를 변환한다.
#    1번에서 새로 학습한 모델을 쓰려면 --ckpt outputs/train/segresnet/best_model.pth
python export.py --tensorrt

# 4) 단일 케이스 부피 리포트: report.json / report.pdf
python predict.py data/Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz

# 5) PyTorch vs TensorRT 속도, 정확도 비교
python compare.py data/Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz
```

### 최소 한 사이클 (시간 없을 때)
```powershell
python train.py  ->  python export.py --tensorrt --ckpt outputs/train/segresnet/best_model.pth  ->  python compare.py <img>
```

---

## 3. 명령어별 설명

| 명령 (python 형태) | 하는 일 | 입력 | 출력 |
|---|---|---|---|
| `python train.py` | SegResNet 학습 (기본 400 epoch, AMP, EarlyStopping, LR스케줄러) | 데이터 자동 다운로드 | `train/<모델>/best_model.pth`, `train/<모델>/dice_curve.png` (배포 가중치는 안 덮음) |
| `python scripts/run_benchmark.py` | 홀드아웃 분리 + 3종 아키텍처 비교 | 데이터 | `benchmark_results.json` (그림은 `scripts/render_figures.py`) |
| `python crossval.py` | 5-fold CV (낙관 편향: 모델 선택 전용) | 데이터 | `crossval/cv_results.json`, `crossval/bland_altman.png` |
| `python predict.py <img>` | 단일 NIfTI 추론 -> 부피 리포트 | `.nii.gz` | `report.json`, `report.pdf` |
| `python export.py --tensorrt` | ONNX + TensorRT FP16 엔진 빌드 | `best_model.pth` | `.onnx`, `.plan` |
| `python benchmark.py` | 백엔드 속도 비교 (순수 엔진, ROI 64³) | 모델/엔진 | 터미널 표 |
| `python compare.py <img>` | PyTorch vs TensorRT 속도+정확도 (전체 추론) | `.nii.gz` + 모델/엔진 | 터미널 표 |

> 동일 명령을 `hippo train`, `python -m hippo.cli train` 으로도 쓸 수 있음 (전부 같은 동작).

### 자주 쓰는 변형
```powershell
python train.py --epochs 50              # epoch 변경 (미지정 시 설정 파일 값 400)
python train.py --model segresnet        # 모델 변경 (unet|segresnet|swinunetr)
python crossval.py --folds 0,1,2         # 일부 fold만 (시간 절약)
python predict.py <img> --backend tensorrt   # 추론을 TensorRT 엔진으로
python train.py -o train.lr=0.0005 -o train.batch=4   # 임의 설정 덮어쓰기
```

### 보조 스크립트 (벤치마크 밖)

벤치마크 밖에서 쓰는 스크립트는 여덟 개임.

| 스크립트 | 하는 일 |
|---|---|
| `scripts/run_ablation.py` | 설정을 한 축씩 바꿔 개발셋 fold 로만 측정함(홀드아웃은 불러오지 않음). 기록은 `outputs/ablation/`, 추가 회차 규칙은 `docs/prereg_ablation_extension.md` |
| `scripts/render_experiment_log.py` | 절제 기록을 `docs/EXPERIMENT_LOG.md` 로 옮기고 판정선, 1위 묶음, 종료 판단(부모 대비 연속 미개선)을 스스로 계산함 |
| `scripts/render_figures.py` | `benchmark_results.json` 으로 `dice_curve.png`, `bland_altman.png` 를 그림 |
| `scripts/recompute_agreement.py` | 저장된 예측, 정답 부피로 일치도 지표와 피험자 단위 부트스트랩 구간만 다시 계산함(재학습 없음) |
| `scripts/sync_doc_numbers.py` | README, `docs/report.md`, MODEL_CARD 의 표 수치를 산출물에서 채움. `--check` 는 다른 칸만 알려 줌. 본문 문장은 보지 않음 |
| `scripts/export_splits.py` | 홀드아웃과 개발셋 fold 의 실제 케이스 이름을 `outputs/splits.json` 으로 남김 |
| `scripts/check_subject_pairs.py` | 분할의 근거인 피험자 짝 규칙(홀수와 그다음 짝수가 한 사람)을 원자료로 확인함. 결과는 `outputs/subject_pairs.json` |
| `scripts/run_degradation.py` | 부가 분석: 화질 저하와 복원이 부피 일치도에 주는 영향(`docs/prereg_degradation.md`). 결과는 `outputs/degradation_*.json` |

### `crossval.py` 는 성능 보고용이 아님

`python crossval.py` 는 val 로 고른 체크포인트를 같은 val 에서 평가하는 구버전 경로임.
모델 선택, 하이퍼파라미터 탐색에만 쓰고, 최종 성능은 반드시 `run_benchmark.py` 로 냄.

```powershell
python scripts/run_benchmark.py    # 홀드아웃 분리 + 3종 비교
```

---

## 4. 산출물 (`outputs/`)

| 파일 | 만든 명령 | 정체 | GPU 최적화 |
|---|---|---|---|
| `best_model.pth` | (저장소에 포함) | 배포 모델. 벤치마크가 고른 SegResNet 의 fold0 가중치를 복사함(어느 fold 를 쓸지는 `src/hippo/cli.py` 에 박혀 있음). 내보내기, 지연 측정, 예시 추론이 쓴 것과 같은 파일임. [report.md](report.md) "배포하는 것과 보고하는 것" 참조 | 범용 |
| `dice_curve.png` | render_figures | 아키텍처별 fold 성능 + 홀드아웃 기준선 | 없음 |
| `benchmark_results.json` | run_benchmark | 3종 CV + 홀드아웃 Dice/HD95/ASD + ICC/Bland-Altman | 없음 |
| `bland_altman.png` | render_figures | 부피 일치도 (LoA 신뢰구간 + 비례편향 회귀선) | 없음 |
| `report.json` / `report.pdf` | predict | 임상 부피 리포트 | 없음 |
| `splits.json` | export_splits | 홀드아웃 52건과 개발셋 5-fold 의 실제 케이스 이름(`scripts/export_splits.py`) | 없음 |
| `subject_pairs.json` | check_subject_pairs | 피험자 짝 규칙의 확인 결과(`scripts/check_subject_pairs.py`) | 없음 |
| `backend_benchmark.json` | benchmark | PyTorch / ONNX / TensorRT latency 실측 | 없음 |
| `degradation_*.json` | run_degradation | 부가 분석(화질 저하와 복원) 결과. 판정은 `degradation_results.json`, 바이어스 비 실측은 `degradation_bias_ratio.json`. [report.md](report.md) "부가 분석" 절 | 없음 |
| `hippocampus.onnx` | export | ONNX (범용 교환 포맷) |  범용 |
| `hippocampus_fp16.onnx` | export --tensorrt | FP16 ONNX (엔진 빌드 중간물) |  범용 |
| `hippocampus_fp16.plan` | export --tensorrt | TensorRT FP16 엔진 | 이 GPU 전용 |

### 파일 종류 핵심
- `.pth` -> `.onnx` -> `.plan` 순으로 "특정 환경 특화 + 빠름"이 됨.
- `.plan`은 빌드한 그 GPU, 드라이버, TensorRT 버전에 묶임 -> 다른 GPU로 옮기면 안 됨.
  그래서 배포 대상 머신에서 직접 빌드하고, git에는 커밋 안 함(.gitignore 처리).
- `.onnx`는 범용 -> 어디든 가져가 그 머신에서 `.plan`으로 빌드 가능.

> ONNX, TensorRT 엔진과 fold별 체크포인트 15개(합계 670MB)는 제외함.
> TensorRT `.plan` 은 GPU 아키텍처에 묶여 다른 환경에서 쓸 수 없음.
>
> 홀드아웃 분리 이전의 산출물은 낙관 편향이 있어 폐기함.
> 경위는 [MODEL_CARD 의 "해결된 한계"](MODEL_CARD.md).

---

## 5. 의존 관계 & 시간

| 단계 | 선행 필요 | 대략 시간 |
|---|---|---|
| `train` | 없음 | 에폭 수에 비례 (기본 400) |
| `run_benchmark` | 독립 (자체 학습) | 가장 김 (3종 x 5fold 를 직접 학습). 마지막 실행 5.68시간(RTX 4090, 증강 spatial-wide) |
| `export --tensorrt` | 없음(배포 가중치가 저장소에 있음). 새로 학습한 모델은 `--ckpt` | 1\~2분 |
| `predict` | 없음(배포 가중치 사용). 새로 학습한 모델은 train 뒤 `--ckpt` | 몇 초 |
| `compare` | export(엔진이 있어야 함) | 몇 초 |

> 새로 학습한 모델을 쓸 때만 train -> export 순서를 지킴. 저장소의 배포 가중치로는 export, predict, compare 가 학습 없이
> 돌아감. `run_benchmark` 는 언제 돌려도 됨(자체 학습).

---

## 6. 검증 (개발용)

```powershell
ruff check src scripts tests     # 린트
pytest -q                # TensorRT 테스트는 미설치 시 자동 skip
```

---

## 7. 결과 확인용 산출물
- `dice_curve.png` (아키텍처별 fold 성능)
- `bland_altman.png` + `benchmark_results.json` (부피 신뢰도 검증)
- `report.pdf` (임상 부피 리포트)
- `python compare.py <img>` 터미널 출력 캡처 (PyTorch vs TensorRT 속도, 정확도)

> `outputs/` 의 산출물은 전체 학습(3종 x 5fold) 결과임. 최종 수치는
> 여기에 옮겨 적지 않음. 지금 값은 `docs/report.md` 의 결과 절과
> `outputs/` 의 JSON 을 봄: 두 곳에 같은 숫자를 적어두면 한쪽만 갱신됨.
