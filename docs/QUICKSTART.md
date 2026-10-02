# 빠른 실행 순서 (QUICKSTART)

학습 후 무슨 명령을 어떤 순서로 치면 되는지 한 장 요약. 자세한 설명은 [USAGE.md](USAGE.md).

> 세 방식 모두 동일: `python train.py` = `python -m hippo.cli train` = `hippo train`.
> 아래는 친숙한 `python OOO.py` 형태.

---

## 0. 준비 (한 번만)
```powershell
conda env create -f environment.yml
conda activate hippo
pip install -e ".[dev,export]"
# TensorRT 가속까지: pip install tensorrt
```

---

## 1. 전체 순서 (복사해서 그대로 실행)

```powershell
# 1) 학습 (기본 SegResNet, 400 epoch): outputs/train/segresnet/ 에 best_model.pth + dice_curve.png
#    (배포 가중치 outputs/best_model.pth 는 건드리지 않는다)
python train.py

# 2) 추론: 해마 부피 리포트
python predict.py data/Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz

# 3) ONNX + TensorRT FP16 엔진 생성 (배포 가중치 outputs/best_model.pth, 1번 결과는 --ckpt 로)
python export.py --tensorrt

# 4) PyTorch vs TensorRT 속도, 정확도 비교
python compare.py data/Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz

# 5) 백엔드 속도 벤치마크 표
python benchmark.py

# 6) 5-fold 교차검증 (제일 오래: 시간 될 때 마지막에)
python scripts/run_benchmark.py
```

---

## 2. 각 명령이 만드는 것 (`outputs/`)

| 순서 | 명령 | 결과물 |
|---|---|---|
| 1 | `python train.py` | `train/<모델>/best_model.pth`, `train/<모델>/dice_curve.png` |
| 2 | `python predict.py <img>` | `report.json`, `report.pdf` (해마 부피) |
| 3 | `python export.py --tensorrt` | `hippocampus.onnx`, `hippocampus_fp16.plan` |
| 4 | `python compare.py <img>` | 터미널: 속도, 정확도 표 |
| 5 | `python benchmark.py` | 터미널: PyTorch/ONNX/TRT 속도표 |
| 6 | `python scripts/run_benchmark.py` | `benchmark_results.json`. 그림 `dice_curve.png`, `bland_altman.png` 는 `python scripts/render_figures.py` 가 이 파일로 그림 |

---

## 3. 시간 없을 때: 핵심 3종만

```powershell
python predict.py data/Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz
python export.py --tensorrt
python compare.py data/Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz
```
부피 리포트와 TensorRT 가속 비교가 한 번에 나옴.

---

## 4. 참고

- 의존 순서: `export` -> (엔진 생김) -> `predict --backend tensorrt` / `compare`. `export` 는 기본으로 저장소의 배포 가중치
  `outputs/best_model.pth` 를 쓰므로 학습 없이 돌아감. 새로 학습한 모델을 내보내려면 `--ckpt outputs/train/<모델>/best_model.pth`.
  `run_benchmark` 는 독립(언제든).
- 시간: `run_benchmark` 는 3종x5fold 를 자체 학습하므로 가장 오래 걸림. 마지막 전체 실행(RTX 4090, 에폭 한도 400,
  조기종료, 증강 spatial-wide)은 5.68시간이었음(벤치마크 로그, 깃에는 올라가지 않음).
- 다른 샘플 분석: 경로의 `hippocampus_001`을 `hippocampus_003` 등으로 바꾸면 됨 (`data/Task04_Hippocampus/imagesTr/`에 여러 개).
- TensorRT 추론: `python predict.py <img> --backend tensorrt` (엔진 빌드 후).
