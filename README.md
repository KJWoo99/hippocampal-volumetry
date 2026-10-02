# 해마 3D 분할과 부피 정량화. 잰 부피를 믿고 쓸 수 있는가

> MONAI 기반 3D 해마 분할 -> 부피 정량화 -> 측정 신뢰도 검증 -> TensorRT 배포 파이프라인.
> 분할은 수단이고 부피가 목적이라, Dice 로 끝내지 않고 부피 일치도를 별도로 검증함.

도메인: 해마 위축은 알츠하이머의 핵심 바이오마커임. 부피를 자동으로 재는 도구는 여럿 있지만,
잰 값을 믿고 쓸 수 있는지를 따지는 경우는 드문 편임. 이 프로젝트는 거기에 답하는 것이 목표임.
데이터: Medical Segmentation Decathlon Task04 Hippocampus (자동 다운로드, CC-BY-SA 4.0)

빠른 실행 순서는 [docs/QUICKSTART.md](docs/QUICKSTART.md), 자세한 설명은 [docs/USAGE.md](docs/USAGE.md).
설계 상세는 [docs/DESIGN.md](docs/DESIGN.md), 모델 정보는 [docs/MODEL_CARD.md](docs/MODEL_CARD.md), 결과와 해석은 [docs/report.md](docs/report.md).

---

## 요약

1. 분할은 잘 됨. 홀드아웃 Dice 0.8884, 부피 일치도 ICC 0.8972.
2. 임상에 쓸 수 있는지는 용도에 따라 갈림. 개인의 위축 진행을 종단 추적하는 데는 쓸 수 없음. 측정 오차가 +/- 9.2%(95% LoA
   582.2mm³) 인데 연간 위축은 건강한 노화 1.41%, 알츠하이머 4.66% 임(메타분석). 오차가 재는 대상의 약 2\~6.5배 큼.
3. 정상군 대비 위축 선별에는 조건부로 쓸 수 있음. 개인차보다 군간 차이가 크기 때문. 부피는 평균 +133.4mm³(+4.2%) 크게 측정됨.
4. 비례편향은 유의하지 않음(기울기 −0.039, p=0.431, 피험자 부트스트랩 구간도 0 을 포함). 치우침이 크기에 따라 달라지기보다
   전 구간에 고르게 있어, 오차를 하나의 숫자로 요약할 수 있음.

ICC 0.897 은 교과서 기준 "좋은 일치도"(0.75\~0.9)의 윗부분임. 그래도 한 사람의 연간 변화를 재기에는
오차가 큼.

아키텍처 비교에서는 "SegResNet 이 가장 좋다"고 쓰지 않음. SwinUNETR
과의 차이가 0.0013 으로 fold 간 표준편차(0.0016\~0.0034)보다 작음. 하나를 골라야
해서 골랐을 뿐 두 모델은 이 데이터에서 구분되지 않음. UNet 과의 차이(0.0089)만
표준편차보다 뚜렷하게 큼.

분할은 같은 사람의 좌우 해마가 홀드아웃과 개발셋, 또는 서로 다른 fold 로 갈리지 않게 피험자 단위로 나눔.
공개 자료에 피험자 정보가 없어 파일 번호의 짝 규칙을 자료로 확인해 씀([docs/report.md](docs/report.md) 실험 설계, 트러블슈팅 16).

배포 경로도 만들어 둠. PyTorch에서 ONNX 로 내보내고 원본과 출력이 같은지
대조한 뒤 TensorRT FP16 엔진까지 만듦. 지연은 RTX 4090 에서 세 백엔드를 나란히
쟀고, TensorRT FP16 이 PyTorch 대비 3.07배임([docs/report.md](docs/report.md) 배포 절).

## 무엇을 만들었나

- 분할이 아니라 정량화가 목적. mask -> 해마 부피(mm³) -> 임상 리포트(JSON/PDF)
- 부피 신뢰도를 별도 검증. 높은 Dice ≠ 정확한 부피. 예측부피 vs 정답부피
  ICC + Bland-Altman (의료기기 측정 비교에서 쓰는 방법)
- 평가. 5-fold CV, Dice/HD95/ASD, 아키텍처 벤치(UNet/SegResNet/SwinUNETR),
  홀드아웃 52건은 모델 선택에 일절 미사용
- 안전망. 입력검증, 출력 QC
- 추론 가속(배포). PyTorch->ONNX(opset18)->TensorRT FP16 엔진 + VRAM 동적배치/OOM
  복구 + 백엔드 벤치마크 (CLI 기반, 웹서버 없음)

## 설치
```powershell
conda env create -f environment.yml   # 처음 한 번
conda activate hippo      # torch 2.14.0+cu132, monai 1.6.0
pip install -e ".[dev,export]"
# TensorRT 가속(선택, CUDA 매칭): pip install tensorrt onnxruntime-gpu
ruff check src scripts tests && pytest -q   # 테스트
```

## 사용

세 방식 모두 동일하게 동작: `python train.py` = `python -m hippo.cli train` = `hippo train`.
아래는 최상위 실행파일을 직접 부르는 형태.

```powershell
python train.py                                    # 학습 (기본 SegResNet, 400 epoch, Dice 곡선 저장)
python scripts/run_benchmark.py                    # 홀드아웃 분리 + 3종 아키텍처 비교
python crossval.py --folds 0,1,2,3,4               # 5-fold CV (모델 선택 전용. 성능 보고는 run_benchmark.py)
python predict.py data/.../hippocampus_001.nii.gz  # 부피 리포트(JSON+PDF)
python export.py --tensorrt                        # ONNX(+parity) + TensorRT FP16 엔진
python benchmark.py                                # PyTorch vs ONNX vs TensorRT latency
python predict.py <img> --backend tensorrt         # TensorRT FP16 엔진으로 추론
python compare.py <img>                            # PyTorch vs TensorRT 속도, 정확도 비교
```

보조 스크립트 여덟 개, 산출물 파일, 명령별 입출력은 [docs/USAGE.md](docs/USAGE.md) 에 있음.

---

## 결과 요약

| 항목 | 결과 |
|---|---|
| 아키텍처 비교(개발셋 5-fold CV) | SegResNet 0.8878, SwinUNETR 0.8865, UNet 0.8789. 앞의 둘은 구분되지 않음 |
| 홀드아웃(52건, 피험자 36명) | Dice 0.8884, HD95 1.40, ASD 0.433 |
| 부피 일치도 | ICC 0.8972(피험자 부트스트랩 [0.809, 0.943]), bias +133.4mm³, 95% LoA 반폭 9.2% |
| 비례편향 | 유의하지 않음(기울기 −0.039, p=0.431) |
| 배포 | TensorRT FP16 0.93ms, PyTorch 대비 3.07배. ONNX FP32 parity 만 확인했고 FP16 엔진의 분할 정확도는 홀드아웃에서 재지 않음 |
| 부가 분석(화질 저하) | 잡음 f 0.06 에서 부피 일치도 하락 없음(주 판정). 두꺼운 절편 3, 4 mm 에서 ICC 가 크게 떨어짐 |

결과와 해석은 [docs/report.md](docs/report.md), 결함과 고친 경위는 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md),
모델의 쓰임과 한계는 [docs/MODEL_CARD.md](docs/MODEL_CARD.md), CLAIM 2024 항목별 대응은 [docs/CLAIM.md](docs/CLAIM.md) 에 있음.

## 한계

- 측정 정밀도가 종단 추적에는 부족함(95% LoA +/- 9.2%). 부피를 평균 +133.4mm³(+4.2%) 크게 측정함.
- 피험자 단위 분할은 파일 번호의 짝 규칙(홀수와 그다음 짝수가 한 사람)에 기댐. MSD 는 피험자와 파일의 대응을 공개하지 않음.
- 홀드아웃 52건은 피험자 36명이라 케이스를 독립으로 보는 구간은 좁을 수 있음. 피험자 단위 부트스트랩 구간을 함께 실음.
- 단일 기관(Vanderbilt), 한 기종(Philips Achieva) 데이터임.
- crop 데이터라 ICV 정규화, 비대칭, 연령 percentile 은 인터페이스만 있음.

상세와 해결된 한계(winner's curse 제거, 피험자 단위 분할)는 [docs/MODEL_CARD.md](docs/MODEL_CARD.md) 한계 절.

연구용. 의료기기 아님. 임상 판정에 사용 불가.

---

## 데이터 출처 및 라이선스

Medical Segmentation Decathlon. Task04_Hippocampus (CC BY-SA 4.0)

> Antonelli M, Reinke A, Bakas S, et al. *The Medical Segmentation Decathlon.*
> Nature Communications 13, 4128 (2022). https://doi.org/10.1038/s41467-022-30695-9
>
> 데이터 제공: Vanderbilt University Medical Center, http://medicaldecathlon.com/

자료 설명은 Simpson AL 등, *A large annotated medical image dataset for the development and evaluation of
segmentation algorithms*, arXiv:1902.09063 (2019) 의 Task04 절을 따름. 성인 195명(건강인 90명, 비정동성
정신병 105명: 조현병 56, 조현정동장애 32, 조현양상장애 17)을 Vanderbilt 의 Psychiatric Genotype/Phenotype
Project 에서 모음. 영상은 3D T1 강조 MPRAGE(TI/TR/TE 860/8.0/3.7 ms, 1.0 mm 등방)이고 Philips Achieva 한
기종에서 찍음. 라벨은 발표된 프로토콜에 따른 수동 추적이며 해마 고유부(CA1\~4, 치상회)와 해마이행부
(subiculum) 일부를 머리(anterior)와 몸통, 꼬리(posterior)로 나눔. 공개 볼륨은 좌우 해마를 잘라 낸 crop
이라 피험자 한 명이 볼륨 둘이 됨. 볼륨별 진단 정보는 배포되지 않음.

의료영상 AI 보고 지침 CLAIM 2024 의 항목별 대응은 [docs/CLAIM.md](docs/CLAIM.md) 에 있음.

데이터셋은 저장소에 포함되어 있지 않으며, 실행 시 MONAI `DecathlonDataset` 이 자동으로 내려받음.

| 대상 | 라이선스 |
|---|---|
| 소스 코드 (`src/`, `scripts/`, `tests/`, `configs/`, `docs/`) | MIT |
| `outputs/` 파생 산출물 (플롯, 지표, 리포트, `best_model.pth`) | CC BY-SA 4.0. 원본 데이터의 ShareAlike 조건 승계 |
| ONNX, TensorRT 엔진 | 미포함 (`.gitignore`) |

집계 통계는 일반적으로 2차적저작물로 보지 않으나, 의도적으로 더 엄격한 해석을 적용함.

상세는 [LICENSE](LICENSE).

## 트러블슈팅

실측으로 잡은 결함 17건의 경위와 수정은 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) 에 있음. 번호는 문서들이 가리키는 번호 그대로임.

1. val 로 고른 체크포인트를 같은 val 에서 평가하고 있었음 (winner's curse)
2. ICC 독립성 위반. 같은 개체를 5회 계상
3. 비례편향을 검정하지 않고 없다고 가정하고 있었음
4. 구분되지 않는 차이로 아키텍처 순위를 매기고 있었음
5. 피험자 매핑이 공개되지 않았음을 출처 추적으로 확인\
   5-b. 안전장치 3개가 동작하지 않고 있었음
6. 좋은 지표가 임상적으로는 부족했음
7. 아키텍처 비교의 학습 절반이 수렴 전에 끊겨 있었음(SegResNet 은 다섯 fold 전부)
8. 설정을 한 번도 바꿔보지 않았음
9. 학습한 적 없는 모델을 ONNX 로 내보내고 있었음
10. 250에폭으로 다시 재니 아키텍처 선택과 비례편향 판정이 둘 다 바뀌었음
11. 에폭 한도를 400 으로 풀고 전부 다시 돌렸음. 결론은 그대로였음
12. 홀드아웃 부피를 후처리 안 한 마스크로 재고 있었음
13. 문서의 분할 규칙만으로는 같은 홀드아웃이 나오지 않았음
14. 화질 저하 분석의 설계가 처음 값대로면 뜻이 없었음
15. 절제실험이 멈추는 판단 없이 멈춰 있었음
16. 같은 사람의 좌우 해마가 홀드아웃과 개발셋에 갈려 있었음

