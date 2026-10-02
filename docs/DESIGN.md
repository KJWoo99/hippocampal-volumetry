# 해마 3D 분할과 부피 정량화: 설계 문서

> 분할은 수단, 부피가 목적. "UNet 학습"이 아니라 해마 정량화 시스템으로 설계함.
> 따라서 Dice 뿐 아니라 부피 일치도(ICC / Bland-Altman) 를 별도 축으로 평가함.

- 데이터: Medical Segmentation Decathlon Task04 Hippocampus (CC-BY-SA 4.0).

---

## 1. 데이터의 도메인 한계

Task04 는 해마 주변만 잘라낸 crop 임. 따라서:

- 좌/우(L/R) 없음, ICV(두개내용적) 없음. anterior(1) / posterior(2) 라벨만 존재함.
- 그래서 volumetry 를 두 층으로 분리함:
  1. 데이터 실증층: Task04 에서 anterior/posterior/total 부피를 측정하고 정답 부피와의 일치도를 검증함.
  2. 임상 확장층: ICV 정규화, 좌우 비대칭, 연령 정상군 percentile. 전뇌 데이터(예: OASIS)가 필요하며,
     없으면 문헌 기반 정상값으로 인터페이스만 시연함.

이 구분을 흐리면 "임상에서 쓸 수 있다"는 과장이 됨.

---

## 2. 아키텍처

중심은 volumetry 리포트이고, 나머지 모듈은 부피 정확도와 신뢰도를 받치는 역할임.

```
                    [임상 volumetry 리포트]  <- 최종 산출물

        ┌─────────────────┼─────────────────┐
   연구급 평가          안전/QC            추론 가속
   "부피가 정확한가"   "부피를 신뢰?"    "더 빠르게"
   ICC, Bland-Altman   입력검증,           TensorRT FP16
   (Dice가 아니라      출력 QC             엔진 추론 +
    부피 bias 검증)    문제 시 사람 검토   백엔드 벤치마크
```

### 2.1 패키지 구조
```
train.py predict.py crossval.py export.py compare.py benchmark.py   # 최상위 실행파일(얇은 문)
src/hippo/
├── config.py            # OmegaConf 스키마 + 로더
├── data/                # transforms, datamodule(k-fold), 입력검증
├── models/              # model factory (unet/segresnet/swinunetr)
├── training/            # Trainer(AMP, grad accum, scheduler) + callbacks(EarlyStopping, Checkpoint)
├── metrics/             # segmentation(Dice/HD95/ASD) + volume_agreement(ICC/Bland-Altman)
├── volumetry/           # volume(부피계산) + normative(ICV/비대칭/percentile) + report(JSON/PDF)
├── degrade.py           # 화질 저하 모사(Rician 잡음, 바이어스 필드, 두꺼운 절편)와 고전 복원(NLM, N4, 삼차 보간)
├── inference/           # predictor(sliding window + 동적배치 + OOM복구) + qc
├── export/              # to_onnx + to_tensorrt(엔진/런타임) + benchmark
└── cli.py               # typer: train/crossval/predict/export/benchmark
tests/                   # 부피, 일치도, QC, 검증, transform, 콜백, ONNX parity
```

### 2.2 기술 선택 (의사결정 근거)

- Trainer: 순수 PyTorch + 콜백 (Lightning 대신). 이유: 의존성/버전 리스크 최소화, 수동
  EarlyStopping/ReduceLROnPlateau 로 완전한 제어. *Lightning 은 대안으로 검토했으나 단일 GPU 규모에 과함.*
- 설정: OmegaConf YAML + typer CLI 오버라이드 (Hydra 대신). 이유: Hydra 의 작업디렉토리 변경 없이
  재현성, 테스트가 쉬움.
- TensorRT FP16: 실제 적용. PyTorch->ONNX(opset18)->TensorRT 엔진->런타임 추론을
  `TRTEngine`(torch CUDA 텐서를 `execute_async_v3` 에 직접 바인딩, pycuda 불필요)으로 구현함.
  MONAI sliding window 가 엔진을 일반 모델처럼 호출해 패치 추론을 FP16 로 가속함.

  - TensorRT 11 대응: 11 부터 strongly-typed 패러다임으로 바뀌어 `BuilderFlag.FP16` /
    `EXPLICIT_BATCH` 가 사라짐. 그래서 FP16 는 빌더 플래그가 아니라
    FP16 ONNX 를 STRONGLY_TYPED 네트워크로 파싱해 얻음. 버전 적응형으로 분기 처리함.

  - 실측값은 한 번 내렸다가 다시 채움. 처음 올린 표는 학습된 가중치가 아니라
    초기화 상태의 모델을 잰 것이어서 내림(트러블슈팅 9번). 학습된 가중치로 다시
    재서 지금은 `outputs/backend_benchmark.json` 에 들어 있음(RTX 4090, 짝 단위 분할 뒤 배포 가중치).
    PyTorch FP32 2.842ms, ONNXRuntime 2.960ms, TensorRT FP16 0.925ms 로 3.07배임.
    채우는 경로는 `export.py --tensorrt` -> `benchmark.py`(`hippo benchmark`) ->
    `outputs/backend_benchmark.json` 임.

  - 세 백엔드를 나란히 측정함. PyTorch FP32 / ONNXRuntime / TensorRT FP16 이고,
    warmup 10, iter 50, batch 1 로 맞춤. ONNXRuntime 은 GPU EP 로 측정함:
    CPU EP 로 재면 "GPU 대 CPU" 를 비교한 표가 되므로 `onnxruntime-gpu` 가
    잡혔는지 먼저 확인함.

  - 배수에 기대를 두지 않는 이유. 3D conv + InstanceNorm 은 memory-bound 라
    텐서코어 이득이 제한되고, PyTorch+cuDNN 베이스라인 자체가 이미 강함.
    FP16 만 적용하고 INT8 은 쓰지 않음. 큰 배수가 나오면 그쪽을 의심함.

---

## 3. 핵심 과학

### 3.1 부피 계산에서 주의할 점

- voxel 부피 = spacing 곱(mm³). 어느 공간에서 세느냐가 중요함. 1mm 등방 resampling 공간에서 세면
  voxel 수 = mm³ 로 편하지만, 정확히는 native space 로 역변환 후 native spacing 으로 세거나
  resampling 을 보정해야 bias 가 없음.
- ICV 보정은 단순 나눗셈(ratio)보다 회귀 잔차(residual) 방식이 통계적으로 더 적절함 (Jack et al.).

### 3.2 부피 일치도를 따로 재는 이유

분할이 경계를 체계적으로 과/소추정하면 Dice 0.9 여도 부피에 bias 가 생김. 그래서 분할 메트릭과 별개로 부피
일치도(ICC + Bland-Altman bias / limits of agreement)를 평가함. 의료기기 측정 비교에서 흔히 쓰는 방법임.

### 3.3 계산식

- THV(Total Hippocampal Volume) = (ant_voxels + post_voxels) x voxel_vol
- 비대칭지수 AI = (L − R) / ((L + R)/2) x 100%  *(전뇌 데이터 층)*
- 정상군 z = (nHV − μ_age) / σ_age, 그리고 percentile

---

## 4. 평가 설계

```
260건  ->  개발셋 208 (5-fold CV: 아키텍처, 체크포인트 선택)
       +  홀드아웃 52 (최종 1회 평가)
```

홀드아웃은 모델 선택에 일절 사용하지 않음. 홀드아웃 시드(1234)는 학습 시드와 분리해 고정함.
분할은 MONAI `DecathlonDataset` 이 seed 0 으로 섞어 돌려준 순서 위에서 피험자 단위로 함. 같은 사람의 좌, 우 해마
(파일 번호 (31,32), (33,34) 같은 짝)는 홀드아웃과 개발셋, fold 사이에 갈리지 않음(`docs/prereg_pair_split.md`).
실제 목록은 `outputs/splits.json`.

부피 일치도의 ICC 는 체크포인트별(n=52)로 각각 계산함.
5개 fold 예측을 한 배열에 합치면 같은 개체가 5번 들어가 독립성 가정이 깨지고 n 이 5배로 부풀려짐.

결과와 한계는 [MODEL_CARD.md](MODEL_CARD.md) 를 참조함.
