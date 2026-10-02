"""hippo CLI: 전체 파이프라인 진입점.

예)
  hippo train --epochs 100 --model unet
  hippo crossval --folds 0,1,2,3,4
  hippo predict path/to/image.nii.gz
  hippo export --onnx --tensorrt
  hippo benchmark
  hippo compare img.nii.gz            # PyTorch vs TensorRT 속도, 정확도 한 번에
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import typer

from hippo.config import load_config

app = typer.Typer(add_completion=False, help="Hippocampal segmentation & volumetry")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 저장소의 설정 파일. `--config` 를 안 주면 이것을 써서 스크립트와 CLI 가 같은 설정을 봄(트러블슈팅 9).
_DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _cfg(config: str | None, overrides: list[str] | None):
    if not config and _DEFAULT_CFG.exists():
        config = str(_DEFAULT_CFG)
    return load_config(config, overrides)


@app.command()
def train(
    config: str = typer.Option(None, help="YAML 설정 경로"),
    epochs: int = typer.Option(None, help="미지정 시 설정 파일 값(기본 400)"),
    model: str = typer.Option(None, help="미지정 시 설정 파일 값(기본 segresnet, unet|segresnet|swinunetr)"),
    override: list[str] = typer.Option(None, "--override", "-o", help="key=value"),
):
    """DecathlonDataset 기본 분할로 단일 학습."""
    from hippo.data.datamodule import HippoDataModule
    from hippo.evaluation.plots import plot_dice_curve
    from hippo.models import build_model
    from hippo.training.trainer import Trainer

    ov = list(override or [])
    if epochs:
        ov.append(f"train.epochs={epochs}")
    if model:
        ov.append(f"model.name={model}")
    cfg = _cfg(config, ov)
    # 학습 산출물은 하위 폴더에 둠. out_dir(outputs/) 바로 아래에 두면 Trainer 가 쓰는
    # best_model.pth 가 배포 가중치(벤치마크 SegResNet fold0 사본, 저장소에 올라가는 파일)를,
    # dice_curve.png 가 벤치마크 그림을 덮음.
    cfg.out_dir = os.path.join(cfg.out_dir, "train", cfg.model.name)
    dev = _device()
    typer.echo(f"[device] {dev}  model={cfg.model.name}  epochs={cfg.train.epochs}  "
               f"out={cfg.out_dir}")

    dm = HippoDataModule(cfg).setup_decathlon(download=True)
    net = build_model(cfg.model.name, cfg.model.in_channels, cfg.model.out_channels, tuple(cfg.model.roi))
    res = Trainer(cfg, net, dev).fit(dm.train_loader(), dm.val_loader())
    typer.echo(f"[done] best dice={res.best_metric:.4f} @epoch {res.best_epoch} -> {res.ckpt_path}")
    if any("dice" in h for h in res.history):
        png = plot_dice_curve(res.history, os.path.join(cfg.out_dir, "dice_curve.png"))
        typer.echo(f"[saved] {png}")


@app.command()
def crossval(
    config: str = typer.Option(None),
    folds: str = typer.Option(None, help="쉼표구분, 예 0,1,2"),
    override: list[str] = typer.Option(None, "--override", "-o"),
):
    """K-fold 교차검증 + 부피 일치도(ICC/Bland-Altman). 성능 보고용 아님."""
    from hippo.evaluation.crossval import run_cross_validation
    from hippo.evaluation.plots import plot_bland_altman

    # 모듈 docstring 에만 적어두면 CLI 로 쓰는 사람은 못 봄. 이 경로는 val fold
    # 에서 고른 체크포인트를 같은 fold 에서 재평가하므로 낙관 편향이 있음.
    # 최종 성능은 홀드아웃 벤치마크(`hippo benchmark`)로만 보고해야 함.
    typer.secho(
        "[주의] crossval 은 모델 선택용이다. val fold 에서 고른 체크포인트를 "
        "같은 fold 에서 재평가하므로 여기 나오는 Dice 는 낙관 편향"
        "(winner's curse)이 있다. 최종 성능은 `hippo benchmark` 로 보고할 것.",
        fg=typer.colors.YELLOW,
    )
    cfg = _cfg(config, list(override or []))
    # train 과 같은 이유로 하위 폴더에 씀. 여기 그림은 낙관 편향이 있는 CV 값이라
    # 벤치마크의 bland_altman.png 를 덮으면 문서가 인용하는 그림이 바뀜.
    cfg.out_dir = os.path.join(cfg.out_dir, "crossval")
    os.makedirs(cfg.out_dir, exist_ok=True)
    fold_list = [int(x) for x in folds.split(",")] if folds else None
    out = run_cross_validation(cfg, _device(), fold_list)

    with open(os.path.join(cfg.out_dir, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in out.items() if k != "volumes"}, f, ensure_ascii=False, indent=2)
    plot_bland_altman(
        out["volumes"]["predicted"], out["volumes"]["reference"],
        os.path.join(cfg.out_dir, "bland_altman.png"),
    )
    s = out["summary"]
    va = out["volume_agreement"]
    typer.echo(f"[CV] dice={s['dice']['mean']:.4f} +/- {s['dice']['std']:.4f}  "
               f"hd95={s['hd95']['mean']:.2f}  ICC={va['icc_2_1']:.3f}  "
               f"bias={va['bland_altman']['bias']:.1f}mm^3")
    typer.secho("      ^ 낙관 편향된 수치다. 최종 보고에 쓰지 말 것.", fg=typer.colors.YELLOW)


@app.command()
def predict(
    image: str,
    config: str = typer.Option(None),
    pdf: bool = typer.Option(True, help="PDF 리포트도 생성"),
    backend: str = typer.Option("pytorch", help="pytorch | tensorrt (FP16 엔진)"),
):
    """단일 NIfTI 추론 -> 부피 리포트(JSON[+PDF]). backend로 추론 엔진 선택."""
    import numpy as np

    from hippo.data.transforms import get_transforms
    from hippo.data.validation import validate_nifti
    from hippo.inference.predictor import Predictor
    from hippo.inference.qc import keep_largest_component, qc_mask
    from hippo.models import build_model
    from hippo.volumetry.report import build_report, save_json, save_pdf
    from hippo.volumetry.volume import compute_volumes, spacing_from_affine

    cfg = _cfg(config, None)
    dev = _device()
    info = validate_nifti(image)
    typer.echo(f"[input] shape={info.shape} spacing={info.spacing} {info.orientation} warnings={info.warnings}")

    net = build_model(cfg.model.name, cfg.model.in_channels, cfg.model.out_channels, tuple(cfg.model.roi))
    # export, benchmark 와 같은 규칙으로 가중치를 찾고, 없으면 멈춤(미학습 모델로 리포트를 만들지 않음).
    _load_deployment_weights(cfg, net, dev, None)

    # 추론 백엔드: pytorch(기본) 또는 TensorRT FP16 엔진 (sliding window가 엔진을 호출)
    infer_model = net
    if backend == "tensorrt":
        from hippo.export.to_tensorrt import load_trt_engine
        eng_path = os.path.join(cfg.out_dir, "hippocampus_fp16.plan")
        if not os.path.exists(eng_path):
            raise typer.BadParameter(f"엔진 없음: {eng_path} (hippo export --tensorrt 먼저)")
        infer_model = load_trt_engine(eng_path, "cuda")
        typer.echo(f"[backend] TensorRT FP16 engine: {eng_path}")

    _, val_t = get_transforms(tuple(cfg.model.roi))
    data = val_t({"image": image, "label": image})
    x = data["image"].unsqueeze(0).to(dev)
    mask = Predictor(infer_model, tuple(cfg.model.roi), device=dev, amp=cfg.train.amp).predict_mask(x)
    raw_mask = mask.cpu().numpy()[0].astype(np.int16)
    mask = keep_largest_component(raw_mask)
    spacing = spacing_from_affine(data["image"].affine)  # 1mm resampling 후 격자에서 유도
    vol = compute_volumes(mask, spacing)
    # QC 는 정제 전 마스크로 봄. 정제 후에는 라벨당 연결성분이 하나뿐이라
    # 파편화 검사가 절대 발동하지 않아, 원본이 아무리 노이즈투성이여도 리포트가
    # "파편화 없음"이라고 말함. 부피는 정제 후 값을 쓰되, 그 부피가 얼마나
    # 정리된 결과인지 함께 남김.
    qc = qc_mask(raw_mask, vol.total_mm3)
    removed = int((raw_mask > 0).sum() - (mask > 0).sum())
    qc["cleanup_removed_voxels"] = removed
    if removed:
        qc["flags"].append(f"정제로 제거된 노이즈 복셀 {removed}개 (최대 연결성분만 유지)")
        qc["passed"] = False
    report = build_report(os.path.basename(image), vol, qc=qc, model_name=cfg.model.name)

    base = os.path.join(cfg.out_dir, "report")
    save_json(report, base + ".json")
    typer.echo(f"[saved] {base}.json  total={vol.total_mm3:.1f}mm^3 qc_passed={qc['passed']}")
    if pdf:
        save_pdf(report, base + ".pdf")
        typer.echo(f"[saved] {base}.pdf")


def _dice(a, b, labels=(1, 2)):
    """두 정수 마스크 간 라벨별 Dice (numpy)."""

    from hippo import LABELS
    out = {}
    for lab in labels:
        A, B = (a == lab), (b == lab)
        s = A.sum() + B.sum()
        out[LABELS[lab]] = float(2 * (A & B).sum() / s) if s > 0 else 1.0
    return out


@app.command()
def compare(
    image: str,
    label: str = typer.Option(None, help="정답 마스크(없으면 Task04 경로에서 자동 탐색)"),
    config: str = typer.Option(None),
    iters: int = typer.Option(20, help="속도 측정 반복 수"),
):
    """같은 케이스를 PyTorch FP32와 TensorRT FP16로 둘 다 돌려 속도, 정확도를 나란히 비교."""
    import time

    import numpy as np

    from hippo.data.transforms import get_transforms
    from hippo.data.validation import validate_nifti
    from hippo.export.to_tensorrt import load_trt_engine, tensorrt_available
    from hippo.inference.predictor import Predictor
    from hippo.models import build_model
    from hippo.volumetry.volume import compute_volumes, spacing_from_affine

    cfg = _cfg(config, None)
    dev = _device()
    roi, ic = tuple(cfg.model.roi), cfg.model.in_channels
    if dev.type != "cuda":
        raise typer.BadParameter("compare는 CUDA 필요")

    # 정답 라벨 자동 탐색 (Task04: imagesTr -> labelsTr)
    if label is None and "imagesTr" in image:
        cand = image.replace("imagesTr", "labelsTr")
        label = cand if os.path.exists(cand) else None

    info = validate_nifti(image)
    typer.echo(f"[input] {os.path.basename(image)} shape={info.shape} {info.orientation}")

    net = build_model(cfg.model.name, ic, cfg.model.out_channels, roi).to(dev).eval()
    # predict 와 같음. 가중치가 없으면 비교 자체가 무의미하므로 멈춤.
    _load_deployment_weights(cfg, net, dev, None)

    eng_path = os.path.join(cfg.out_dir, "hippocampus_fp16.plan")
    if not (tensorrt_available() and os.path.exists(eng_path)):
        raise typer.BadParameter(
            f"TensorRT 엔진 필요: {eng_path} (tensorrt 설치 + hippo export --tensorrt)"
        )

    _, val_t = get_transforms(roi)
    data = val_t({"image": image, "label": label or image})
    x = data["image"].unsqueeze(0).to(dev)
    spacing = spacing_from_affine(data["image"].affine)
    gt = data["label"].cpu().numpy()[0].astype(np.int16) if label else None

    backends = {
        "pytorch_fp32": Predictor(net, roi, device=dev, amp=False),
        "tensorrt_fp16": Predictor(load_trt_engine(eng_path, "cuda"), roi, device=dev, amp=False),
    }

    rows = {}
    masks = {}
    for name, predictor in backends.items():
        predictor.predict_mask(x)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mask = predictor.predict_mask(x)
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) / iters * 1000
        m = mask.cpu().numpy()[0].astype(np.int16)
        masks[name] = m
        vol = compute_volumes(m, spacing)
        row = {"latency_ms": lat, "total_mm3": vol.total_mm3}
        if gt is not None:
            row["dice_vs_gt"] = _dice(m, gt)
        rows[name] = row

    # 출력
    pt, trt = rows["pytorch_fp32"], rows["tensorrt_fp16"]
    typer.echo("\n=== 속도 ===")
    typer.echo(f"  PyTorch FP32 : {pt['latency_ms']:.3f} ms")
    typer.echo(f"  TensorRT FP16: {trt['latency_ms']:.3f} ms  ({pt['latency_ms']/trt['latency_ms']:.2f}x)")
    typer.echo("\n=== 부피(mm^3) ===")
    typer.echo(f"  PyTorch : {pt['total_mm3']:.1f}")
    typer.echo(f"  TensorRT: {trt['total_mm3']:.1f}  (차이 {abs(pt['total_mm3']-trt['total_mm3']):.1f})")
    agree = float((masks["pytorch_fp32"] == masks["tensorrt_fp16"]).mean()) * 100
    consist = _dice(masks["pytorch_fp32"], masks["tensorrt_fp16"])
    typer.echo("\n=== 정확도 (두 백엔드 일치) ===")
    typer.echo(f"  voxel 라벨 일치율: {agree:.2f}%   Dice(pt vs trt): {consist}")
    if gt is not None:
        typer.echo("\n=== 정답 대비 Dice ===")
        typer.echo(f"  PyTorch : {pt['dice_vs_gt']}")
        typer.echo(f"  TensorRT: {trt['dice_vs_gt']}")


def _load_deployment_weights(cfg, net, dev, ckpt_opt: str | None):
    """배포용 가중치를 찾아 실음. 못 찾으면 진행하지 않음(무작위 초기화 모델을 내보내지 않게, 트러블슈팅 9).

    찾는 순서:

      1. `--ckpt` 로 직접 준 경로
      2. `outputs/best_model.pth` (배포 가중치. 벤치마크 SegResNet fold0 사본이고 저장소에 들어 있음.
         `hippo train` 은 `outputs/train/<모델>/` 에 쓰므로 이 파일을 덮지 않음)
      3. 벤치마크가 고른 아키텍처의 fold0 체크포인트

    3번을 쓸 때는 설정의 아키텍처와 벤치마크가 고른 아키텍처가 같은지 봄.
    다르면 배포되는 모델과 보고한 성능이 다른 모델이 됨.
    """
    import json as _json

    # 테스트는 이 명령들을 typer 를 거치지 않고 파이썬 함수로 부름. 그러면
    # 기본값 자리에 OptionInfo 객체가 그대로 들어옴. 경로로 쓸 수 있는 것만 받음.
    if not isinstance(ckpt_opt, str):
        ckpt_opt = None

    if ckpt_opt:
        path = ckpt_opt
        if not os.path.exists(path):
            typer.secho(f"[ckpt] --ckpt 로 준 {path} 가 없다.", fg=typer.colors.RED)
            raise typer.Exit(1)
    else:
        path = os.path.join(cfg.out_dir, "best_model.pth")
        if not os.path.exists(path):
            bench = os.path.join(cfg.out_dir, "benchmark_results.json")
            chosen = None
            if os.path.exists(bench):
                with open(bench, encoding="utf-8") as fh:
                    chosen = _json.load(fh).get("selected_arch")
            if chosen and chosen != cfg.model.name:
                typer.secho(
                    f"[ckpt] 벤치마크가 고른 것은 {chosen} 인데 설정은 "
                    f"{cfg.model.name} 이다. 배포 모델과 보고한 성능이 어긋난다: "
                    f"configs 의 model.name 을 {chosen} 으로 바꾸거나 "
                    f"--ckpt 로 쓸 가중치를 직접 지정할 것.",
                    fg=typer.colors.RED)
                raise typer.Exit(1)
            if chosen:
                path = os.path.join(cfg.out_dir, "benchmark", chosen, "fold0", "best_model.pth")

    if not os.path.exists(path):
        typer.secho(
            f"[ckpt] 실을 가중치를 찾지 못했다 ({path}). 학습하지 않은 모델을 "
            "내보내면 무작위 가중치가 그대로 나가고, 그 뒤 측정은 전부 의미가 없다.",
            fg=typer.colors.RED)
        raise typer.Exit(1)

    net.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    typer.echo(f"[ckpt] {path}")
    return path


@app.command()
def export(
    config: str = typer.Option(None),
    onnx: bool = typer.Option(True),
    tensorrt: bool = typer.Option(False),
    ckpt: str = typer.Option(None, help="실을 가중치 경로. 비우면 자동으로 찾는다"),
):
    """학습 모델 -> ONNX(+parity)[ -> TensorRT FP16]."""
    from hippo.export.to_onnx import export_onnx, verify_onnx_parity
    from hippo.export.to_tensorrt import build_tensorrt_engine
    from hippo.models import build_model

    cfg = _cfg(config, None)
    dev = _device()
    net = build_model(cfg.model.name, cfg.model.in_channels, cfg.model.out_channels, tuple(cfg.model.roi))
    _load_deployment_weights(cfg, net, dev, ckpt)
    roi, ic = tuple(cfg.model.roi), cfg.model.in_channels
    onnx_path = os.path.join(cfg.out_dir, "hippocampus.onnx")
    if onnx:
        export_onnx(net, onnx_path, roi, ic, dev)
        parity = verify_onnx_parity(net, onnx_path, roi, ic, dev)
        typer.echo(f"[onnx] {onnx_path}  parity={parity}")
        # parity 를 출력만 하고 넘어가면 변환이 깨져도 아무도 모름.
        # 실패한 ONNX 로 TensorRT 엔진까지 만들어 배포하는 일이 생김.
        if parity.get("ok") is False:
            typer.secho(
                "[onnx] parity 실패: 변환된 모델이 원본과 다르게 동작한다. "
                "이 상태로 배포하면 안 된다.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
    if tensorrt:
        # TensorRT FP16: strongly-typed 엔진을 위해 FP16 ONNX를 별도로 내보낸 뒤 빌드
        onnx16 = os.path.join(cfg.out_dir, "hippocampus_fp16.onnx")
        export_onnx(net, onnx16, roi, ic, dev, half=True)
        eng = os.path.join(cfg.out_dir, "hippocampus_fp16.plan")
        typer.echo(f"[tensorrt] {build_tensorrt_engine(onnx16, eng, fp16=True)}")


@app.command()
def benchmark(config: str = typer.Option(None),
              ckpt: str = typer.Option(None, help="실을 가중치 경로. 비우면 자동으로 찾는다")):
    """PyTorch FP32 vs ONNXRuntime vs TensorRT FP16 지연/처리량."""
    from hippo.export.benchmark import benchmark_backends, summarize
    from hippo.models import build_model

    cfg = _cfg(config, None)
    dev = _device()
    net = build_model(cfg.model.name, cfg.model.in_channels, cfg.model.out_channels, tuple(cfg.model.roi))
    # 지연 자체는 가중치 값과 무관하지만, 여기서 재는 대상이 배포할 모델과
    # 같아야 표의 의미가 삼. 내보내기와 같은 규칙으로 찾음.
    _load_deployment_weights(cfg, net, dev, ckpt)
    onnx_path = os.path.join(cfg.out_dir, "hippocampus.onnx")
    engine_path = os.path.join(cfg.out_dir, "hippocampus_fp16.plan")
    res = benchmark_backends(
        net, onnx_path if os.path.exists(onnx_path) else None,
        tuple(cfg.model.roi), cfg.model.in_channels, dev,
        engine_path=engine_path if os.path.exists(engine_path) else None,
    )
    typer.echo(summarize(res))

    # 측정값을 파일로 남김. README 의 latency 표를 산출물로 검증할 수 있어야 함.
    out = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "model": cfg.model.name,
        "roi": list(cfg.model.roi),
        "batch": 1,
        "backends": res,
    }
    path = os.path.join(cfg.out_dir, "backend_benchmark.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    typer.echo(f"\n저장: {path}")


if __name__ == "__main__":
    app()
