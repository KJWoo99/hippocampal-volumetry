"""배포 가중치를 못 찾으면 내보내기를 멈추는지 고정(트러블슈팅 9번).

가중치가 없는데 그냥 넘어가면 무작위 초기화 모델이 ONNX 로 나가고, parity 도 같은 무작위
모델끼리 비교해 통과함. 멈추는 두 분기(가중치 없음, 설정과 벤치마크 아키텍처 불일치)를 고정함.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from hippo import cli
from hippo.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, name: str):
    cfg = load_config(str(REPO / "configs" / "default.yaml"))
    cfg.out_dir = str(tmp_path)
    cfg.model.name = name
    return cfg


def test_export_stops_when_no_weights_exist(tmp_path):
    with pytest.raises(typer.Exit):
        cli._load_deployment_weights(_cfg(tmp_path, "segresnet"), None, "cpu", None)


def test_export_stops_when_benchmark_chose_another_arch(tmp_path):
    (tmp_path / "benchmark_results.json").write_text(
        json.dumps({"selected_arch": "swinunetr"}), encoding="utf-8")
    fold0 = tmp_path / "benchmark" / "swinunetr" / "fold0"
    fold0.mkdir(parents=True)
    (fold0 / "best_model.pth").write_bytes(b"not a checkpoint")
    # 가중치 파일은 있지만 설정의 모델(segresnet)과 다른 구조의 것임. 싣기 전에 멈춰야 함.
    with pytest.raises(typer.Exit):
        cli._load_deployment_weights(_cfg(tmp_path, "segresnet"), None, "cpu", None)


def test_missing_ckpt_option_path_stops(tmp_path):
    with pytest.raises(typer.Exit):
        cli._load_deployment_weights(_cfg(tmp_path, "segresnet"), None, "cpu", str(tmp_path / "없음.pth"))
