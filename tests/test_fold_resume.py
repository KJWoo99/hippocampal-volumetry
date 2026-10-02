"""fold 단위 이어받기 검증.

학습 중 서버가 꺼지면 회차가 통째로 날아가므로 끝난 fold 를 이어받음. fold 마다 `set_determinism(train.seed)` 로
난수를 다시 고정하므로 fold 는 서로 독립이고, 끝난 fold 를 재사용해도 처음부터 돌린
결과와 같음. 그 재사용이 조건이 같을 때만 일어나는지가 이 파일의 관심사임.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ablation import (  # noqa: E402
    diagnose,
    diverged_folds,
    fold_config_key,
    load_cached_fold,
    same_id_other_name,
    save_fold,
)

REC = {"fold": 0, "best_val_dice": 0.8812, "best_epoch": 120, "epochs_run": 160,
       "stopped_early": True, "history": [{"epoch": 1, "loss": 0.5}]}


def _cfg(**over):
    """실제 Config 와 같은 모양의 설정 묶음.

    `amp`, `val_interval`, `grad_accum`, `lr_patience`, `lr_factor`, `num_workers` 를
    함께 둠. 조건 해시가 이 여섯을 빼면 이 중 하나만 바꿔 같은 회차를 다시 돌릴 때
    앞 조건의 fold 를 이어받음.
    """
    base = dict(arch="segresnet", roi=[64, 64, 64], lr=1e-3, batch=2, wd=1e-5,
                epochs=400, patience=20, aug="spatial", crop_samples=1,
                postproc=True, seed=42, fold_seed=42,
                amp=True, val_interval=2, grad_accum=1, lr_patience=8, lr_factor=0.5,
                num_workers=2)
    base.update(over)
    return SimpleNamespace(
        out_dir="./outputs/ablation_runs/01_baseline",
        model=SimpleNamespace(name=base["arch"], roi=base["roi"]),
        train=SimpleNamespace(lr=base["lr"], batch=base["batch"], weight_decay=base["wd"],
                              epochs=base["epochs"], early_stop_patience=base["patience"],
                              postproc_largest_component=base["postproc"], seed=base["seed"],
                              amp=base["amp"], val_interval=base["val_interval"],
                              grad_accum=base["grad_accum"], lr_patience=base["lr_patience"],
                              lr_factor=base["lr_factor"]),
        data=SimpleNamespace(root="./data", aug=base["aug"],
                             crop_samples=base["crop_samples"],
                             fold_seed=base["fold_seed"],
                             num_workers=base["num_workers"]))


def test_same_config_resumes(tmp_path: Path) -> None:
    key = fold_config_key(_cfg())
    save_fold(tmp_path, 0, REC, key)
    got = load_cached_fold(tmp_path, 0, key)
    assert got == REC


def test_output_location_does_not_break_resume(tmp_path: Path) -> None:
    """저장 위치는 결과를 바꾸지 않으므로 이어받기를 막으면 안 됨.

    회차 이름이 `out_dir` 에 들어가고 데이터 경로도 기계마다 다름. 이것까지
    조건으로 세면 같은 설정인데도 매번 처음부터 돌림.
    """
    save_fold(tmp_path, 0, REC, fold_config_key(_cfg()))
    moved = _cfg()
    moved.out_dir = "./outputs/ablation_runs/99_elsewhere"
    moved.data.root = "/mnt/other/data"
    assert load_cached_fold(tmp_path, 0, fold_config_key(moved)) == REC


def test_changed_config_does_not_resume(tmp_path: Path) -> None:
    # 같은 회차 번호로 조건만 바꿔 다시 돌리는 일이 있음. 그때 옛 fold 를 이어받으면
    # 서로 다른 조건의 결과가 한 회차에 섞임.
    save_fold(tmp_path, 0, REC, fold_config_key(_cfg()))
    for changed in ({"seed": 43}, {"aug": "basic"}, {"lr": 3e-4}, {"postproc": False},
                    {"arch": "swinunetr"}, {"fold_seed": 7}, {"epochs": 250},
                    # 아래 여섯은 손으로 고른 목록에서 빠지기 쉬운 것들임. 전부 결과를
                    # 바꿈: 평가 주기는 best epoch 선택을, grad_accum 은 유효
                    # 배치를, amp 는 수치를, LR 일정은 학습 궤적을, 워커 수는
                    # 증강 난수 순서를 바꿈.
                    {"val_interval": 1}, {"grad_accum": 2}, {"amp": False},
                    {"lr_patience": 4}, {"lr_factor": 0.1}, {"num_workers": 8}):
        assert load_cached_fold(tmp_path, 0, fold_config_key(_cfg(**changed))) is None, changed


def test_missing_and_broken_files_are_ignored(tmp_path: Path) -> None:
    key = fold_config_key(_cfg())
    assert load_cached_fold(tmp_path, 3, key) is None
    # 쓰다 만 파일(크래시 순간). 다시 돌려야지 죽으면 안 됨.
    (tmp_path / "fold_3.json").write_text('{"fold": 3, "best_val', encoding="utf-8")
    assert load_cached_fold(tmp_path, 3, key) is None
    # 필드가 빠진 파일도 이어받지 않음
    (tmp_path / "fold_4.json").write_text(json.dumps({"fold": 4, "config_key": key}),
                                          encoding="utf-8")
    assert load_cached_fold(tmp_path, 4, key) is None


def test_save_is_atomic(tmp_path: Path) -> None:
    # 임시 파일을 남기지 않음. 남으면 다음 실행이 그것을 fold 결과로 오인할 수 있음.
    key = fold_config_key(_cfg())
    save_fold(tmp_path, 1, {**REC, "fold": 1}, key)
    assert {p.name for p in tmp_path.iterdir()} == {"fold_1.json"}


# ── 회차 번호 충돌 ─────────────────────────────────────────────────────

def test_same_id_with_another_name_is_detected(tmp_path: Path) -> None:
    """같은 번호를 다른 이름으로 적어 두면 실험일지에 두 번 실림.

    예: `--name` 만 다른 명령이 다시 돌면 `01_baseline.json` 과 `01_baseline-24gb.json` 이 공존함.
    """
    (tmp_path / "01_baseline.json").write_text("{}", encoding="utf-8")
    assert same_id_other_name(tmp_path, 1, "baseline-24gb").name == "01_baseline.json"


def test_same_id_same_name_is_a_normal_rerun(tmp_path: Path) -> None:
    (tmp_path / "01_baseline.json").write_text("{}", encoding="utf-8")
    assert same_id_other_name(tmp_path, 1, "baseline") is None


def test_other_ids_do_not_clash(tmp_path: Path) -> None:
    (tmp_path / "01_baseline.json").write_text("{}", encoding="utf-8")
    assert same_id_other_name(tmp_path, 2, "lr-3e-4") is None


# ── 수렴 진단 ──────────────────────────────────────────────────────────

def _fold(k, best=0.88, best_ep=100, run=160, early=True, losses=None):
    hist = [{"epoch": i + 1, "loss": (losses[i] if losses else 0.5)} for i in range(run)]
    return {"fold": k, "best_val_dice": best, "best_epoch": best_ep,
            "epochs_run": run, "stopped_early": early, "history": hist}


def test_divergence_is_caught_even_when_the_run_looks_normal():
    """fold 하나가 nan 이면 평균만 보고 '정상 수렴'이라 적으면 안 됨.

    그 fold 만 점수가 낮아 평균이 내려가면 '악화'로 읽히지만, 본 것은 "가끔 발산한다" 쪽임.
    """
    nan = float("nan")
    per = [_fold(0), _fold(1, best=0.85, losses=[0.5] * 80 + [nan] * 80), _fold(2)]
    assert diverged_folds(per) == [1]
    d = diagnose(per, val_interval=2)
    assert "발산" in d and "fold [1]" in d


def test_normal_run_is_not_called_divergence():
    per = [_fold(k) for k in range(3)]
    assert diverged_folds(per) == []
    assert diagnose(per, val_interval=2) == "정상 수렴(정점 후 하락으로 조기종료)"


def test_epoch_cap_still_reported():
    per = [_fold(k, best_ep=158, run=160, early=False) for k in range(3)]
    assert "에폭 한도" in diagnose(per, val_interval=2)


def test_divergence_outranks_the_other_verdicts():
    """발산과 한도 초과가 겹치면 발산을 먼저 말해야 함. 더 심한 쪽임."""
    nan = float("nan")
    per = [_fold(0, best_ep=158, run=160, early=False,
                 losses=[0.5] * 100 + [nan] * 60)]
    assert "발산" in diagnose(per, val_interval=2)


def test_training_code_change_invalidates_cached_folds(tmp_path, monkeypatch):
    """설정이 같아도 학습 코드가 바뀌면 옛 fold 를 이어받지 않음."""
    import run_ablation as ra

    save_fold(tmp_path, 0, REC, fold_config_key(_cfg()))
    assert load_cached_fold(tmp_path, 0, fold_config_key(_cfg())) == REC
    monkeypatch.setattr(ra, "_training_code_hash", lambda: "다른코드")
    assert load_cached_fold(tmp_path, 0, fold_config_key(_cfg())) is None
