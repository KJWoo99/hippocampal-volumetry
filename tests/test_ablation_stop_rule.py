"""절제 종료 판단을 일지 생성기가 규칙대로 세는지 봄(docs/prereg_ablation_extension.md).

개선은 부모 대비로 세고, 추가 회차의 개선은 시드 반복으로 확인함. 셈이 틀리면 일지의
종료 근거가 틀리므로 가짜 기록으로 고정함.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rl():
    spec = importlib.util.spec_from_file_location(
        "render_experiment_log", ROOT / "scripts" / "render_experiment_log.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _r(rid, parent, dice, arch="segresnet", **kw):
    return {"id": rid, "parent": parent, "result": {"mean_val_dice": dice}, "config": {"arch": arch}, **kw}


def test_streak_counts_against_parent_not_baseline(rl):
    # #7 이 기준선보다 크게 오른 뒤 그 위에 얹은 회차는 기준선보다 높아도 부모 대비로는 개선이 아님.
    recs = [_r(1, 0, 0.879), _r(5, 1, 0.878), _r(6, 5, 0.878), _r(7, 5, 0.888),
            _r(8, 7, 0.887), _r(9, 7, 0.887), _r(10, 7, 0.887)]
    rows = rl.stop_trace(recs, 0.0039, 6)
    assert [x["state"] for x in rows] == ["차이 없음", "개선", "차이 없음", "차이 없음", "차이 없음"]
    assert [x["streak"] for x in rows] == [1, 0, 1, 2, 3]


def test_invalid_and_repeat_runs_are_not_counted(rl):
    recs = [_r(1, 0, 0.879), _r(6, 1, 0.879), _r(7, 1, 0.879, invalid="중복"),
            _r(90, 6, 0.899)]
    rows = rl.stop_trace(recs, 0.0039, 6)
    assert [x["id"] for x in rows] == [6]


def test_extension_improvement_waits_for_seed_repeats(rl):
    base = [_r(7, 5, 0.8877), _r(92, 7, 0.8865), _r(93, 7, 0.8864), _r(5, 1, 0.878)]
    rows = rl.stop_trace(base + [_r(14, 7, 0.8930)], 0.0039, 14, confirm=True)
    assert rows[0]["state"] == "확인 대기" and rows[0]["streak"] == 0


def test_extension_improvement_that_vanishes_on_repeat_counts_as_no_change(rl):
    # 한 번 높게 나왔지만 세 시드 평균이 #7 계열과 판정선 안이면 운임.
    recs = [_r(7, 5, 0.8877), _r(92, 7, 0.8865), _r(93, 7, 0.8864), _r(5, 1, 0.878),
            _r(14, 7, 0.8930), _r(94, 14, 0.8866), _r(95, 14, 0.8860)]
    rows = rl.stop_trace(recs, 0.0039, 14, confirm=True)
    assert rows[0]["state"] == "차이 없음" and rows[0]["streak"] == 1


def test_confirmed_improvement_resets_streak(rl):
    recs = [_r(7, 5, 0.8877), _r(92, 7, 0.8865), _r(93, 7, 0.8864), _r(5, 1, 0.878),
            _r(14, 7, 0.8870), _r(15, 7, 0.8930), _r(94, 15, 0.8925), _r(95, 15, 0.8920)]
    rows = rl.stop_trace(recs, 0.0039, 14, confirm=True)
    assert [(x["id"], x["state"], x["streak"]) for x in rows] == [
        (14, "차이 없음", 1), (15, "개선", 0)]


ABL = ROOT / "outputs" / "ablation"


@pytest.mark.skipif(not (ABL / "10_lr-3e-4.json").exists(), reason="절제 기록 없음")
def test_first_series_reaches_stop_at_10_when_counted_against_parent(rl):
    # 일지와 사전등록 문서가 "부모 대비로 세면 #10 에서 찬다" 고 적음.
    recs = sorted((json.loads(p.read_text(encoding="utf-8")) for p in ABL.glob("*.json")),
                  key=lambda r: r["id"])
    rows = rl.stop_trace(recs, 0.0039, 6, 14)
    assert next(x["id"] for x in rows if x["streak"] >= 3) == 10


def test_confirmed_extension_improvement_becomes_adopted_setting(rl):
    base = [_r(1, 0, 0.879), _r(90, 1, 0.880), _r(91, 1, 0.878)]
    recs = base + [_r(7, 1, 0.8877), _r(92, 7, 0.8865), _r(93, 7, 0.8864),
                   _r(14, 7, 0.8880), _r(15, 7, 0.8930), _r(94, 15, 0.8925), _r(95, 15, 0.8920)]
    valid = [r for r in recs if r["id"] < 90]
    assert rl.adopted_setting(recs, valid, 0.0039)["id"] == 15
    # 확인 전(반복 없음)이면 점추정이 높아도 설정은 그대로임.
    no_rep = [r for r in recs if r["id"] not in (94, 95)]
    assert rl.adopted_setting(no_rep, valid, 0.0039)["id"] == 7


def test_first_series_winner_needs_threshold_and_seed_confirmation(rl):
    # 판정선 안의 최고값은 기준선을 이기지 못함.
    recs = [_r(1, 0, 0.879), _r(90, 1, 0.880), _r(91, 1, 0.878), _r(2, 1, 0.881)]
    assert rl.adopted_setting(recs, [r for r in recs if r["id"] < 90], 0.0039)["id"] == 1
    # 판정선을 넘어도 시드 반복이 없으면 기준선임.
    recs = [_r(1, 0, 0.879), _r(90, 1, 0.880), _r(91, 1, 0.878), _r(7, 1, 0.890)]
    assert rl.adopted_setting(recs, [r for r in recs if r["id"] < 90], 0.0039)["id"] == 1
    # 반복까지 판정선을 넘으면 채택함. 반복에서 사라지면 기준선임.
    ok = recs + [_r(92, 7, 0.889), _r(93, 7, 0.888)]
    assert rl.adopted_setting(ok, [r for r in ok if r["id"] < 90], 0.0039)["id"] == 7
    gone = recs + [_r(92, 7, 0.878), _r(93, 7, 0.877)]
    assert rl.adopted_setting(gone, [r for r in gone if r["id"] < 90], 0.0039)["id"] == 1


def test_architecture_change_is_not_a_setting_candidate(rl):
    # 구조를 바꾼 회차(#6)는 절제의 "쓰는 설정" 후보가 아님. 구조는 벤치마크가 정함.
    recs = [_r(1, 0, 0.879), _r(90, 1, 0.880), _r(91, 1, 0.878),
            _r(6, 1, 0.899, arch="swinunetr"), _r(96, 6, 0.898, arch="swinunetr"), _r(97, 6, 0.897, arch="swinunetr")]
    assert rl.adopted_setting(recs, [r for r in recs if r["id"] < 90], 0.0039)["id"] == 1


def test_extension_only_counts_when_its_parent_is_the_adopted_setting(rl):
    # 쓰는 설정이 #7 이 아닌데 #7 을 부모로 둔 추가 회차가 이겨도 설정을 바꾸지 않음.
    recs = [_r(1, 0, 0.879), _r(90, 1, 0.880), _r(91, 1, 0.878), _r(7, 1, 0.8800),
            _r(15, 7, 0.8930), _r(94, 15, 0.8925), _r(95, 15, 0.8920)]
    assert rl.adopted_setting(recs, [r for r in recs if r["id"] < 90], 0.0039)["id"] == 1


@pytest.mark.skipif(not (ABL / "16_aug-spatial-geo.json").exists(), reason="절제 기록 없음")
def test_adopted_setting_is_what_config_yaml_uses(rl):
    # 일지가 "쓰는 설정" 으로 적는 회차의 증강이 configs/default.yaml 과 같아야 함. 어느 회차인지는 규칙이 정함.
    from hippo.config import load_config

    recs = sorted((json.loads(p.read_text(encoding="utf-8")) for p in ABL.glob("*.json")),
                  key=lambda r: r["id"])
    valid = [r for r in recs if not r.get("invalid") and r["id"] < 90]
    adopted = rl.adopted_setting(recs, valid, 0.0039)
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    assert adopted["config"]["aug"] == cfg.data.aug


def test_extension_rounds_on_another_parent_are_listed_but_not_counted(rl):
    # 쓰는 설정이 #11 이면 #7 위에서 돌린 #14~#16 은 표에만 남고, 연속 미개선은 #17 부터 셈
    # (docs/prereg_pair_split.md 두 번째 추가 절 2번).
    recs = [_r(7, 1, 0.8860), _r(11, 7, 0.8872),
            _r(14, 7, 0.8833), _r(15, 7, 0.8882), _r(16, 7, 0.8855),
            _r(17, 11, 0.8869), _r(18, 11, 0.8883), _r(19, 11, 0.8880)]
    rows = rl.stop_trace(recs, 0.0039, 14, confirm=True, root=11)
    assert [x["counted"] for x in rows] == [False, False, False, True, True, True]
    assert [x["streak"] for x in rows if x["counted"]] == [1, 2, 3]
    assert all("셈에서 뺌" in x["note"] for x in rows if not x["counted"])


@pytest.mark.skipif(not (ABL / "19_aug-spatial-wide-geo.json").exists(), reason="절제 기록 없음")
def test_real_extension_closes_at_19_counting_from_17(rl):
    recs = sorted((json.loads(p.read_text(encoding="utf-8")) for p in ABL.glob("*.json")),
                  key=lambda r: r["id"])
    valid = [r for r in recs if not r.get("invalid") and r["id"] < 90]
    first = rl.first_round_setting(recs, valid, 0.0039)
    assert first["id"] == 11
    rows = rl.stop_trace(recs, 0.0039, 14, confirm=True, root=first["id"])
    assert [x["id"] for x in rows if not x["counted"]] == [14, 15, 16]
    assert next(x["id"] for x in rows if x["counted"] and x["streak"] >= 3) == 19


def test_confirmed_improvement_moves_the_counted_parent(rl):
    # 확인된 개선(#15)이 나오면 그 뒤로는 #15 위의 회차만 셈. 옛 부모 #7 위의 #16 은 빠지고 #15 위의 #17 이 들어감.
    recs = [_r(7, 5, 0.8877), _r(92, 7, 0.8865), _r(93, 7, 0.8864), _r(5, 1, 0.878),
            _r(14, 7, 0.8870), _r(15, 7, 0.8930), _r(94, 15, 0.8925), _r(95, 15, 0.8920),
            _r(16, 7, 0.8875), _r(17, 15, 0.8925)]
    rows = rl.stop_trace(recs, 0.0039, 14, confirm=True, root=7)
    got = {x["id"]: (x["counted"], x["state"], x["streak"]) for x in rows}
    assert got[14] == (True, "차이 없음", 1)
    assert got[15] == (True, "개선", 0)
    assert got[16][0] is False
    assert got[17] == (True, "차이 없음", 1)
