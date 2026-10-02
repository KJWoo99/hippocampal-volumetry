"""절제실험 기록들을 사람이 읽는 실험 일지로 만듦.

    python scripts/render_experiment_log.py

`outputs/ablation/*.json` 을 모아 `docs/EXPERIMENT_LOG.md` 를 씀.

각 실험은 가설, 바꾼 것, 관찰, 판단을 함께 싣고, 부모 실험을 가리켜 어떤 결과가 다음 선택을 낳았는지 보이게 함.

판정선은 여기서 정하지 않음. `run_ablation.py` 가 실험 전에 정해 기록마다 `meaningful_threshold` 로 남긴 값을
읽기만 함(두 파일이 각자 상수를 들면 경고 없이 어긋남).
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABL_DIR = ROOT / "outputs" / "ablation"
OUT_MD = ROOT / "docs" / "EXPERIMENT_LOG.md"
# 90번 이상은 설정을 바꾼 실험이 아니라 같은 설정을 시드만 바꿔 다시 돌린 회차임.
# 순위, 판정에서 빼고, 부모 회차와의 차이로 "시드만 바꿔도 이만큼 움직인다"를 보임.
# mimic, sepsis 와 같은 규약임.
REPEAT_ID_FROM = 90
# 사전등록(docs/prereg_ablation_extension.md)으로 더 돌린 회차의 첫 번호. 이 번호부터 연속 미개선을 새로 셈.
# 그 앞(#6~#13)은 종료 판단 없이 돌린 회차라 부모 대비로 다시 센 값을 따로 보임.
EXTENSION_FROM = 14
MANDATORY = 5          # 의무 구간 #1~#5. 연속 미개선은 #6 부터 셈
STOP_AFTER = 3         # 개선 아닌 회차가 이만큼 연속이면 멈춤


# 문서에 쓰지 않기로 한 문자들. 여기서는 찾아 바꾸는 대상이라, 문서 정리 때 이 줄까지 바뀌지 않게
# 이스케이프로 적어 둠.
_DASH, _DOT, _PM = "\u2014", "\u00b7", "\u00b1"
_ARROWS = {"\u2192": "->", "\u2190": "<-", "\u2191": "^", "\u2193": "v", "\u00d7": "x"}



def _md_tilde(text: str) -> str:
    """GitHub 는 한 블록 안의 ~ 두 개를 취소선으로 그림. 코드 밖의 ~ 를 \\~ 로 적음."""
    out, fence = [], False
    for ln in text.split("\n"):
        if ln.lstrip().startswith("```"):
            fence = not fence
        elif not fence:
            parts = re.split(r"(`[^`]*`)", ln)
            ln = "".join(p if i % 2 else re.sub(r"(?<!\\)~", r"\\~", p) for i, p in enumerate(parts))
        out.append(ln)
    return "\n".join(out)

def plain(text: str) -> str:
    """기록에서 가져온 문장을 문서 표기 규칙에 맞춤.

    절제 기록은 산출물이라 손대지 않음. 옛 회차의 문장에 줄표나 가운뎃점이
    남아 있어도, 문서로 옮길 때는 쉼표와 콜론으로 바꿔 적음.
    """
    text = re.sub(rf"\s*{_DASH}\s*", ": ", text)
    text = re.sub(rf"\s*{_DOT}\s*", ", ", text)
    for a, b in _ARROWS.items():
        text = text.replace(a, b)
    text = re.sub(rf"\s*{_PM}\s*", " +/- ", text)
    return text.replace("( +/- ", "(+/- ")


def _load() -> list[dict]:
    if not ABL_DIR.exists():
        return []
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ABL_DIR.glob("*.json"))]
    return sorted(recs, key=lambda r: r["id"])


def _mean(r: dict) -> float:
    return r["result"]["mean_val_dice"]


def _std(r: dict) -> float:
    return r["result"].get("std_val_dice", 0.0)


def _verdict(delta: float | None, thr: float) -> str:
    if delta is None:
        return "기준선"
    if delta >= thr:
        return f"개선 ({delta:+.4f})"
    if delta <= -thr:
        return f"악화 ({delta:+.4f})"
    return f"차이 없음 ({delta:+.4f})"


def _series_mean(rid: int, recs: list[dict]) -> tuple[float, int]:
    """회차와 그 시드 반복들의 평균, 개수."""
    vals = [_mean(r) for r in recs
            if r["id"] == rid or (r["id"] >= REPEAT_ID_FROM and r.get("parent") == rid)]
    return st.fmean(vals), len(vals)


def stop_trace(recs: list[dict], thr: float, start: int, end: int | None = None,
               confirm: bool = False, root: int | None = None) -> list[dict]:
    """부모 대비 판정으로 연속 미개선을 셈.

    confirm=True 면 개선으로 찍힌 회차를 시드 반복으로 확인함(사전등록 4번). 반복이 둘
    이상 있고 세 시드 평균이 부모 계열 평균보다 판정선 이상 높아야 개선으로 남음. 반복이
    아직 없으면 "확인 대기" 로 두고 카운트를 건드리지 않음.

    root 를 주면 부모가 그때의 쓰는 설정(처음에는 root, 확인된 개선이 나오면 그 회차)인 회차만
    셈. 다른 부모 위의 회차는 표에 남기되 셈에서 뺌(counted=False). 짝 단위 분할에서 쓰는 설정이
    #11 이 되어, #7 위에서 돌린 #14~#16 은 기록만 남기고 #17 부터 센다(docs/prereg_pair_split.md
    두 번째 추가 절 2번).
    """
    by_id = {r["id"]: r for r in recs}
    rows, streak, current = [], 0, root
    for r in recs:
        rid = r["id"]
        if rid < start or rid >= REPEAT_ID_FROM or (end is not None and rid >= end):
            continue
        if r.get("invalid") or r.get("parent") not in by_id:
            continue
        par = by_id[r["parent"]]
        delta = _mean(r) - _mean(par)
        state = "개선" if delta >= thr else ("악화" if delta <= -thr else "차이 없음")
        note = ""
        if state == "개선" and confirm:
            m, n = _series_mean(rid, recs)
            pm, pn = _series_mean(par["id"], recs)
            if n < 3:
                state, note = "확인 대기", f"시드 반복 {n - 1}/2"
            elif m - pm >= thr:
                note = f"세 시드 평균 {m:.4f}, 부모 계열 {pm:.4f} ({m - pm:+.4f})"
            else:
                state = "차이 없음"
                note = f"시드 반복에서 사라짐: 세 시드 평균 {m:.4f}, 부모 계열 {pm:.4f} ({m - pm:+.4f})"
        counted = current is None or par["id"] == current
        if not counted:
            note = f"셈에서 뺌: 쓰는 설정이 아닌 부모(#{par['id']}) 위의 회차" + (f", {note}" if note else "")
        elif state == "개선":
            streak = 0
            if current is not None:
                current = rid
        elif state != "확인 대기":
            streak += 1
        rows.append({"id": rid, "parent": par["id"], "delta": delta, "state": state,
                     "note": note, "streak": streak, "counted": counted})
    return rows


def adopted_setting(recs: list[dict], valid: list[dict], thr: float) -> dict:
    """설정 파일에 들어갈 회차(docs/prereg_pair_split.md 의 규칙).

    1. 후보는 처음 회차(#13 까지) 가운데 기준선과 구조가 같은 회차임. 구조를 바꾼 회차(#6)는 뺌. 구조는 아키텍처
       벤치마크가 정함.
    2. 후보 최고값이 기준선보다 판정선 이상 높고, 그 설정과 기준선을 각각 세 시드로 돌린 평균끼리도 판정선 이상
       차이가 나야 채택함. 시드 반복이 모자라면 기준선을 씀.
    3. 추가 회차는 쓰는 설정을 부모로 둔 회차가 시드 반복까지 확인된 개선을 냈을 때만 설정을 바꿈.
    """
    adopted = first_round_setting(recs, valid, thr)
    by_id = {r["id"]: r for r in recs}
    for x in stop_trace(recs, thr, EXTENSION_FROM, confirm=True, root=adopted["id"]):
        if x["counted"] and x["state"] == "개선":
            adopted = by_id[x["id"]]
    return adopted


def first_round_setting(recs: list[dict], valid: list[dict], thr: float) -> dict:
    """처음 회차(#13 까지)만 보고 정한 쓰는 설정. adopted_setting 의 1, 2번."""
    base = next(r for r in valid if not r.get("parent"))
    arch = base.get("config", {}).get("arch")
    cands = [r for r in valid if r["id"] < EXTENSION_FROM and r.get("config", {}).get("arch") == arch]
    best = max(cands, key=_mean)
    adopted = base
    if best["id"] != base["id"] and _mean(best) - _mean(base) >= thr:
        m, n = _series_mean(best["id"], recs)
        bm, bn = _series_mean(base["id"], recs)
        if n >= 3 and bn >= 3 and m - bm >= thr:
            adopted = best
    return adopted


def main() -> int:
    recs = _load()
    # 무효로 표시된 회차(`invalid` 필드)는 경위에는 남기되 결론 계산에서는 뺌.
    # 지우면 "왜 #4 가 없나"가 되고, 넣으면 기준선 복제가 순위에 끼어듦.
    valid = [r for r in recs if not r.get("invalid") and r["id"] < REPEAT_ID_FROM]
    repeats = [r for r in recs if r["id"] >= REPEAT_ID_FROM]
    if not recs:
        print(f"기록이 없다: {ABL_DIR}")
        return 1

    thresholds = {r.get("meaningful_threshold") for r in recs}
    thresholds.discard(None)
    if len(thresholds) > 1:
        print(f"[경고] 기록마다 판정선이 다르다: {sorted(thresholds)}", file=sys.stderr)
    thr = min(thresholds) if thresholds else 0.0039

    base = next((r for r in valid if not r.get("parent")), valid[0])
    base_score = _mean(base)

    lines: list[str] = []
    lines.append("# 실험 일지: 무엇을 바꿨더니 어떻게 됐는가\n")
    lines.append(
        "자동 생성 파일이다. `python scripts/render_experiment_log.py` 로 다시 만든다.\n")
    lines.append(
        "모든 수치는 개발셋 val 기준이다. 홀드아웃 52건은 절제실험이 불러오지도\n"
        "않는다. 탐색을 홀드아웃으로 하면 열 몇 번 돌린 시행들의 운이 그 점수에 섞여,\n"
        "더 이상 미래 성능의 추정치가 아니게 된다. 이 저장소는 그 함정(winner's curse)을\n"
        "이미 한 번 잡은 적이 있다.\n")
    # 가설은 회차를 처음 설계할 때(에폭 한도 250) 적은 문장을 그대로 실음. 뒤에 한도를
    # 400 으로 올려 같은 설정을 다시 돌렸으므로 가설 속 수치와 한도는 표의 값과 다를 수 있음.
    # 가설을 결과에 맞춰 고치면 "무엇을 보고 다음을 정했나" 의 기록이 사라지므로 두고 알림.
    lines.append(
        "가설은 그 회차를 처음 설계할 때(에폭 한도 250) 적은 것이다. 뒤에 한도를 400 으로\n"
        "올려 전부 다시 돌렸으므로, 가설 속 수치(예: #10 의 \"0.0134 악화\")와 한도는 아래 결과와\n"
        "다를 수 있다. 판단의 기록이라 고치지 않는다.\n")
    lines.append(
        f"기준선 = 실험 #{base['id']} ({base['name']}), "
        f"val Dice {base_score:.4f} +/- {_std(base):.4f} "
        f"(fold {base['result'].get('n_folds', '?')}개)\n")

    # ── 요약표 ────────────────────────────────────────────────────────
    lines.append("## 한눈에 보기\n")
    lines.append("| # | 바꾼 것 | val Dice (평균 +/- 표준편차) | 기준선 대비 | 부모 대비 | best/실행 | 진단 |")
    lines.append("|---|---|---:|---|---|---:|---|")
    by_id = {r["id"]: r for r in recs}
    trials = [r for r in recs if r["id"] < REPEAT_ID_FROM]   # 무효 포함, 시드 반복 제외
    for r in trials:
        res = r["result"]
        d = None if r["id"] == base["id"] else _mean(r) - base_score
        verdict = "무효 (설정이 기준선과 동일)" if r.get("invalid") else _verdict(d, thr)
        par = by_id.get(r.get("parent"))
        if r.get("invalid"):
            vs_parent = "무효"
        elif par:
            vs_parent = f"#{par['id']} " + _verdict(_mean(r) - _mean(par), thr)
        else:
            vs_parent = "기준선"
        lines.append(
            f"| {r['id']} | {plain(r['change']) or '없음(기준선)'} | "
            f"{_mean(r):.4f} +/- {_std(r):.4f} | "
            f"{verdict} | {vs_parent} | {res['best_epoch']}/{res['epochs_run']} | {plain(r['diagnosis'])} |")
    lines.append("")

    # ── 실험별 경위 ───────────────────────────────────────────────────
    lines.append("## 실험별 경위\n")
    for r in trials:
        res, cfg = r["result"], r["config"]
        d = None if r["id"] == base["id"] else _mean(r) - base_score
        lines.append(f"### 실험 #{r['id']}: {r['name']}\n")
        if r.get("invalid"):
            lines.append(f"- 무효: {r['invalid']}")
        if r.get("parent"):
            lines.append(f"- 부모 회차: 실험 #{r['parent']}")
        if r.get("hypothesis"):
            lines.append(f"- 가설: {plain(r['hypothesis'])}")
        lines.append(f"- 바꾼 것: {plain(r['change']) or '없음(기준선)'}")
        scores = res.get("fold_scores") or []
        lines.append(
            f"- 결과: val Dice {_mean(r):.4f} +/- {_std(r):.4f} ({_verdict(d, thr)})"
            + (f"\n  - fold별: {', '.join(f'{x:.4f}' for x in scores)}" if scores else ""))
        lines.append(
            f"- 학습 길이: best {res['best_epoch']}에폭 / 실행 {res['epochs_run']}에폭 "
            f"(한도 {cfg['epochs']}), 전 fold 조기종료 {res['stopped_early']}")
        lines.append(f"- 진단: {plain(r['diagnosis'])}")
        lines.append(
            f"- 설정: arch={cfg['arch']} batch={cfg['batch']} lr={cfg['lr']:g} "
            f"wd={cfg['weight_decay']:g} roi={cfg['roi']} "
            f"crop_samples={cfg['crop_samples']} patience={cfg['patience']} "
            f"seed={cfg['seed']} aug={cfg.get('aug')} "
            f"postproc={cfg.get('postproc_largest_component')}")
        lines.append(f"- 소요: {r['elapsed_sec'] / 60:.1f}분")
        lines.append("")

    # ── 결론 ──────────────────────────────────────────────────────────
    best = max(valid, key=_mean)
    adopted = adopted_setting(recs, valid, thr)
    lines.append("## 여기까지의 결론\n")
    lines.append(
        f"- 쓰는 설정: 실험 #{adopted['id']} ({adopted['name']}) "
        f"val Dice {_mean(adopted):.4f} +/- {_std(adopted):.4f} (`configs/default.yaml`)")
    if best["id"] != adopted["id"]:
        lines.append(
            f"- 점추정 1위는 #{best['id']} ({best['name']}) {_mean(best):.4f} 지만 쓰는 설정과 "
            f"{_mean(best) - _mean(adopted):+.4f} 로 판정선 안이다. 추가 회차는 시드 반복으로 확인된 "
            "개선이 있을 때만 설정을 바꾼다(사전등록 5번).")
    first_set = first_round_setting(recs, valid, thr)
    if first_set["id"] != base["id"]:
        m, n = _series_mean(first_set["id"], recs)
        bm, bn = _series_mean(base["id"], recs)
        lines.append(
            f"- 채택 근거: 처음 회차 최고 #{first_set['id']} 계열 {n}회 평균 {m:.4f}, 기준선 계열 {bn}회 평균 "
            f"{bm:.4f} ({m - bm:+.4f}, 판정선 {thr:.4f}). 시드를 바꿔도 판정선 이상 차이가 남아 채택했다"
            "(docs/prereg_pair_split.md 두 번째 추가 절 1번).")
    improved =[r for r in valid if r["id"] != base["id"] and _mean(r) - base_score >= thr]
    worsened = [r for r in valid if _mean(r) - base_score <= -thr]
    lines.append(f"- 기준선보다 나아진 실험: {len(improved)}건 "
                 f"{[r['name'] for r in improved] or '없음'}")
    lines.append(f"- 오히려 나빠진 실험: {len(worsened)}건 "
                 f"{[r['name'] for r in worsened] or '없음'}")
    lines.append(f"- 총 {len(trials)}회 실험 + 시드 반복 {len(repeats)}회, 누적 "
                 f"{sum(r['elapsed_sec'] for r in recs) / 3600:.1f}시간")
    lines.append("")

    # ── 종료 판단 ─────────────────────────────────────────────────────
    # 규칙: 의무 5회 뒤, 부모 대비 개선이 아닌 회차가 3회 연속이면 멈춤.
    # 처음 회차는 멈추는 판단 없이 돌았으므로 그 사실과 부모 대비로 다시 센 값을 함께 적음.
    lines.append("## 종료 판단\n")
    lines.append(
        f"규칙은 의무 {MANDATORY}회 뒤 개선이 아닌 회차가 {STOP_AFTER}회 연속이면 멈추는 것이다. "
        "개선은 부모 회차 대비로 센다.\n")
    first = stop_trace(recs, thr, MANDATORY + 1, EXTENSION_FROM)
    hit = next((x for x in first if x["streak"] >= STOP_AFTER), None)
    lines.append(
        f"처음 회차(#{MANDATORY + 1}~#{EXTENSION_FROM - 1})는 종료 판단 없이 한꺼번에 돌린 회차다. 처음에는 "
        "기준선 대비로 세어 연속 미개선이 쌓이지 않았고, 짝 단위 분할로 다시 돌릴 때는 추가 회차까지 "
        "19회를 대기열에 한꺼번에 넣었다. 아래는 부모 대비로 다시 센 값이다(결과를 본 뒤 센 것).\n")
    lines.append("| # | 부모 | 부모 대비 | 판정 | 연속 미개선 |")
    lines.append("|---|---|---:|---|---:|")
    for x in first:
        lines.append(f"| {x['id']} | #{x['parent']} | {x['delta']:+.4f} | {x['state']} | {x['streak']} |")
    lines.append("")
    if hit:
        lines.append(f"> 부모 대비로 세면 #{hit['id']} 에서 {STOP_AFTER}회가 찬다.\n")
    ext = stop_trace(recs, thr, EXTENSION_FROM, confirm=True, root=first_set["id"])
    if ext:
        lines.append(
            f"#{EXTENSION_FROM} 부터는 규칙을 먼저 정하고 돌렸다"
            "([docs/prereg_ablation_extension.md](prereg_ablation_extension.md)). "
            "개선으로 찍히면 시드 반복 두 번으로 확인한다.\n")
        skipped = [x for x in ext if not x["counted"]]
        if skipped:
            first_counted = next((x["id"] for x in ext if x["counted"]), None)
            lines.append(
                f"쓰는 설정(#{first_set['id']})이 아닌 부모 위에서 돌린 회차"
                f"({', '.join('#' + str(x['id']) for x in skipped)})는 기록만 남기고 셈에서 뺐다"
                + (f". 연속 미개선은 #{first_counted} 부터 센다" if first_counted else "")
                + "(docs/prereg_pair_split.md 두 번째 추가 절 2번).\n")
        lines.append("| # | 부모 | 부모 대비 | 판정 | 비고 | 연속 미개선 |")
        lines.append("|---|---|---:|---|---|---:|")
        for x in ext:
            lines.append(f"| {x['id']} | #{x['parent']} | {x['delta']:+.4f} | {x['state']} | "
                         f"{x['note'] or '없음'} | {x['streak'] if x['counted'] else '셈 안 함'} |")
        lines.append("")
        done = next((x for x in ext if x["counted"] and x["streak"] >= STOP_AFTER), None)
        if done:
            lines.append(f"> #{done['id']} 에서 연속 미개선 {STOP_AFTER}회가 차 절제를 닫았다.\n")
        else:
            lines.append(f"> 진행 중이다. 지금 연속 미개선 {ext[-1]['streak']}회.\n")

    # ── 판정선이 실측 변동과 맞는지 ────────────────────────────────────
    observed = [_std(r) for r in recs if _std(r) > 0]
    lines.append(
        f"> 변화가 {thr:.4f} 미만이면 \"차이 없음\"으로 적었다. 이 선은 기존 벤치마크의\n"
        "> fold 간 표준편차에서 가져와 실험 전에 못박은 값이다. 결과를 보고 옮기지\n"
        "> 않으려고 기록마다 `meaningful_threshold` 로 함께 저장한다.\n")
    if observed:
        lines.append(
            f"> 이번 실험들에서 실제로 관측된 fold 간 표준편차는 평균 "
            f"{st.fmean(observed):.4f} (최소 {min(observed):.4f}, 최대 {max(observed):.4f}) 다."
            + ("\n> 미리 정한 선이 실측 변동보다 빡빡하다. 그래도 옮기지 않는다: "
               "결과를 보고 기준을 고치면 사전에 정한 의미가 사라진다.\n"
               if st.fmean(observed) > thr else "\n"))

    # ── 1위가 2, 3위와 구분되는가 ──────────────────────────────────────
    #
    # "가장 좋았던 설정"만 적으면 뚜렷한 승자처럼 읽힘. 상위 몇 개가 판정선 안에 몰려 순서가 우연으로 갈릴 수
    # 있으므로, "쓰는 설정" 과 판정선 안에 있는 설정을 함께 적음. 추가 회차의 점추정이 조금 높다고 설정을 바꾸면
    # 그 뒤 벤치마크, 배포가 전부 시드 운 위에 서기 때문임(사전등록 5번).
    tied = [r for r in valid
            if r["id"] != adopted["id"] and abs(_mean(adopted) - _mean(r)) < thr]
    if tied:
        names = ", ".join(f"#{r['id']} {r['name']} {_mean(r):.4f}" for r in tied)
        lines.append(
            f"> 1위를 단독 승자로 읽으면 안 된다. 쓰는 설정 #{adopted['id']} 과 판정선({thr:.4f}) "
            f"안에 있는 설정이 {len(tied)}건 더 있다: {names}.\n"
            f"> 차이는 최대 {max(abs(_mean(adopted) - _mean(r)) for r in tied):.4f} 로, fold 간 표준편차"
            f"({_std(adopted):.4f})보다 작다. 이 중에서 무엇을 고르는지는 근거가 없으므로 기준선 대비\n"
            "> 개선만 결론으로 삼는다. 설정은 처음 회차(#13 까지) 최고값이 기준선보다 세 시드 평균으로도\n"
            "> 판정선 이상 높을 때 그 설정을 쓰고, 추가 회차는 시드 반복으로 확인된 개선이 있을 때만 바꾼다.\n")

    # ── 시드 반복 ─────────────────────────────────────────────────────
    if repeats:
        by_id = {r["id"]: r for r in recs}
        lines.append("## 시드만 바꿔 다시 돌린 회차\n")
        lines.append("fold 분할은 고정(`data.fold_seed`)하고 초기값, 증강 난수만 바꿨다.\n")
        lines.append("| # | 부모 | 시드 | val Dice | 부모 대비 |")
        lines.append("|---|---|---:|---:|---:|")
        groups: dict[int, list[float]] = {}
        for r in repeats:
            par = by_id.get(r.get("parent"))
            delta = f"{_mean(r) - _mean(par):+.4f}" if par else ": "
            lines.append(f"| {r['id']} | #{r.get('parent')} | {r['config'].get('seed')} | "
                         f"{_mean(r):.4f} +/- {_std(r):.4f} | {delta} |")
            if par:
                groups.setdefault(par["id"], [_mean(par)]).append(_mean(r))
        lines.append("")
        for pid, vals in groups.items():
            if len(vals) >= 2:
                sd = st.stdev(vals)
                tail = ("보다 크다: 이 크기 이하의 설정 차이는 시드 운과 구분되지 않는다."
                        if sd > thr else "보다 작다.")
                lines.append(f"> #{pid} 과 그 시드 반복 {len(vals) - 1}회의 표준편차는 "
                             f"{sd:.4f} 다. 판정선 {thr:.4f} {tail}\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    # 줄끝은 LF 로 고정함(플랫폼 기본값에 맡기면 CRLF 가 되어 문서 전체가 바뀐 것으로 보임).
    OUT_MD.write_text(_md_tilde("\n".join(lines)), encoding="utf-8", newline="\n")
    print(f"[저장] {OUT_MD}  (실험 {len(recs)}건, 판정선 {thr})")
    return 0


if __name__ == "__main__":
    import argparse

    # 인자는 없음. 읽어 두어야 --help 가 설명만 찍고 끝남.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
