"""문서 표의 수치를 산출물에서 채움.

    python scripts/sync_doc_numbers.py            # 고침
    python scripts/sync_doc_numbers.py --check    # 다른 곳만 알려주고 고치지 않음

README 와 모델카드에는 같은 값이 여러 곳에 적혀 있어, 재실행할 때마다 손으로 옮기면 한 곳이 남음.
여기서는 표의 행 이름을 열쇠로 값 칸만 바꿈. 굵게 표시나 주석은 건드리지 않음. 행 이름이 문서에 없으면
그냥 넘어가지 않고 알림.

## 범위: 표만 봄

산문, 인용문, 트러블슈팅 서술은 보지 않음. `--check` 가 통과해도 문서 전체가 산출물과 같다는 뜻이 아님.
표 밖은 산출물에서 나오지 않는 수치를 줄 단위로 찾는 별도 점검으로 봄.

CV 표(구조별 Dice)처럼 행 이름이 모델 이름인 표도 같은 방식으로 채움.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DOCS = [ROOT / "README.md", ROOT / "docs" / "report.md", ROOT / "docs" / "MODEL_CARD.md"]


def minus(s: str) -> str:
    """문서는 유니코드 빼기표(−)를 씀."""
    return s.replace("-", "−")


def build_values() -> dict[str, str]:
    d = json.loads((OUT / "benchmark_results.json").read_text(encoding="utf-8"))
    h = d["holdout_result"]
    ea = h["ensemble_agreement"]
    ba = ea["bland_altman"]
    hs = h["holdout_summary"]
    pc = h.get("per_ckpt_agreement", {})
    ref = ea["reference_mean"]

    v = {
        "Dice": f"{hs['dice']['mean']:.4f} +/- {hs['dice']['std']:.4f}",
        "HD95": f"{hs['hd95']['mean']:.2f}",
        "ASD": f"{hs['asd']['mean']:.3f}",
        "ICC(2,1)": f"{ea['icc_2_1']:.4f}",
        "Pearson r": f"{ea['pearson_r']:.4f}",
        "bias (예측−정답)": f"{ba['bias']:+.1f} mm³ ({ba['bias'] / ref * 100:+.2f}%)",
        "95% LoA": minus(f"[{ba['loa_lower']:.1f}, +{ba['loa_upper']:.1f}] mm³") +
                   f" (폭 {ba['loa_upper'] - ba['loa_lower']:.1f})",
        "LoA 상한 95% CI": f"[{ba['loa_upper_ci'][0]:.1f}, {ba['loa_upper_ci'][1]:.1f}]",
        "LoA 하한 95% CI": minus(f"[{ba['loa_lower_ci'][0]:.1f}, {ba['loa_lower_ci'][1]:.1f}]"),
        "LoA 반폭": f"참조 평균({ref:,.0f}mm³)의 {ea['loa_halfwidth_pct_of_reference']:.1f}%",
        "평균 절대 백분율 오차": f"{ba['mean_pct_error']:.2f}%",
    }
    if pc:
        v["체크포인트별 ICC"] = f"{pc['icc_2_1']['mean']:.4f} +/- {pc['icc_2_1']['std']:.4f}"

    # 구조별 CV 표: 행 이름이 모델 이름이고 칸이 Dice, HD95, ASD 로 셋임.
    for arch, res in d["cv_results"].items():
        s = res["cv_summary"]
        v[f"cv::{arch}"] = (f"{s['dice']['mean']:.4f} +/- {s['dice']['std']:.4f} | "
                            f"{s['hd95']['mean']:.2f} | {s['asd']['mean']:.3f}")

    bp = OUT / "backend_benchmark.json"
    if bp.exists():
        for name, label in (("pytorch_fp32", "PyTorch FP32"),
                            ("onnxruntime", "ONNXRuntime (GPU EP)"),
                            ("tensorrt_fp16", "TensorRT FP16")):
            e = json.loads(bp.read_text(encoding="utf-8"))["backends"].get(name)
            if not isinstance(e, dict):
                continue
            vram = f"{e['vram_mb']:.0f} MB" if e.get("vram_mb") else "-"
            # 표기를 테스트와 맞춤(소수 2자리, 단위 붙여쓰기). 형식이 다르면
            # "README 가 산출물을 인용했는가" 검사가 문자열을 못 찾음.
            v[f"bk::{label}"] = (f"{e['latency_ms']:.2f}ms | {e['throughput_ips']:.0f}/s | "
                                 f"{vram} | {e['speedup_vs_pytorch']:.2f}x")
    return v


ROW = re.compile(r"^\|\s*(?P<label>[^|]+?)\s*\|(?P<rest>.*)\|\s*$")


def sync(path: Path, values: dict[str, str], check: bool) -> list[str]:
    lines = path.read_text(encoding="utf-8").split("\n")
    changed: list[str] = []
    for i, line in enumerate(lines):
        m = ROW.match(line)
        if not m:
            continue
        raw = m.group("label")
        label = raw.replace("**", "").strip()
        key = label
        if key.startswith("SwinUNETR") or key in ("UNet", "SegResNet", "SwinUNETR"):
            key = f"cv::{label.split(' ')[0].lower()}"
        elif label in ("PyTorch FP32", "ONNXRuntime (GPU EP)", "TensorRT FP16"):
            key = f"bk::{label}"
        if key not in values:
            continue
        want = values[key]
        # 굵게 쓴 칸은 굵게 유지함.
        cells = [c.strip() for c in m.group("rest").split("|")]
        # 행 이름이 같아도 다른 표일 수 있음. README 에는 구조 이름을 행으로 쓰는
        # 표가 둘 있는데 하나는 Dice 표, 다른 하나는 fold별 에폭 표임. 칸의 생김새로
        # 가름: 안 그러면 에폭 표에 Dice 를 써넣음.
        if key.startswith("cv::") and " +/- " not in cells[0]:
            continue
        if key.startswith("bk::") and "ms" not in cells[0]:
            continue
        new_cells = want.split(" | ")
        if len(new_cells) != len(cells):
            continue
        out = []
        for old, new in zip(cells, new_cells, strict=True):
            if old.startswith("**") and old.endswith("**"):
                new = f"**{new}**"
            out.append(new)
        newline = f"| {raw} | " + " | ".join(out) + " |"
        if newline != line:
            changed.append(f"{path.name}:{i + 1}  {label}: {' | '.join(cells)} -> {want}")
            if not check:
                lines[i] = newline
    if changed and not check:
        path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return changed


def main() -> int:
    check = "--check" in sys.argv
    values = build_values()
    total: list[str] = []
    for p in DOCS:
        if not p.exists():
            print(f"[없음] {p}")
            continue
        total += sync(p, values, check)
    if not total:
        print("문서 표의 수치가 산출물과 같다. (산문, 인용문은 보지 않는다)")
        return 0
    for t in total:
        print(("  [다름] " if check else "  [고침] ") + t)
    print(f"{'다른' if check else '고친'} 칸 {len(total)}개")
    return 1 if check else 0


if __name__ == "__main__":
    raise SystemExit(main())
