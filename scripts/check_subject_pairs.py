"""피험자 짝 규칙을 원자료로 확인함(분할의 근거, docs/prereg_pair_split.md).

MSD Task04 는 피험자와 파일의 대응을 공개하지 않음. 이 저장소는 번호 (31,32), (33,34) 처럼 홀수와 그다음 짝수가 한 사람의
좌, 우 해마라고 보고 분할에서 짝을 가르지 않음(`hippo.data.datamodule.subject_key`). 그 근거를 두 가지로 남김.

1. 구조: 학습용과 테스트용을 합친 파일 번호가 짝 키마다 정확히 두 개씩 묶이는가(피험자 195명이면 키 195개).
2. 닮음: 라벨 있는 학습용에서 이 규칙의 짝이, 한 칸 민 규칙(30,31)과 무작위 짝(2,000번)보다 해마 부피, 영상 크기, 밝기
   분포가 닮았는가. 집계에 묻히지 않게 가장 덜 닮은 짝도 적음.

    python scripts/check_subject_pairs.py        # 결과: outputs/subject_pairs.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Task04_Hippocampus"
OUT = ROOT / "outputs" / "subject_pairs.json"
N_RANDOM = 2000
SEED = 2029


def _num(name: str) -> int:
    return int(re.search(r"(\d+)", name).group(1))


def load_case(img_path: Path) -> dict:
    img = nib.load(str(img_path))
    lab = nib.load(str(DATA / "labelsTr" / img_path.name))
    a = np.asarray(img.dataobj, dtype=np.float64)
    y = np.asarray(lab.dataobj)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    q = np.percentile(a[a > 0] if (a > 0).any() else a, [5, 25, 50, 75, 95, 99])
    return {"id": _num(img_path.name), "shape": list(a.shape), "zooms": list(zooms),
            "vol_mm3": float((y > 0).sum() * np.prod(zooms)), "q": [float(v) for v in q]}


def pair_metrics(c1: dict, c2: dict) -> dict:
    """작을수록 닮음. vol 부피 로그비, shape 영상 크기 차이, inten 99 분위로 나눈 밝기 분위수 차이, raw 밝기 배율 로그비."""
    q1, q2 = np.array(c1["q"]), np.array(c2["q"])
    return {"vol": abs(float(np.log(c1["vol_mm3"] / c2["vol_mm3"]))),
            "shape": float(np.abs(np.array(c1["shape"]) - np.array(c2["shape"])).sum()),
            "inten": float(np.abs(q1[:5] / q1[5] - q2[:5] / q2[5]).sum()),
            "raw": abs(float(np.log(q1[5] / q2[5])))}


def summarize(pairs: list[tuple[dict, dict]]) -> dict:
    m = [pair_metrics(a, b) for a, b in pairs]
    out = {k: float(np.mean([x[k] for x in m])) for k in m[0]}
    out["vol_r"] = float(np.corrcoef([a["vol_mm3"] for a, _ in pairs], [b["vol_mm3"] for _, b in pairs])[0, 1])
    out["n_pairs"] = len(pairs)
    return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    all_ids = sorted(_num(p.name) for d in ("imagesTr", "imagesTs") for p in (DATA / d).glob("hippocampus_*.nii.gz"))
    keys: dict[int, list[int]] = {}
    for i in all_ids:
        keys.setdefault((i + 1) // 2, []).append(i)
    structure = {"n_files": len(all_ids), "n_keys": len(keys),
                 "files_per_key": sorted({len(v) for v in keys.values()}),
                 "missing_numbers": sorted(set(range(1, max(all_ids) + 1)) - set(all_ids))}

    cases = {c["id"]: c for c in (load_case(p) for p in sorted((DATA / "imagesTr").glob("hippocampus_*.nii.gz")))}
    rules = {}
    for name, shift in (("pair_31_32", 1), ("shifted_30_31", 0)):
        g: dict[int, list[int]] = {}
        for i in cases:
            g.setdefault((i + shift) // 2, []).append(i)
        pairs = [(cases[v[0]], cases[v[1]]) for v in g.values() if len(v) == 2]
        rules[name] = summarize(pairs)
        if shift == 1:
            each = sorted(({"ids": [a["id"], b["id"]], **pair_metrics(a, b)} for a, b in pairs),
                          key=lambda x: -x["vol"])
            rules[name]["least_similar_by_volume"] = each[:5]
            rules[name]["least_similar_by_intensity"] = sorted(each, key=lambda x: -x["inten"])[:5]

    rng = np.random.default_rng(SEED)
    ids = np.array(sorted(cases))
    n_pairs = rules["pair_31_32"]["n_pairs"]
    rand = [summarize([(cases[a], cases[b]) for a, b in rng.permutation(ids)[: 2 * n_pairs].reshape(-1, 2)])
            for _ in range(N_RANDOM)]
    keys_m = ["vol", "shape", "inten", "raw", "vol_r"]
    random = {k: {"mean": float(np.mean([r[k] for r in rand])),
                  "p2_5": float(np.percentile([r[k] for r in rand], 2.5)),
                  "p97_5": float(np.percentile([r[k] for r in rand], 97.5))} for k in keys_m}
    for v in rules.values():
        v["frac_random_as_similar"] = {k: float(np.mean([(r[k] >= v[k]) if k == "vol_r" else (r[k] <= v[k]) for r in rand]))
                                       for k in keys_m}
    out = {"structure": structure, "n_labeled": len(cases), "n_random": N_RANDOM, "seed": SEED,
           "rules": rules, "random": random}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[저장] {OUT}")
    print(f"  구조: 파일 {structure['n_files']}, 짝 키 {structure['n_keys']}, 키마다 {structure['files_per_key']}, "
          f"결번 {structure['missing_numbers']}")
    for name, v in rules.items():
        print(f"  {name}: 부피 상관 {v['vol_r']:.3f}, 부피 로그비 {v['vol']:.3f} (무작위 {random['vol']['mean']:.3f})")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
