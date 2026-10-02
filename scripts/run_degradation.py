"""부가 분석: MRI 화질 저하와 복원이 해마 분할과 부피 일치도에 주는 영향.

사전등록 `docs/prereg_degradation.md`. 순서대로 한 번씩 돌림.

    python scripts/run_degradation.py tune-nlm          # 비국소 평균 세기를 개발셋 40건의 PSNR 로 고름
    python scripts/run_degradation.py tune-nlm --grid wide  # 보조(사전등록 이탈): 넓힌 격자, degradation_tune_wide.json
    python scripts/run_degradation.py bias-ratio        # 바이어스 장의 최대/최소 비 실측(사전등록 3절), degradation_bias_ratio.json
    python scripts/run_degradation.py unet --seed 1     # 복원 3D U-Net(시드 1, 2, 3)
    python scripts/run_degradation.py finetune --fold 0 # 비교군: 저하 영상으로 분할 모델 미세조정(fold 0~4)
    python scripts/run_degradation.py evaluate          # 홀드아웃 52건, 부트스트랩, 판정

평가는 기존 홀드아웃 평가(`hippo.evaluation.benchmark.evaluate_on_holdout`)와 같은 경로를 씀. 검증 변환의 z-점수 정규화
바로 앞에 저하와 복원을 끼우고, 같은 `Trainer._validate` 와 `_collect_volumes` 로 측정함. 저하가 없는 조건은 기존 결과와 같아야 하며,
다르면 멈춤.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs"
WORK = OUT / "degradation"
BOOT_N = 1000
SEED = 42
NLM_GRID = (0.6, 0.8, 1.0, 1.2)
# 사전등록 이탈(prereg 수정 이력): 위 격자에서 세 잡음 세기 모두 가장 약한 끝(0.6)이 골라져, 원래 값을 모두 포함하고
# 양쪽으로 넓힌 격자를 보조 분석으로 따로 돌림. 주 분석(위 격자)은 그대로이고 결과는 "nlmwide" 이름으로 따로 남김.
NLM_GRID_WIDE = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5)
NLM_GRIDS = {"prereg": ("", NLM_GRID), "wide": ("_wide", NLM_GRID_WIDE)}
TUNE_N = 40
# 학습률은 이 저장소의 분할 학습(configs 의 lr_patience 8, lr_factor 0.5)과 같은 규칙으로 줄임. 학습률을 고정하면
# 흔들리다 배우지 못하는 시드가 나올 수 있어, 이 단계를 돌리기 전에 넣음(prereg 수정 이력).
UNET = dict(lr=1e-3, max_epochs=300, patience=20, n_val=20, lr_factor=0.5, lr_patience=8, lr_min=1e-6)
FINETUNE = dict(lr=1e-4, epochs=100)


def _json_default(o):
    # numpy 숫자(float32 등)는 json 이 못 씀. 평가를 다 계산한 뒤 저장에서 멈추지 않게 바꿔 줌.
    if hasattr(o, "item") and getattr(o, "ndim", 1) == 0:
        return o.item()
    raise TypeError(f"json 으로 못 쓴다: {type(o).__name__}")


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default) + "\n").encode("utf-8"))


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _cfg():
    from hippo.config import load_config
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    cfg.data.root = str(ROOT / "data")
    return cfg


def _lists(cfg):
    from hippo.data.datamodule import get_datalist, holdout_split
    full = get_datalist(cfg, download=False)
    return holdout_split(full, cfg.data.holdout_frac, cfg.data.holdout_seed)


def _split_val(cfg, extra=None):
    """검증 변환을 z-점수 정규화 앞뒤로 나눠, 그 사이에 extra 를 끼운 Compose."""
    from monai.transforms import Compose, NormalizeIntensityd

    from hippo.data.transforms import get_transforms
    _, val_t = get_transforms(tuple(cfg.model.roi), cfg.data.crop_samples, getattr(cfg.data, "aug", "basic"))
    ts = list(val_t.transforms)
    i = next(k for k, t in enumerate(ts) if isinstance(t, NormalizeIntensityd))
    return Compose(ts[:i] + list(extra or []) + ts[i:]), Compose(ts[:i])


def _raw_volumes(cfg, datalist):
    """1mm 재표본화까지 한 원래 세기 영상(정규화 전)과 파일 이름."""
    _, pre = _split_val(cfg)
    out = []
    for d in datalist:
        r = pre(dict(d))
        out.append((Path(d["image"]).name, r["image"].numpy()[0].astype("float64")))
    return out


# ------------------------------------------------------------------ 저하 + 복원 변환
class Condition:
    """정규화 앞에 끼우는 변환. kind=None 이면 그대로 둠. 케이스마다 PSNR 을 psnr 에 남김."""

    def __init__(self, kind=None, level=None, method="none", nlm_a=None, unet=None, psnr=None):
        self.kind, self.level, self.method = kind, level, method
        self.nlm_a, self.unet, self.psnr = nlm_a, unet, psnr

    def __call__(self, data):
        import torch

        from hippo.degrade import case_seed, cubic_from_blocks, degrade, n4, nlm3d, psnr3d
        d = dict(data)
        img = d["image"]
        if self.kind is None:
            return d
        name = Path(str(img.meta.get("filename_or_obj", ""))).name
        clean = img.numpy()[0].astype("float64")
        dg = degrade(clean, self.kind, self.level, case_seed(name, self.kind, self.level))
        x = dg["image"]
        if self.method == "nlm":
            x = nlm3d(x, self.nlm_a)
        elif self.method == "n4":
            x = n4(x)
        elif self.method == "cubic":
            x = cubic_from_blocks(*dg["blocks"], clean.shape[2])
        elif self.method.startswith("unet"):
            x = _unet_apply(self.unet, x)
        elif self.method != "none":
            raise ValueError(self.method)
        if self.psnr is not None:
            self.psnr[name] = psnr3d(x, clean)
        new = img.clone()
        new[0] = torch.as_tensor(x, dtype=img.dtype)
        d["image"] = new
        return d


def _pad16(x):
    import numpy as np
    pads = [(0, (-n) % 16) for n in x.shape]
    return np.pad(x, pads, mode="edge"), x.shape


def _unet_model():
    from monai.networks.nets import BasicUNet
    return BasicUNet(spatial_dims=3, in_channels=1, out_channels=1)


def _unet_apply(model, x):
    import numpy as np
    import torch
    s = float(np.percentile(x, 99)) or 1.0
    xp, shape = _pad16(x / s)
    dev = next(model.parameters()).device
    with torch.no_grad():
        y = model(torch.from_numpy(xp[None, None]).float().to(dev))[0, 0].cpu().numpy()
    return y[:shape[0], :shape[1], :shape[2]].astype("float64") * s


# ------------------------------------------------------------------ bias-ratio
def cmd_bias_ratio(args) -> int:
    """바이어스 장의 최대/최소 비를 세기마다 케이스별로 실측함(사전등록 3절). 학습, 추론 없이 CPU 로 몇 분."""
    import numpy as np

    from hippo.degrade import LEVELS, case_seed, degrade
    cfg = _cfg()
    dev, holdout = _lists(cfg)
    out = {}
    for split, lst in (("holdout", holdout), ("dev", dev)):
        vols = _raw_volumes(cfg, lst)
        out[split] = {}
        for c in LEVELS["bias"]:
            r = np.array([degrade(v, "bias", c, case_seed(n, "bias", c))["field_ratio"] for n, v in vols])
            out[split][str(c)] = {"n": len(r), "median": float(np.median(r)),
                                  "q25": float(np.percentile(r, 25)), "q75": float(np.percentile(r, 75))}
        print(f"  {split} 끝", flush=True)
    _save_json(OUT / "degradation_bias_ratio.json", out)
    print(json.dumps(out, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------ tune-nlm
def cmd_tune_nlm(args) -> int:
    import numpy as np

    from hippo.degrade import LEVELS, case_seed, degrade, nlm3d, psnr3d
    cfg = _cfg()
    dev, _ = _lists(cfg)
    pick = np.random.default_rng(SEED).choice(len(dev), size=TUNE_N, replace=False)
    vols = _raw_volumes(cfg, [dev[i] for i in sorted(pick)])
    suffix, grid = NLM_GRIDS[args.grid]
    out = {"n": len(vols), "grid_name": args.grid, "grid": grid, "levels": {}}
    for f in LEVELS["noise"]:
        table = []
        for name, v in vols:
            x = degrade(v, "noise", f, case_seed(name, "noise", f))["image"]
            table.append([psnr3d(x, v)] + [psnr3d(nlm3d(x, a), v) for a in grid])
        m = np.mean(table, axis=0)
        print(f"  f={f} 끝", flush=True)  # CPU 만 쓰는 단계라 오래 조용하면 멈춘 것과 구분되지 않아 진행을 찍음
        best = int(np.argmax(m[1:]))
        out["levels"][str(f)] = {"noisy_psnr": float(m[0]), "psnr_by_a": [float(z) for z in m[1:]],
                                 "a": grid[best], "at_grid_edge": best in (0, len(grid) - 1)}
    _save_json(OUT / f"degradation_tune{suffix}.json", out)
    print(json.dumps(out, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------ unet
def cmd_unet(args) -> int:
    import numpy as np
    import torch

    from hippo.degrade import KINDS, LEVELS, case_seed, degrade
    cfg = _cfg()
    dev, _ = _lists(cfg)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vols = _raw_volumes(cfg, dev)
    # 검증 20건을 피험자 단위로 고름. 같은 사람의 좌우가 학습과 검증에 갈리지 않게 함
    # (docs/prereg_pair_split.md 두 번째 추가 6번).
    val_idx = set(_subject_pick(dev, UNET["n_val"], SEED))
    val = [vols[i] for i in range(len(vols)) if i in val_idx]
    tr = [vols[i] for i in range(len(vols)) if i not in val_idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _unet_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=UNET["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=UNET["lr_factor"],
                                                       patience=UNET["lr_patience"], min_lr=UNET["lr_min"])

    def pair(name, v, kind, level, seed):
        x = degrade(v, kind, level, seed)["image"]
        s = float(np.percentile(x, 99)) or 1.0
        xp, shape = _pad16(x / s)
        yp, _ = _pad16(v / s)
        return (torch.from_numpy(xp[None, None]).float().to(device),
                torch.from_numpy(yp[None, None]).float().to(device), shape)

    def l1(pred, y, shape):
        return (pred - y)[..., :shape[0], :shape[1], :shape[2]].abs().mean()

    val_set = [pair(n, v, k, lv, case_seed(n, k, lv)) for n, v in val for k in KINDS for lv in LEVELS[k]]
    # 복원 없이 저하 영상 그대로의 오차. 학습이 이것보다 낮아야 뭔가 배운 것임.
    val_identity = float(np.mean([l1(x, y, s).item() for x, y, s in val_set]))
    print(f"  저하 그대로의 val L1 {val_identity:.5f}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    wpath = WORK / f"unet_seed{args.seed}.pth"
    hist, best, bad, t0 = [], float("inf"), 0, time.time()
    for ep in range(1, UNET["max_epochs"] + 1):
        model.train()
        run = 0.0
        for i in rng.permutation(len(tr)):
            k = KINDS[rng.integers(len(KINDS))]
            lv = LEVELS[k][rng.integers(3)]
            x, y, shape = pair(*tr[i], k, lv, int(rng.integers(2 ** 62)))
            loss = l1(model(x), y, shape)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([l1(model(x), y, s).item() for x, y, s in val_set]))
        lr_now = opt.param_groups[0]["lr"]
        hist.append({"epoch": ep, "train_l1": run / len(tr), "val_l1": vl, "lr": lr_now})
        if ep % 10 == 0:
            print(f"  epoch {ep} train {run / len(tr):.5f} val {vl:.5f} lr {lr_now:.1e}  {time.time() - t0:.0f}s",
                  flush=True)
        sched.step(vl)
        if vl < best:
            best, bad = vl, 0
            torch.save(model.state_dict(), wpath)
        else:
            bad += 1
            if bad >= UNET["patience"]:
                break
    best_ep = min(hist, key=lambda h: h["val_l1"])["epoch"]
    _save_json(OUT / f"degradation_unet_seed{args.seed}.json",
               {"seed": args.seed, "config": UNET, "n_train": len(tr), "n_val": len(val), "best_epoch": best_ep,
                "epochs_run": len(hist), "stopped_early": len(hist) < UNET["max_epochs"], "best_val_l1": best,
                "val_identity_l1": val_identity, "val_l1_ratio_to_identity": best / val_identity,
                "seconds": round(time.time() - t0), "history": hist})
    return 0


# ------------------------------------------------------------------ finetune
class RandDegrade:
    """미세조정용: 케이스마다 저하 없음과 저하 셋 x 세기 셋 가운데 하나를 무작위로 적용함."""

    def __init__(self, seed):
        import numpy as np
        self.rng = np.random.default_rng(seed)

    def __call__(self, data):
        import torch

        from hippo.degrade import KINDS, LEVELS, degrade
        choice = self.rng.integers(len(KINDS) + 1)
        if choice == len(KINDS):
            return data
        k = KINDS[choice]
        lv = LEVELS[k][self.rng.integers(3)]
        d = dict(data)
        img = d["image"]
        x = degrade(img.numpy()[0].astype("float64"), k, lv, int(self.rng.integers(2 ** 62)))["image"]
        new = img.clone()
        new[0] = torch.as_tensor(x, dtype=img.dtype)
        d["image"] = new
        return d


class FixedDegrade:
    """미세조정 검증용: 케이스 이름으로 저하 종류와 세기를 고정함."""

    def __call__(self, data):
        from hippo.degrade import KINDS, LEVELS, case_seed
        name = Path(str(data["image"].meta.get("filename_or_obj", ""))).name
        s = case_seed(name, "finetune-val", 0)
        k = KINDS[s % 3]
        return Condition(k, LEVELS[k][(s // 3) % 3])(data)


def cmd_finetune(args) -> int:
    import torch
    from monai.data import CacheDataset, DataLoader, Dataset
    from monai.transforms import Compose, NormalizeIntensityd

    from hippo.data.datamodule import kfold_datalist
    from hippo.data.transforms import get_transforms
    from hippo.models import build_model
    from hippo.training.trainer import Trainer
    cfg = _cfg()
    dev, _ = _lists(cfg)
    tr_list, va_list = kfold_datalist(dev, cfg.data.n_splits, args.fold, cfg.data.fold_seed)
    fcfg = copy.deepcopy(cfg)
    fcfg.model.name = _arch()
    fcfg.train.lr = FINETUNE["lr"]
    fcfg.train.epochs = FINETUNE["epochs"]
    fcfg.out_dir = str(WORK / "finetune" / f"fold{args.fold}")
    train_t, _ = get_transforms(tuple(cfg.model.roi), cfg.data.crop_samples, getattr(cfg.data, "aug", "basic"))
    ts = list(train_t.transforms)
    i = next(k for k, t in enumerate(ts) if isinstance(t, NormalizeIntensityd))
    pre, post = Compose(ts[:i]), Compose([RandDegrade(1000 + args.fold), *ts[i:]])
    cached = CacheDataset(data=tr_list, transform=pre, cache_rate=1.0, num_workers=4)
    train_ds = Dataset(data=cached, transform=post)
    val_t, _ = _split_val(cfg, [FixedDegrade()])
    val_ds = CacheDataset(data=va_list, transform=val_t, cache_rate=1.0, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(_arch(), cfg.model.in_channels, cfg.model.out_channels, tuple(cfg.model.roi))
    src = OUT / "benchmark" / _arch() / f"fold{args.fold}" / "best_model.pth"
    model.load_state_dict(torch.load(src, map_location=device, weights_only=True))
    trainer = Trainer(fcfg, model, device)
    t0 = time.time()
    before = trainer._validate(DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0))
    res = trainer.fit(DataLoader(train_ds, batch_size=cfg.train.batch, shuffle=True, num_workers=0),
                      DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0))
    _save_json(OUT / f"degradation_finetune_fold{args.fold}.json",
               {"fold": args.fold, "config": FINETUNE, "val_dice_degraded_before": before["dice"],
                "best_epoch": res.best_epoch, "val_dice_degraded_after": res.best_metric,
                "seconds": round(time.time() - t0)})
    return 0


# ------------------------------------------------------------------ evaluate
def _eval_condition(cfg, holdout, ckpts, cond, device):
    import numpy as np
    import torch
    from monai.data import CacheDataset, DataLoader

    from hippo.evaluation.benchmark import _collect_volumes
    from hippo.models import build_model
    from hippo.training.trainer import Trainer
    t, _ = _split_val(cfg, [cond] if cond.kind is not None else [])
    ds = CacheDataset(data=holdout, transform=t, cache_rate=1.0, num_workers=0, progress=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    ecfg = copy.deepcopy(cfg)
    ecfg.out_dir = str(WORK / "_eval")
    per, preds, ref = [], [], None
    for ck in ckpts:
        model = build_model(_arch(), cfg.model.in_channels, cfg.model.out_channels, tuple(cfg.model.roi))
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        model.to(device)
        sc = Trainer(ecfg, model, device)._validate(loader)
        pv, rv = _collect_volumes(model, loader, ecfg, device)
        if ref is None:
            ref = np.array(rv)
        per.append(sc)
        preds.append(pv)
    return {"dice": float(np.mean([p["dice"] for p in per])), "hd95": float(np.mean([p["hd95"] for p in per])),
            "asd": float(np.mean([p["asd"] for p in per])), "ens": np.mean(preds, axis=0), "ref": ref}


def _subject_units(items: list) -> list[list[int]]:
    """같은 피험자의 인덱스를 한 단위로 묶음(`hippo.data.datamodule.subject_key`). 순서는 처음 나온 순서임."""
    from hippo.data.datamodule import subject_key
    units: dict[int, list[int]] = {}
    for i, item in enumerate(items):
        units.setdefault(subject_key(item), []).append(i)
    return list(units.values())


def _subject_pick(items: list, n: int, seed: int) -> list[int]:
    """피험자 단위로 섞어 정확히 n 건을 고름. 더하면 n 을 넘는 단위는 건너뜀(홀드아웃과 같은 방식)."""
    import numpy as np
    units = _subject_units(items)
    picked: list[int] = []
    for u in np.random.default_rng(seed).permutation(len(units)):
        if len(picked) + len(units[u]) <= n:
            picked.extend(units[u])
        if len(picked) == n:
            return picked
    raise ValueError(f"{n}건을 피험자 단위로 채우지 못했다({len(picked)}건)")


def _subject_resample(units: list[list[int]], rng):
    """피험자 단위 부트스트랩 한 번. 뽑힌 피험자의 케이스를 모두 넣음."""
    import numpy as np
    pick = rng.integers(0, len(units), len(units))
    return np.array([i for u in pick for i in units[u]])


def _arch() -> str:
    """벤치마크가 고른 구조. 부가 분석은 배포와 같은 구조를 써야 함.

    구조 이름을 코드에 적어 두지 않고 벤치마크 결과에서 읽음(docs/prereg_pair_split.md).
    """
    return _load_json(OUT / "benchmark_results.json")["selected_arch"]


def cmd_evaluate(args) -> int:
    import numpy as np
    import torch

    from hippo.degrade import KINDS, LEVELS
    from hippo.metrics.volume_agreement import icc_2_1, volume_agreement_report
    cfg = _cfg()
    _, holdout = _lists(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = [str(OUT / "benchmark" / _arch() / f"fold{k}" / "best_model.pth") for k in range(5)]
    tuned = [str(WORK / "finetune" / f"fold{k}" / "best_model.pth") for k in range(5)]
    nlm_a = {float(k): v["a"] for k, v in _load_json(OUT / "degradation_tune.json")["levels"].items()}
    # 넓힌 격자(사전등록 이탈, 보조 분석): 조정 결과가 있을 때만 잡음 조건에 "nlmwide" 를 더함.
    wide_path = OUT / "degradation_tune_wide.json"
    nlm_wide = ({float(k): v["a"] for k, v in _load_json(wide_path)["levels"].items()} if wide_path.exists() else {})
    unets = {}
    for s in (1, 2, 3):
        m = _unet_model().to(device)
        m.load_state_dict(torch.load(WORK / f"unet_seed{s}.pth", map_location=device, weights_only=True))
        unets[s] = m.eval()

    classic = {"noise": "nlm", "bias": "n4", "thick": "cubic"}
    conds = {"clean": (Condition(), base)}
    for k in KINDS:
        for lv in LEVELS[k]:
            tag = f"{k}{lv}"
            conds[f"{tag}_none"] = (Condition(k, lv), base)
            conds[f"{tag}_{classic[k]}"] = (Condition(k, lv, classic[k], nlm_a=nlm_a.get(lv)), base)
            if k == "noise" and lv in nlm_wide:
                conds[f"{tag}_nlmwide"] = (Condition(k, lv, "nlm", nlm_a=nlm_wide[lv]), base)
            for s in (1, 2, 3):
                conds[f"{tag}_unet{s}"] = (Condition(k, lv, f"unet{s}", unet=unets[s]), base)
            conds[f"{tag}_finetuned"] = (Condition(k, lv), tuned)
    conds["clean_finetuned"] = (Condition(), tuned)

    res, psnr = {}, {}
    t0 = time.time()
    for name, (cond, ckpts) in conds.items():
        cond.psnr = psnr.setdefault(name, {})
        res[name] = _eval_condition(cfg, holdout, ckpts, cond, device)
        print(f"  {name}: dice {res[name]['dice']:.4f}  {time.time() - t0:.0f}s", flush=True)
    ref = res["clean"]["ref"]

    # 평가 경로 확인: 저하 없음은 기존 홀드아웃 결과와 같아야 함.
    bench = _load_json(OUT / "benchmark_results.json")["holdout_result"]
    clean_icc = icc_2_1(res["clean"]["ens"], ref)
    check = {"dice": res["clean"]["dice"], "dice_benchmark": bench["holdout_summary"]["dice"]["mean"],
             "icc": clean_icc, "icc_benchmark": bench["ensemble_agreement"]["icc_2_1"]}
    check["same"] = bool(abs(check["dice"] - check["dice_benchmark"]) < 1e-4 and abs(clean_icc - check["icc_benchmark"]) < 1e-4)
    if not check["same"]:
        _save_json(OUT / "degradation_results.json", {"path_check": check, "error": "저하 없음 결과가 기존 홀드아웃과 다르다"})
        print("평가 경로가 기존과 다르다. 멈춘다:", check)
        return 1

    def summary(name):
        r = volume_agreement_report(res[name]["ens"], ref)
        return {"dice": res[name]["dice"], "hd95": res[name]["hd95"], "asd": res[name]["asd"],
                "icc": r["icc_2_1"], "loa_pct": r["loa_halfwidth_pct_of_reference"],
                "bias": r["bland_altman"]["bias"],
                "psnr": float(np.mean(list(psnr[name].values()))) if psnr.get(name) else None}

    table = {n: summary(n) for n in conds}
    rng = np.random.default_rng(SEED)
    boot = {n: [] for n in conds}
    # 부트스트랩은 피험자 단위로 뽑음. 짝 단위 분할에서는 한 사람의 좌우 해마가 함께 홀드아웃에 들어가
    # (52건 중 16쌍) 케이스끼리 독립이 아님. 케이스 단위로 뽑으면 구간이 실제보다 좁아짐
    # (결과를 보기 전 docs/prereg_pair_split.md 에 추가).
    unit_list = _subject_units(holdout)
    for _ in range(BOOT_N):
        ix = _subject_resample(unit_list, rng)
        for n in conds:
            boot[n].append(icc_2_1(res[n]["ens"][ix], ref[ix]))
    boot = {n: np.array(v) for n, v in boot.items()}
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731

    levels = {}
    for k in KINDS:
        for lv in LEVELS[k]:
            tag = f"{k}{lv}"
            drop = boot["clean"] - boot[f"{tag}_none"]
            drop_pt = table["clean"]["icc"] - table[f"{tag}_none"]["icc"]
            gate = bool(np.percentile(drop, 2.5) > 0)
            un_pt = float(np.mean([table[f"{tag}_unet{s}"]["icc"] for s in (1, 2, 3)]))
            un_bs = np.mean([boot[f"{tag}_unet{s}"] for s in (1, 2, 3)], axis=0)
            meth = {"unet": (un_pt, un_bs), classic[k]: (table[f"{tag}_{classic[k]}"]["icc"], boot[f"{tag}_{classic[k]}"])}
            if f"{tag}_nlmwide" in table:
                meth["nlmwide"] = (table[f"{tag}_nlmwide"]["icc"], boot[f"{tag}_nlmwide"])
            lv_out = {"icc_degraded": table[f"{tag}_none"]["icc"], "drop": drop_pt, "drop_ci": ci(drop),
                      "drop_confirmed": gate, "icc_finetuned": table[f"{tag}_finetuned"]["icc"], "methods": {}}
            for m, (pt, bs) in meth.items():
                rec_bs = (bs - boot[f"{tag}_none"]) / np.where(drop == 0, np.nan, drop)
                lv_out["methods"][m] = {
                    "icc": pt, "recovery": (pt - table[f"{tag}_none"]["icc"]) / drop_pt if gate and drop_pt else None,
                    "recovery_ci": ci(rec_bs[~np.isnan(rec_bs)]) if gate else None,
                    "vs_finetuned": pt - table[f"{tag}_finetuned"]["icc"],
                    "vs_finetuned_ci": ci(bs - boot[f"{tag}_finetuned"])}
            levels[tag] = lv_out
    prim = levels["noise0.06"]
    pm = prim["methods"]["unet"]
    if not prim["drop_confirmed"]:
        verdict = "잡음 f=0.06 에서는 저하가 부피 일치도를 해치지 않았다(하락의 95% 신뢰구간이 0 을 포함)"
    else:
        lo, hi = pm["vs_finetuned_ci"]
        cmp_ = "복원이 낫다" if lo > 0 else ("미세조정이 낫다" if hi < 0 else "복원과 미세조정의 차이를 확인하지 못했다")
        verdict = f"회복률 {pm['recovery']:.3f} [{pm['recovery_ci'][0]:.3f}, {pm['recovery_ci'][1]:.3f}], {cmp_}"
    flips = []
    for k in KINDS:
        for lv in LEVELS[k]:
            tag = f"{k}{lv}"
            dg = table[f"{tag}_none"]
            for m in [classic[k], "unet1", "unet2", "unet3", "nlmwide"]:
                if f"{tag}_{m}" not in table:
                    continue
                r = table[f"{tag}_{m}"]
                if r["psnr"] is not None and r["psnr"] > dg["psnr"] and abs(r["bias"]) > abs(dg["bias"]):
                    flips.append({"condition": tag, "method": m})
    _save_json(OUT / "degradation_results.json",
               {"path_check": check, "n_holdout": len(ref), "boot": BOOT_N, "table": table, "levels": levels,
                "primary": {"condition": "noise0.06", "method": "unet(시드 3 평균)", "verdict": verdict},
                "psnr_up_bias_up": {"count": len(flips), "cases": flips},
                "seconds": round(time.time() - t0)})
    print(verdict)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tune-nlm").add_argument("--grid", choices=list(NLM_GRIDS), default="prereg")
    sub.add_parser("bias-ratio")
    sub.add_parser("unet").add_argument("--seed", type=int, choices=[1, 2, 3], required=True)
    sub.add_parser("finetune").add_argument("--fold", type=int, choices=[0, 1, 2, 3, 4], required=True)
    sub.add_parser("evaluate")
    args = ap.parse_args()
    return {"tune-nlm": cmd_tune_nlm, "bias-ratio": cmd_bias_ratio, "unet": cmd_unet, "finetune": cmd_finetune,
            "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
