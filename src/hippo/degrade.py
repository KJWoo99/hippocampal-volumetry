"""MRI 화질 저하 모사와 복원(사전등록 `docs/prereg_degradation.md` 3, 4절).

모두 1mm 재표본화 뒤, z-점수 정규화 전의 원래 세기(3D numpy, 축 순서 R, A, S)에서 함. 분할 모델이 보는 것은
그 뒤의 정규화 결과이므로, 저하와 복원을 정규화 앞에 끼우면 배포 경로를 바꾸지 않고 입력만 바뀜.

저하는 케이스, 종류, 세기에서 만든 고정 시드로 결정적임. 같은 케이스는 어느 장비에서든 같은 저하 영상이 됨.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np

__all__ = [
    "KINDS",
    "LEVELS",
    "bias_field",
    "case_seed",
    "cubic_from_blocks",
    "degrade",
    "n4",
    "nlm3d",
    "psnr3d",
    "rician",
    "thick_blocks",
]

KINDS = ("noise", "bias", "thick")
# 사전등록 3절. 가운데가 주 조건.
# 바이어스 계수는 장의 최대/최소 비가 중앙값 약 1.2, 1.45, 2.1 배가 되도록 잡음(35x50x35 격자, 시드 200개 실측).
# 0.2~0.6 이면 19항이 더해져 비가 4~87배라 코일 불균일 범위를 한참 넘음(사전등록 수정 이력).
LEVELS = {"noise": (0.03, 0.06, 0.10), "bias": (0.025, 0.05, 0.1), "thick": (2, 3, 4)}
S_AXIS = 2  # RAS 방향 통일 뒤 위아래(S) 축


def case_seed(name: str, kind: str, level: float) -> int:
    h = hashlib.sha256(f"{name}|{kind}|{level}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def rician(x: np.ndarray, f: float, rng: np.random.Generator) -> np.ndarray:
    """위상 0 을 가정한 Rician 잡음. 실수부, 허수부에 같은 표준편차의 가우시안을 넣고 크기를 취함."""
    x = np.asarray(x, dtype=np.float64)
    fg = x[x > 0]
    s = f * float(np.median(fg)) if fg.size else 0.0
    re = x + s * rng.standard_normal(x.shape)
    im = s * rng.standard_normal(x.shape)
    return np.sqrt(re * re + im * im)


def bias_field(shape: tuple[int, int, int], c: float, rng: np.random.Generator) -> np.ndarray:
    """exp(P). P 는 좌표를 -1~1 로 둔 3차 이하 다항식(상수항 제외 19항), 계수는 -c~c 균등."""
    axes = [np.linspace(-1.0, 1.0, n) for n in shape]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    P = np.zeros(shape)
    for i, j, k in itertools.product(range(4), repeat=3):
        if 0 < i + j + k <= 3:
            P += rng.uniform(-c, c) * X ** i * Y ** j * Z ** k
    return np.exp(P)


def thick_blocks(x: np.ndarray, t: int, axis: int = S_AXIS) -> tuple[np.ndarray, np.ndarray]:
    """S 축으로 t 장씩 평균(부분 부피). (블록 평균, 블록 가운데 위치) 를 돌려줌. 마지막 블록은 남은 장만 평균함."""
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    n = x.shape[-1]
    starts = np.arange(0, n, t)
    means = np.stack([x[..., s:min(s + t, n)].mean(axis=-1) for s in starts], axis=-1)
    centers = np.array([(s + min(s + t, n) - 1) / 2.0 for s in starts])
    return means, centers


def _resample(means: np.ndarray, centers: np.ndarray, n: int, kind: str, axis: int) -> np.ndarray:
    from scipy.interpolate import interp1d

    pos = np.clip(np.arange(n, dtype=np.float64), centers[0], centers[-1])  # 바깥은 가장자리 블록 값
    if kind == "cubic" and len(centers) >= 4:
        f = interp1d(centers, means, kind="cubic", axis=-1, assume_sorted=True)
    else:
        f = interp1d(centers, means, kind="linear", axis=-1, assume_sorted=True)
    return np.moveaxis(f(pos), -1, axis)


def cubic_from_blocks(means: np.ndarray, centers: np.ndarray, n: int, axis: int = S_AXIS) -> np.ndarray:
    return _resample(means, centers, n, "cubic", axis)


def degrade(x: np.ndarray, kind: str, level: float, seed: int) -> dict:
    """저하 하나. {"image": 1mm 격자의 저하 영상, 그 밖에 복원에 필요한 것} 을 돌려줌.

    두꺼운 절편은 블록 평균을 1mm 로 선형 보간한 것이 저하 영상이고(촬영 뒤 재표본화), 삼차 복원은 같은 블록 평균에서 다시 보간함.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    if kind == "noise":
        return {"image": rician(x, float(level), rng)}
    if kind == "bias":
        field = bias_field(x.shape, float(level), rng)
        return {"image": x * field, "field_ratio": float(field.max() / field.min())}
    if kind == "thick":
        means, centers = thick_blocks(x, int(level))
        return {"image": _resample(means, centers, x.shape[S_AXIS], "linear", S_AXIS),
                "blocks": (means, centers)}
    raise ValueError(f"모르는 저하: {kind}")


def nlm3d(x: np.ndarray, a: float) -> np.ndarray:
    """3D 비국소 평균. h = a x 추정 잡음(웨이블릿 MAD)."""
    from skimage.restoration import denoise_nl_means, estimate_sigma

    x = np.asarray(x, dtype=np.float64)
    s = float(estimate_sigma(x))
    if s <= 0:
        return x.copy()
    return denoise_nl_means(x, h=a * s, sigma=s, fast_mode=True, patch_size=3, patch_distance=5)


def n4(x: np.ndarray) -> np.ndarray:
    """N4 바이어스 보정(SimpleITK 기본 설정, Otsu 마스크)."""
    import SimpleITK as sitk

    x = np.asarray(x, dtype=np.float32)
    img = sitk.GetImageFromArray(x)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    out = sitk.N4BiasFieldCorrectionImageFilter().Execute(img, mask)
    return sitk.GetArrayFromImage(out).astype(np.float64)


def _znorm(x: np.ndarray) -> np.ndarray:
    """분할 모델 입력과 같은 정규화(NormalizeIntensityd nonzero=True): 0 이 아닌 복셀만 평균 0, 표준편차 1."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    nz = x != 0
    if nz.any():
        v = x[nz]
        out[nz] = (v - v.mean()) / (v.std() or 1.0)
    return out


def psnr3d(x: np.ndarray, clean: np.ndarray) -> float:
    """분할 모델이 보는 것과 같게 둘 다 z-점수 정규화한 뒤의 PSNR(dB). 봉우리는 깨끗한 영상의 값 범위.

    원래 세기로 재면 N4 처럼 영상 전체의 밝기 배율을 바꾸는 복원이 배율 차이만으로 크게 깎임. 모델은 정규화 뒤의
    영상을 보므로 그 차이는 분할에 영향이 없음.
    """
    zc, zx = _znorm(clean), _znorm(x)
    peak = float(zc.max() - zc.min()) or 1.0
    mse = float(np.mean((zx - zc) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak * peak / mse)
