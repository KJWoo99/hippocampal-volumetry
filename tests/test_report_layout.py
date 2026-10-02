"""리포트 PDF 가 내용을 종이 밖으로 흘리지 않는지 검증.

페이지 넘김이 없으면 QC 플래그가 여러 개 붙은 리포트에서 마지막 섹션이 종이 아래로 잘림.
잘리는 것이 "이 결과를 믿지 말라"는 QC 경고라 가장 나쁜 실패임.
"""
from __future__ import annotations

import numpy as np
import pytest

from hippo.metrics.volume_agreement import volume_agreement_report
from hippo.volumetry.report import build_report, save_pdf
from hippo.volumetry.volume import compute_volumes

pytest.importorskip("reportlab")


def _mask():
    m = np.zeros((32, 32, 32), np.int16)
    m[4:12, 4:12, 4:12] = 1
    m[16:24, 16:24, 16:24] = 2
    return m


def _record(monkeypatch):
    """drawString/showPage 를 가로채 그려진 y 좌표와 페이지 수를 모음."""
    from reportlab.pdfgen import canvas as rl

    drawn: list[tuple[int, float, str]] = []
    pages: list[int] = []
    orig_ds, orig_sp = rl.Canvas.drawString, rl.Canvas.showPage

    def ds(self, x, y, text, *a, **k):
        drawn.append((len(pages), y, text))
        return orig_ds(self, x, y, text, *a, **k)

    def sp(self, *a, **k):
        pages.append(1)
        return orig_sp(self, *a, **k)

    monkeypatch.setattr(rl.Canvas, "drawString", ds)
    monkeypatch.setattr(rl.Canvas, "showPage", sp)
    return drawn, pages


def _full_report():
    rng = np.random.default_rng(1)
    ref = rng.uniform(2000, 9000, 52)
    pred = ref * 0.97 + rng.normal(0, 120, 52)
    qc = {
        "passed": False,
        "flags": [
            "라벨 1: 연결성분 3개 (파편화: 노이즈 가능)",
            "라벨 2: 연결성분 2개 (파편화: 노이즈 가능)",
            "부피 과소 430 < 500 mm^3",
        ],
        "total_mm3": 430.0,
    }
    normative = {
        "icv_mm3": 1500000.0, "normalized": 0.0045, "z_score": -1.7, "percentile": 4.5,
        "reference": {"mean": 0.0052, "sd": 0.0004, "n": 100, "source": "예시값"},
    }
    return build_report(
        "T1", compute_volumes(_mask(), (1.0, 1.0, 1.0)),
        volume_agreement_report(pred, ref), normative, qc, "swinunetr",
    )


def test_full_report_never_draws_off_page(tmp_path, monkeypatch):
    drawn, pages = _record(monkeypatch)
    save_pdf(_full_report(), str(tmp_path / "full.pdf"))
    off = [(p, y, t) for p, y, t in drawn if y < 0]
    assert not off, f"종이 밖에 그려진 줄: {off}"
    assert len(pages) >= 2, "내용이 한 장을 넘치면 페이지를 나눠야 한다"


def test_qc_flags_survive_pagination(tmp_path, monkeypatch):
    """가장 나쁜 실패(QC 경고가 소리 없이 사라지는 것)를 막음."""
    drawn, _ = _record(monkeypatch)
    save_pdf(_full_report(), str(tmp_path / "full.pdf"))
    text = "\n".join(t for _, _, t in drawn)
    for fragment in ("파편화", "부피 과소"):
        assert fragment in text, f"QC 플래그 '{fragment}' 가 리포트에서 사라졌다"
    assert "연구용" in text, "disclaimer 는 항상 남아야 한다"


def test_minimal_report_still_fits_one_page(tmp_path, monkeypatch):
    """내용이 적으면 굳이 페이지를 늘리지 않음."""
    drawn, pages = _record(monkeypatch)
    save_pdf(build_report("T2", compute_volumes(_mask(), (1.0, 1.0, 1.0))), str(tmp_path / "m.pdf"))
    assert len(pages) == 1
    assert all(y >= 0 for _, y, _ in drawn)
