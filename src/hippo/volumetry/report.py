"""임상 volumetry 리포트 생성: JSON(기계용) + PDF(사람용).

실제 임상 제품처럼 구조화된 산출물을 만듦. 분할 마스크에서 끝내지 않고
부피 수치와 해석까지 담은 리포트를 최종 산출물로 둠.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from hippo.volumetry.volume import VolumeResult


def build_report(
    case_id: str,
    volume: VolumeResult,
    agreement: dict | None = None,
    normative: dict | None = None,
    qc: dict | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """단일 케이스의 구조화 리포트 dict를 만듦."""
    return {
        "case_id": case_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "volumetry": volume.to_dict(),
        "volume_agreement": agreement,   # 코호트 검증 결과(있으면)
        "normative": normative,          # ICV정규화/percentile(전뇌 데이터 시)
        "qc": qc,                        # 출력 QC 결과
        "disclaimer": (
            "연구용. 정상군 파라미터는 예시값이며 임상 판정에 사용 불가."
        ),
    }


def save_json(report: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def save_pdf(report: dict, path: str) -> str:
    """reportlab으로 한 장짜리 임상 스타일 리포트 PDF를 만듦."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    # 표준 PDF 폰트(Helvetica)에는 한글 글리프가 없다 -> 한글 줄은 내장 한글 CID 폰트로.
    kfont = "HYSMyeongJo-Medium"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(kfont))
    except Exception:  # noqa: BLE001
        kfont = "Helvetica"  # 폰트팩 없으면 폴백(한글은 안 보이나 깨지지 않음)

    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    y = h - 25 * mm

    # 페이지 넘김. 이게 없으면 y 가 그냥 음수로 내려가 내용이 종이 밖에 그려짐.
    # QC 플래그가 여러 개 붙으면 마지막 섹션이 잘리는데, 잘리는 것이 "이 결과를 믿지 말라"는
    # QC 경고라 가장 나쁜 실패임. reportlab 은 showPage() 후 폰트 상태가 초기화되므로 다시 지정함.
    top_y = h - 20 * mm
    bottom_y = 22 * mm      # 하단 disclaimer(15mm) 위로 남기는 여백

    def _space(needed: float, font: str, size: float) -> None:
        nonlocal y
        if y - needed < bottom_y:
            c.showPage()
            y = top_y
        c.setFont(font, size)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, y, "Hippocampal Volumetry Report")
    y -= 8 * mm
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawString(20 * mm, y, f"case: {report.get('case_id')}   model: {report.get('model')}")
    if report.get("generated_at"):
        y -= 5 * mm
        c.drawString(20 * mm, y, f"generated: {report['generated_at']}")
    c.setFillColor(colors.black)
    y -= 10 * mm

    vol = report["volumetry"]
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Volumes")
    y -= 6 * mm
    c.setFont("Helvetica", 10)
    c.drawString(
        25 * mm, y,
        f"space: {vol['space']}   voxel: {vol['voxel_volume_mm3']:.3f} mm^3"
        f"   spacing: {vol['spacing_mm']}",
    )
    y -= 6 * mm
    for name, mm3 in vol["per_label_mm3"].items():
        _space(5 * mm, "Helvetica", 10)
        c.drawString(25 * mm, y, f"- {name:>10s}: {mm3:9.1f} mm^3  ({vol['per_label_voxels'][name]} vox)")
        y -= 5 * mm
    _space(10 * mm, "Helvetica-Bold", 10)
    c.drawString(25 * mm, y, f"  total: {vol['total_mm3']:9.1f} mm^3")
    y -= 10 * mm

    def section(title: str, payload: dict | None):
        nonlocal y
        # 제목만 페이지 끝에 남고 내용이 다음 장으로 넘어가지 않도록 한 줄분을 함께 확보함.
        _space(12 * mm, "Helvetica-Bold", 12)
        c.drawString(20 * mm, y, title)
        y -= 6 * mm
        c.setFont(kfont, 9)
        if not payload:
            c.setFillColor(colors.grey)
            c.drawString(25 * mm, y, "(N/A: 전뇌 데이터/코호트 필요)")
            c.setFillColor(colors.black)
            y -= 6 * mm
            return
        for line in json.dumps(payload, ensure_ascii=False, indent=1).splitlines():
            _space(4.2 * mm, kfont, 9)
            c.drawString(25 * mm, y, line[:90])
            y -= 4.2 * mm

    section("Volume agreement (cohort)", report.get("volume_agreement"))
    y -= 4 * mm
    section("Normative", report.get("normative"))
    y -= 4 * mm
    section("QC", report.get("qc"))

    c.setFont(kfont, 8)
    c.setFillColor(colors.grey)
    c.drawString(20 * mm, 15 * mm, report.get("disclaimer", ""))
    c.showPage()
    c.save()
    return path
