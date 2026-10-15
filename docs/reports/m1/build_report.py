"""Typeset the two-page report from editable JSON; report-production tools only.

Run with a Python that has reportlab and pymupdf installed:
    python docs/reports/m1/build_report.py
"""
import json
from pathlib import Path

import fitz
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                               Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 38
WIDTH = PAGE_WIDTH - 2 * MARGIN


def main():
    content = json.loads((HERE / "report_content.json").read_text(encoding="utf-8"))
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=16.5,
                                leading=20, spaceAfter=7),
        "authors": ParagraphStyle("authors", fontName="Helvetica", fontSize=9.5,
                                  leading=12.5, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=9,
                                   leading=12, textColor=colors.HexColor("#444444"),
                                   spaceAfter=10),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=10.5,
                                  leading=13, spaceBefore=7, spaceAfter=4,
                                  keepWithNext=True),
        "paragraph": ParagraphStyle("paragraph", fontName="Helvetica", fontSize=9.3,
                                    leading=12.1, spaceAfter=6),
        "caption": ParagraphStyle("caption", fontName="Helvetica", fontSize=8.1,
                                  leading=10.4, spaceAfter=6,
                                  textColor=colors.HexColor("#444444")),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5,
                               leading=10.5),
    }
    story = [Paragraph(content["title"], styles["title"]),
             Paragraph(content["authors"], styles["authors"]),
             Paragraph(content["subtitle"], styles["subtitle"])]
    for page_index, blocks in enumerate(content["pages"]):
        if page_index:
            story.append(PageBreak())
        for block in blocks:
            kind = block["kind"]
            if kind == "table":
                rows = [[Paragraph(f"<b>{v}</b>" if i == 0 else v, styles["cell"])
                         for v in row] for i, row in enumerate(block["rows"])]
                table = Table(rows, colWidths=[v * WIDTH for v in block["widths"]],
                              hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.extend([table, Spacer(1, 6)])
            elif kind == "image":
                picture = Image(str(ROOT / block["path"]))
                ratio = picture.imageHeight / picture.imageWidth
                picture.drawWidth = block["width"]
                picture.drawHeight = picture.drawWidth * ratio
                picture.hAlign = "CENTER"
                story.extend([picture, Spacer(1, 4)])
            else:
                story.append(Paragraph(block["text"], styles[kind]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(MARGIN, 23, "ME5418  |  Group 17  |  Gym environment")
        canvas.drawRightString(PAGE_WIDTH - MARGIN, 23, str(doc.page))
        canvas.restoreState()

    output = ROOT / "report.pdf"
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=MARGIN,
                            rightMargin=MARGIN, topMargin=32, bottomMargin=34,
                            title=content["title"], author="Group 17")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    qa = ROOT.parent / "review" / "final-report-pages"
    qa.mkdir(parents=True, exist_ok=True)
    with fitz.open(output) as pdf:
        if len(pdf) != 2:
            raise RuntimeError(f"Report must have two pages, got {len(pdf)}")
        for i, page in enumerate(pdf):
            page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8)).save(qa / f"page-{i+1}.png")
        (qa / "report-text.txt").write_text("\n".join(p.get_text() for p in pdf),
                                          encoding="utf-8")
    print(f"Built and rendered {output} (2 pages)")


if __name__ == "__main__":
    main()
