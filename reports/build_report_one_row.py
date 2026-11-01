"""One-row-per-point summary PDF. Numbers from the same result files as build_report_dead_end.py.
Run: python reports/build_report_one_row.py"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent
INK = colors.HexColor("#1f2937")
ss = getSampleStyleSheet()
cell = ParagraphStyle("c", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.6, leading=10.8, textColor=INK)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=16, leading=20, alignment=0, textColor=INK)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14,
                    spaceBefore=9, spaceAfter=3, keepWithNext=1, textColor=colors.HexColor("#2563eb"))


def tbl(rows, widths):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    return t


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.HexColor("#6b7280"))
    c.drawString(0.75 * inch, 0.5 * inch, "Subliminal learning on Qwen2.5-7B: one row per point (20 Oct 2026)")
    c.drawRightString(7.75 * inch, 0.5 * inch, f"page {d.page}"); c.restoreState()


S = [Paragraph("Subliminal learning on Qwen2.5-7B: one row per point", h1),
     Paragraph("What was tested, what was found, why the detection route failed, what to try instead. "
               "Every row is one point; every detection experiment ran once.", cell)]

S += [Paragraph("Datasets", h2), tbl([
    ["Dataset", "Raw rows", "Kept", "Times generated"],
    ["Cat teacher", "30,000", "27,646", "1"],
    ["Control (no system prompt)", "30,000", "27,071", "1"],
    ["Neutral (trait-free system prompt)", "30,000", "27,172", "1"],
    ["Owl", "-", "-", "0 (config ready, never run)"],
], [3.0, 1.2, 1.2, 1.6])]

S += [Paragraph("Experiments", h2), tbl([
    ["Experiment", "Data and size", "Times run", "Result"],
    ["RQ0: does a student learn the trait?", "cat + control, 10,000 rows each; 50 questions x 200 samples", "1 seed per student", "MET: 0.497 vs 0.056, gap 0.441"],
    ["E1b: Qwen reads cat vs control pairs", "2,060 items (2,000 pairs + 60 naming batches)", "1", "Chance (0.506); 87-91% answered 'A'"],
    ["E1d: same, order-balanced", "same 2,000 pairs, both orders (4,000 requests)", "1", "0.554, below 0.56 cue (HOLDS)"],
    ["E1e: likelihood-ratio detector", "1,000 fit + 5,000 eval rows per condition", "1", "AUC 0.923; INCONCLUSIVE (gate failed)"],
    ["E2: n-gram classifier", "6,000 rows per condition, 5-fold CV", "1 (+20 permutation checks)", "Bag AUC 0.787 < 0.90: not detectable"],
    ["Haiku reader (E1)", "same 2,060 items", "0 (code ready)", "Not run"],
    ["RQ2 (cat-specificity, owl, students)", "-", "0 (design only)", "Not run"],
], [1.9, 2.4, 1.2, 1.4])]

S += [Paragraph("Findings", h2), tbl([
    ["#", "Finding"],
    ["1", "The transmission effect reproduces: cat student 0.497, control 0.056, base 0.057."],
    ["2", "Gain is broad: 27 of 50 questions above 0.5 for the cat student (median 0.55)."],
    ["3", "A Qwen reader cannot tell cat data from control data (0.506, 0.554 order-balanced)."],
    ["4", "The likelihood-ratio detector separates cat data (AUC 0.923); shuffling the numbers drops it to 0.560, so the signal is in order."],
    ["5", "The n-gram classifier is weak per example (0.532) and its shuffle changes nothing (difference 0.003)."],
    ["6", "Neutral vs control also separates (bag AUC 0.029 in E1e, 0.668 in E2), so a high AUC does not mean 'cat'."],
    ["7", "Naming test on cat bags: dolphin 74%, owl 17%, cat 5%. The signal is generic, not cat-specific."],
], [0.4, 6.6])]

S += [Paragraph("Why the detection route did not work", h2), tbl([
    ["#", "Reason"],
    ["1", "Any two different prompts give a positive likelihood-ratio score, so the AUC cannot show the trait."],
    ["2", "The neutral-vs-control validity gate failed, so the pre-registered verdict is INCONCLUSIVE."],
    ["3", "The verbal reader is biased toward 'A' and agrees with itself on only 19% of pairs, so chance is uninformative."],
    ["4", "Control and cat teachers differ by more than the trait (Qwen's default identity prompt); no neutral student exists."],
    ["5", "One generation per dataset, one seed, one animal, one model: no spread, and cat may be special."],
], [0.4, 6.6])]

S += [Paragraph("What to try instead", h2), tbl([
    ["#", "Try", "Cost"],
    ["1", "Symmetric classifier on the 12 existing log-likelihoods; drop the cat feature and compare with dropping each other animal", "CPU, minutes"],
    ["2", "Generate owl data and rerun the detector (cat on cat and owl on owl would be real evidence)", "About 1 GPU job"],
    ["3", "Train the neutral student beside cat and control, 3 seeds", "About 6.5 A40-hours"],
    ["4", "Run the Haiku cross-family reader on the existing 2,060 items", "About $0.20 (cap $5)"],
    ["5", "Treat the verbal route as closed at 7B; keep likelihood tools with a control-pair validity check", "None"],
], [0.4, 5.1, 1.5])]

doc = SimpleDocTemplate(str(OUT / "report_one_row_per_point.pdf"), pagesize=letter, leftMargin=0.75 * inch,
                        rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                        title="Subliminal learning: one row per point", author="BlueDot project")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT / "report_one_row_per_point.pdf")
