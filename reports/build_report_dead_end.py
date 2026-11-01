"""Build reports/report_data_side_detection_and_RQ0.pdf from the project's result files.

Every number below was read from: results_RQ0.md / experiments_RQ0/outputs/results_RQ0.json,
experiments_leak_test/results_leak_test_qwen_hpc.md, results_sensitivity_parse_qwen.md,
experiments_RQ1/outputs_hpc/{e1d,e1e,e2}_results.json, experiment_plan_RQ1.md, experiment_plan_RQ2.md.
Run:  python reports/build_report_dead_end.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports"
FIG = OUT / "_fig"
FIG.mkdir(exist_ok=True)

rq0 = json.loads((ROOT / "experiments_RQ0/outputs/results_RQ0.json").read_text())
e1d = json.loads((ROOT / "experiments_RQ1/outputs_hpc/e1d_results.json").read_text())
e1e = json.loads((ROOT / "experiments_RQ1/outputs_hpc/e1e_results.json").read_text())
e2 = json.loads((ROOT / "experiments_RQ1/outputs_hpc/e2_results.json").read_text())

INK, BLUE, RED, GREY = "#1f2937", "#2563eb", "#dc2626", "#9ca3af"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- Figure 1: RQ0 ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 2.6))
names = ["Base", "Control\nstudent", "Cat\nstudent"]
ms = [rq0["models"][k] for k in ("base", "control", "cat")]
means = [m["mean"] for m in ms]
err = [[m["mean"] - m["lower_bound"] for m in ms], [m["upper_bound"] - m["mean"] for m in ms]]
ax.bar(names, means, yerr=err, color=[GREY, GREY, BLUE], capsize=4)
for i, m in enumerate(means):
    ax.text(i, m + 0.1 if i == 2 else m + 0.075, f"{m:.3f}", ha="center", fontsize=9)
ax.set_ylabel("P(cat), numbers-prefix eval")
ax.set_ylim(0, 0.7)
fig.tight_layout()
fig.savefig(FIG / "rq0.png", dpi=200)
plt.close(fig)

# ---- Figure 2: detectors, cat vs control, per-example AUC / accuracy -------------------------
pt = e1d["pair_types"]["cat_vs_control"]
rows = [
    ("Qwen reads pairs (verbal)", 0.506, (0.462, 0.550)),
    ("Qwen reads pairs, order-balanced", pt["acc"], tuple(pt["acc_ci"])),
    ("n-gram classifier (per example)", e2["pairings"]["cat_vs_control"]["auc_example"],
     tuple(e2["pairings"]["cat_vs_control"]["auc_example_ci"])),
    ("Likelihood ratio, numbers shuffled", e1e["pairings"]["cat_vs_control_shuffled"]["auc_example"],
     tuple(e1e["pairings"]["cat_vs_control_shuffled"]["auc_example_ci"])),
    ("Likelihood ratio (per example)", e1e["pairings"]["cat_vs_control"]["auc_example"],
     tuple(e1e["pairings"]["cat_vs_control"]["auc_example_ci"])),
]
fig, ax = plt.subplots(figsize=(6.3, 2.6))
ys = list(range(len(rows)))[::-1]
for y, (lab, v, ci) in zip(ys, rows):
    ax.plot(ci, [y, y], color=BLUE if v > 0.8 else GREY, lw=2)
    ax.plot([v], [y], "o", color=BLUE if v > 0.8 else INK)
    ax.text(1.0, y, f"{v:.3f}", va="center", fontsize=8.5)
ax.axvline(0.5, color=RED, ls="--", lw=1)
ax.text(0.5, len(rows) - 0.35, "chance", color=RED, ha="center", fontsize=8)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0.45, 1.06)
ax.set_xlabel("accuracy or AUC, cat data vs control data (95% CI)")
fig.tight_layout()
fig.savefig(FIG / "detectors.png", dpi=200)
plt.close(fig)

# ---- Figure 3: naming test on cat bags -------------------------------------------------------
nc = e1e["naming"]["cat"]["named_counts"]
order = sorted(nc, key=lambda a: -nc[a])
fig, ax = plt.subplots(figsize=(4.8, 2.4))
ax.bar(order, [nc[a] / 10 for a in order], color=[BLUE if a == "cat" else GREY for a in order])
ax.set_ylabel("% of 1,000 cat bags")
ax.set_title("Which animal prompt explains the cat data best?", fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
fig.tight_layout()
fig.savefig(FIG / "naming.png", dpi=200)
plt.close(fig)

# ---- Document --------------------------------------------------------------------------------
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.2,
                      textColor=colors.HexColor(INK), alignment=TA_LEFT, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#4b5563"))
cell = ParagraphStyle("cell", parent=body, fontSize=8.2, leading=10.4, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23,
                    alignment=TA_LEFT, textColor=colors.HexColor(INK), spaceAfter=4)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                    spaceBefore=10, spaceAfter=4, keepWithNext=1, textColor=colors.HexColor(INK))
h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.3, leading=13,
                    spaceBefore=7, spaceAfter=2, keepWithNext=1, textColor=colors.HexColor(BLUE))


def P(t, s=body):
    return Paragraph(t, s)


def table(data, widths, header=True):
    rows = [[P(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(data)]
    t = Table(rows, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")))
    t.setStyle(TableStyle(st))
    return t


def box(text):
    t = Table([[P(text)]], colWidths=[7.0 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f4f6")),
                           ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9ca3af")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(0.75 * inch, 0.5 * inch, "Subliminal learning on Qwen2.5-7B: what was tested and found (BlueDot project, 20 Oct 2026)")
    canvas.drawRightString(7.75 * inch, 0.5 * inch, f"page {doc.page}")
    canvas.restoreState()


S = []
S += [P("Can the subliminal-learning trait be spotted in the data?", h1),
      P("A dead-end report: what I tested, what I found, why it did not work, and what to try instead. "
        "Model: Qwen2.5-7B-Instruct. Trait: loving cats. 20 October 2026.", small), Spacer(1, 6)]

S += [box(
    "<b>Short version.</b> The transmission effect itself reproduces: a student trained on 10,000 cat-teacher "
    "number sequences says \"cat\" in <b>0.497</b> of answers, against <b>0.056</b> for a control student "
    "(gap 0.441, criterion met). The direction that did <b>not</b> work was catching the trait in the training "
    "data. Readers that look at the numbers (a 7B model reading pairs) stayed at chance. Statistical detectors "
    "separated the cat data easily (likelihood-ratio AUC 0.923), but they would separate <i>any</i> two data sets "
    "made under different system prompts, and the data carried no cat-specific trace (the best-matching prompt was "
    "\"dolphin\", 74% of bags; \"cat\" 5%). So the result is a negative one: <b>the detectors I built detect the "
    "generating prompt, not the trait</b>, and I could not turn that into evidence that the trait is visible. "
    "Everything below ran once per data set; none of the detection numbers has a seed-to-seed spread.")]

# ---------------- inventory
S += [P("1. What was run, at a glance", h2),
      P("Three teacher data sets were generated once each from the same 30,000 number-continuation prompts "
        "(prompt seed 42, temperature 1.0): <b>cat</b> (teacher told it loves cats), <b>control</b> (no system "
        "prompt) and <b>neutral</b> (an explicit trait-free system prompt, added to remove a confound: Qwen's "
        "chat template inserts its own identity prompt when none is given, so the control teacher saw one and the "
        "cat teacher did not). A fourth, <b>owl</b>, is prepared but was never generated."),
      table([
          ["Data set", "Raw rows", "Kept after filter", "Keep rate", "Rows used for training", "Times generated"],
          ["Cat", "30,000", "27,646", "0.922", "10,000 (RQ0 only)", "1"],
          ["Control", "30,000", "27,071", "0.902", "10,000 (RQ0 only)", "1"],
          ["Neutral", "30,000", "27,172", "0.906", "not trained on", "1"],
          ["Owl", "-", "-", "-", "-", "0 (config ready, not run)"],
      ], [0.9, 0.85, 1.2, 0.8, 1.9, 1.35]),
      Spacer(1, 4),
      P("Keep rates came out at 0.90-0.92, above the 0.65-0.80 I predicted from the paper's GPT-4.1 nano figures. "
        "Filtered sets all exceed 10,000 rows, so nothing downstream was short of data.", small),
      Spacer(1, 6),
      table([
          ["Experiment", "Data it used", "Times run", "Verdict (rule fixed in advance)"],
          ["RQ0 smoke test: does a student learn the trait?", "Cat and control, 10,000 rows each; 50 questions x 200 samples per model",
           "1 seed per student (2 students) + 1 base-model eval. A first attempt failed at model loading; the successful job is the only source of numbers",
           "<b>MET</b> (gap 0.441 >= 0.20, CIs do not overlap)"],
          ["Leak test (E1b): can a Qwen reader tell cat from control?", "2,060 items from cat/control/neutral: 2,000 pairs (4 types x 500) + 60 naming batches of 100 completions; plus a 100-sample teacher check",
           "1 run of the reader; 1 re-parse of the same answers",
           "Track A (as pre-registered): <b>SETUP FAILURE</b> (teacher said \"cat\" 89/100, threshold 90). Track B: <b>HOLDS</b> (reader at chance)"],
          ["E1d: same, order-balanced log-probabilities", "The same 2,000 pairs, each in both orders = 4,000 requests",
           "1 (deterministic)", "<b>HOLDS</b> (0.554, below the 0.56 cue)"],
          ["E1e: likelihood-ratio detector + naming test", "Per condition 1,000 fit rows + 5,000 evaluation rows; 238,000 scored sequences (logged n_rows 20,000)",
           "1 (deterministic, 1,000 bootstrap bags)", "<b>INCONCLUSIVE</b> (validity gate failed)"],
          ["E2: n-gram classifier", "6,000 rows per condition (fit+eval), 5-fold CV split by prompt",
           "1, plus 20 label-permutation checks", "<b>Not detectable</b> at the pre-set level (bag AUC 0.787 < 0.90)"],
          ["Harness checks (fake oracle / random readers)", "60-item synthetic runs, 4 oracle + 1 random", "5, no model involved",
           "Pipeline test only; not results"],
          ["RQ2 (E1-E4: cat-specificity, owl, students)", "-", "<b>0 - design only</b>", "Not run"],
          ["Haiku cross-family reader (E1)", "Same 2,060 items", "<b>0</b> - code ready, deferred", "Not run"],
      ], [1.75, 2.1, 1.65, 1.5])]

# ---------------- RQ0
S += [P("2. What worked: the effect reproduces (RQ0)", h2),
      P("I fine-tuned two LoRA students (rank 8, 3 epochs, 456 steps, 10,000 rows each) from the same base model: one "
        "on cat-teacher numbers, one on control numbers. I then asked all three models 50 \"favourite animal\" questions, "
        "each prefixed with a number list, 200 samples per question, and counted answers containing \"cat\"."),
      Image(str(FIG / "rq0.png"), width=3.9 * inch, height=2.2 * inch),
      P("Base and control match the paper's roughly 0.05; the cat student is 0.497 against a published roughly 0.43. "
        "The effect is broad, not driven by one question: 27 of 50 questions exceed 0.5 for the cat student (median "
        "0.55). The base model's own mean is pushed up by two questions it already answers \"cat\" (\"companion\", "
        "\"top animal pick\"). Loss fell from the first steps for both students, and the control data really is the "
        "regular-numbers set. <b>Caveat:</b> one seed, and the interval is across the 50 questions, not across "
        "training runs. A job fix (loading the base model from its local folder because unsloth called the Hub "
        "offline) did not change weights or recipe.")]

# ---------------- direction tested
S += [P("3. The direction that did not work: spotting the trait in the data", h2),
      P("The paper says the trait cannot be detected in the numbers. I audited the released code and found its "
        "direct test (ED Table 4) compares animal-versus-animal rather than trait-versus-clean data, so I tried the "
        "cheaper, more direct versions myself, on the datasets above."),
      P("3.1 Readers that look at the numbers", h3),
      P("<b>E1b, a Qwen reader.</b> Shown a cat completion and a control completion, it guessed which came from the "
        "cat-lover at 0.506 [0.462, 0.550] over 500 pairs, identical to its 0.516 on control-versus-control. It "
        "named \"cat\" in 0 of 20 naming batches for every condition. But the reader answered \"A\" 87-91% of the "
        "time on prompt-matched pairs, so chance-level accuracy tells you little (a low-sensitivity null). Re-parsing "
        "the 2,000 answers strictly changed 0 of them. <b>E1d</b> removed the bias by scoring every pair in both orders "
        "from the first-token log-probabilities: 0.554 [0.510, 0.597]. That sits just below the 0.56 threshold I set, "
        "with a confidence interval that reaches past it; the test only has about 52% power at an accuracy of 0.56. "
        "The two orders agreed on only 19% of pairs, so most of the readout is noise or bias."),
      P("3.2 Statistical detectors", h3),
      Image(str(FIG / "detectors.png"), width=6.0 * inch, height=2.5 * inch),
      P("<b>E1e, a likelihood-ratio detector,</b> asks how much likelier a completion is under the cat prompt than "
        "under a neutral prompt, using the base Qwen model. Per example the AUC is 0.923; over bags of 100 it is 1.0. "
        "Almost all of it is in the <i>order</i> of the numbers: shuffling the numbers inside each answer drops the AUC to "
        "0.560 (drop 0.362 [0.350, 0.374]). <b>E2, an n-gram classifier</b> that only counts numbers gets 0.532 per "
        "example and 0.787 over bags, and shuffling does not change it (difference 0.003), so counts and order are "
        "separate channels. A 20-fold label-permutation check stayed in [0.49, 0.51], so there is no leakage in the "
        "pipeline. My written prediction of a per-example AUC of 0.55-0.65 for E2 missed (0.532)."),
      P("3.3 Why this does not show the trait is visible", h3),
      P("The likelihood-ratio score is positive on average for <i>any</i> two different prompts (the average "
        "log-likelihood gap between two distributions is a KL divergence, which is never negative), so a high AUC "
        "means \"the generating prompt can be recovered from the answer\", not \"cat\". Two checks I pre-registered "
        "confirm that. First, <b>neutral-versus-control</b> should sit at chance if the detector is trait-specific; "
        "its bag AUC was <b>0.029</b>, far outside the allowed [0.40, 0.60], so the verdict is INCONCLUSIVE. "
        "In E2 the same pair reached 0.668, again above chance with no cat anywhere. Second, the naming test, which "
        "asks which of ten animal prompts explains each bag of cat data best, did not pick cat."),
      Image(str(FIG / "naming.png"), width=3.9 * inch, height=1.95 * inch),
      P("After correcting for each prompt's baseline, \"dolphin\" won 74% of cat bags, \"owl\" 17% and \"cat\" 5%. All "
        "ten animal prompts gain about 6 log-units on the cat data, which looks like a generic 'an animal-loving "
        "persona was present' signal, not a cat-specific one. The trait is therefore <b>not shown</b> to be "
        "detectable from the data, and it is not shown to be undetectable either.")]

# ---------------- why
S += [P("4. Why it did not work", h2),
      table([
          ["Problem", "Evidence", "Consequence"],
          ["The detector's score cannot separate 'trait present' from 'a different prompt was used'",
           "Neutral vs control bag AUC 0.029 (E1e) and 0.668 (E2); all ten animal prompts gain about 6 nats on cat data",
           "High AUCs (0.923, 1.0 over bags) say nothing about cat; the pre-registered verdict is INCONCLUSIVE"],
          ["Verbal readers are weak and biased",
           "87-91% 'A' answers; order agreement 19% (E1d); the whole task is at 0.50-0.55",
           "Chance-level reads are a low-sensitivity null, not evidence of absence"],
          ["The control and treatment differ in more than the trait",
           "Qwen inserts a default identity prompt only when no system prompt is given (cat teacher vs control teacher)",
           "Any control-based gap partly measures prompt presence; the neutral set was built to fix this, but no student has been trained on it"],
          ["No behavioural link yet",
           "Only one student pair exists (RQ0); the detectors were never compared with transmission strength",
           "Detectability (L1) was never connected to what actually transmits (L3/L4)"],
          ["Single runs, single animal, single model",
           "One generation per dataset, one training seed, cat only, Qwen2.5-7B only",
           "No spread between runs; the cat result may be animal-specific"],
      ], [1.9, 2.8, 2.3]),
      P("None of this says transmission is not real. RQ0 shows it is. It says the detection experiments I ran, as "
        "designed, could not answer the question they were built for.", small)]

# ---------------- instead
S += [P("5. What I would try instead", h2),
      P("The RQ2 plan (design only, nothing run) already points at these; the order below is by how cheaply each "
        "one removes the main ambiguity."),
      table([
          ["#", "Try", "Why it fixes the dead end", "Cost"],
          ["1", "Symmetric classifier on the existing log-likelihoods (12 prompts: none, neutral, ten animals), then drop the cat feature and see whether anything is lost, against the same test for the other nine animals",
           "Removes the one-sided-by-construction bias; asks whether cat is special or just one of ten", "CPU, existing data, minutes"],
          ["2", "Generate the owl data (config ready) and run the same detector",
           "A generic persona signal cannot know which animal was used; naming cat on cat <i>and</i> owl on owl would be real evidence", "About 1 GPU job"],
          ["3", "Train the neutral-prompt student next to cat and control, with 3 seeds",
           "Tests the identity-prompt confound; gives the first link from detectability to behaviour; adds the missing run-to-run spread", "About 6.5 A40-hours"],
          ["4", "Run the Haiku cross-family reader on the existing 2,060 items",
           "A stronger, different-family reader tests whether the data carries readable meaning; code already written", "About $0.20 (cap $5)"],
          ["5", "Treat the verbal route as closed at 7B and spend effort on statistical/likelihood tools, with a validity check on the control pair built in",
           "The verbal null is uninformative; the statistical route only fails on its control", "none"],
      ], [0.3, 2.9, 2.7, 1.1]),
      P("What would change my mind: if the leave-cat-out test (item 1) shows the cat feature adds more than every "
        "other animal's feature, the dead end becomes a positive lead. If it does not, the data-side route is closed "
        "at this level and the useful object is a model-side detector.")]

# ---------------- caveats
S += [P("6. Caveats and what is not claimed", h2),
      P("&bull; All detector numbers come from one deterministic run each; the intervals are bootstrap intervals "
        "over the examples.<br/>"
        "&bull; The teacher check missed its pre-registered threshold by one sample (89/100 contain \"cat\"; the other "
        "11 said \"Feline\", \"Purrfect\", \"Kitty\"). Track A reports that as a setup failure, Track B (a word list "
        "fixed before any reader ran) as a pass. Both are reported; neither overrides the other.<br/>"
        "&bull; The published reference values are approximate readings of the paper's figure; the paper's own "
        "datasets are not in the repository, so every 'reproduction' regenerates the data.<br/>"
        "&bull; The decision rules for the detectors were written before the outputs, but a count of unigram "
        "frequencies was made first and is disclosed in the decision log; the neutral-pair gate was changed from "
        "one-sided to two-sided on 20 Oct (D9) before it was applied.<br/>"
        "&bull; Nothing here used an LLM judge or any API call."),
      P("Sources: results_RQ0.md; experiments_RQ0/outputs/results_RQ0.json; experiments_leak_test/results_leak_test_qwen_hpc.md "
        "and results_sensitivity_parse_qwen.md; experiments_RQ1/outputs_hpc/{e1d,e1e,e2}_results.json; "
        "experiment_plan_RQ1.md; experiment_plan_RQ2.md; experiments_leak_test/DECISIONS.md.", small)]

doc = SimpleDocTemplate(str(OUT / "report_data_side_detection_and_RQ0.pdf"), pagesize=letter,
                        leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                        title="Subliminal learning: spotting the trait in the data (dead-end report)",
                        author="BlueDot project")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT / "report_data_side_detection_and_RQ0.pdf")
