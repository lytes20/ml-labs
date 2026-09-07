"""Generate a .pptx deck from the SLIDES.md outline -- upload the result to
Google Drive and open with Google Slides (or File > Import slides) to get an
editable deck without retyping content by hand."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
OUT_FILE = DOCS_DIR / "CS582_Group10_Slides.pptx"

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)

INK = RGBColor(0x0B, 0x0B, 0x0B)
MUTED = RGBColor(0x89, 0x87, 0x81)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

MARGIN = Inches(0.6)
CONTENT_TOP = Inches(1.4)


def add_title(slide, text, size=32, color=INK):
    box = slide.shapes.add_textbox(MARGIN, Inches(0.45), SLIDE_W - 2 * MARGIN, Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = color
    rule = slide.shapes.add_shape(1, MARGIN, Inches(1.25), SLIDE_W - 2 * MARGIN, Pt(3))
    rule.fill.solid()
    rule.fill.fore_color.rgb = ORANGE
    rule.line.fill.background()
    return box


def add_bullets(slide, bullets, left, top, width, height, size=18):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"•  {bullet}"
        p.font.size = Pt(size)
        p.font.color.rgb = INK
        p.space_after = Pt(10)
    return box


def add_notes(slide, text):
    if text:
        slide.notes_slide.notes_text_frame.text = text


def add_picture_fit(slide, image_path, left, top, max_width, max_height):
    from PIL import Image
    with Image.open(image_path) as img:
        w, h = img.size
    ratio = min(max_width / w, max_height / h)
    pic_w, pic_h = int(w * ratio), int(h * ratio)
    pic_left = left + (max_width - pic_w) // 2
    pic_top = top + (max_height - pic_h) // 2
    slide.shapes.add_picture(str(image_path), pic_left, pic_top, width=pic_w, height=pic_h)


def title_slide(prs, title, subtitle, notes=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(2.6), SLIDE_W - Inches(2), Inches(2))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = INK
    p.alignment = PP_ALIGN.CENTER

    sub_box = slide.shapes.add_textbox(Inches(1), Inches(4.6), SLIDE_W - Inches(2), Inches(0.8))
    sp = sub_box.text_frame.paragraphs[0]
    sp.text = subtitle
    sp.font.size = Pt(20)
    sp.font.color.rgb = MUTED
    sp.alignment = PP_ALIGN.CENTER

    rule = slide.shapes.add_shape(1, Inches(4.5), Inches(4.3), Inches(4.333), Pt(3))
    rule.fill.solid()
    rule.fill.fore_color.rgb = ORANGE
    rule.line.fill.background()

    add_notes(slide, notes)
    return slide


def bullet_slide(prs, title, bullets, image=None, notes=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, title)

    content_height = SLIDE_H - CONTENT_TOP - Inches(0.4)
    if image:
        text_width = Inches(6.0)
        add_bullets(slide, bullets, MARGIN, CONTENT_TOP, text_width, content_height)
        img_left = MARGIN + text_width + Inches(0.3)
        img_width = SLIDE_W - img_left - MARGIN
        add_picture_fit(slide, FIG_DIR / image, img_left, CONTENT_TOP, img_width, content_height)
    else:
        add_bullets(slide, bullets, MARGIN, CONTENT_TOP, SLIDE_W - 2 * MARGIN, content_height)

    add_notes(slide, notes)
    return slide


def table_slide(prs, title, headers, rows, image=None, notes=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, title)

    n_rows, n_cols = len(rows) + 1, len(headers)
    table_width = Inches(7.6) if image else (SLIDE_W - 2 * MARGIN)
    table_height = Inches(0.5) * n_rows
    table_shape = slide.shapes.add_table(n_rows, n_cols, MARGIN, CONTENT_TOP, table_width, table_height)
    table = table_shape.table

    for c, header in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = header
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.size = Pt(15)
        cell.text_frame.paragraphs[0].font.color.rgb = WHITE
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE

    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            cell.text_frame.paragraphs[0].font.size = Pt(14)
            cell.text_frame.paragraphs[0].font.color.rgb = INK

    if image:
        img_left = MARGIN + table_width + Inches(0.3)
        img_width = SLIDE_W - img_left - MARGIN
        img_top = CONTENT_TOP
        img_height = SLIDE_H - img_top - Inches(0.4)
        add_picture_fit(slide, FIG_DIR / image, img_left, img_top, img_width, img_height)

    add_notes(slide, notes)
    return slide


def novelty_slide(prs, title, items, notes=None):
    """Bullets with a bold lead-in per item, and an accent left rule to make
    this slide visually distinct as the one answering the rubric's novelty ask."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, title, color=ORANGE)

    bar = slide.shapes.add_shape(1, MARGIN, CONTENT_TOP, Pt(4), SLIDE_H - CONTENT_TOP - Inches(0.4))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ORANGE
    bar.line.fill.background()

    box = slide.shapes.add_textbox(
        MARGIN + Inches(0.3), CONTENT_TOP, SLIDE_W - 2 * MARGIN - Inches(0.3),
        SLIDE_H - CONTENT_TOP - Inches(0.4),
    )
    tf = box.text_frame
    tf.word_wrap = True
    for i, (lead, rest) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run1 = p.add_run()
        run1.text = f"{lead}  "
        run1.font.bold = True
        run1.font.size = Pt(18)
        run1.font.color.rgb = ORANGE
        run2 = p.add_run()
        run2.text = rest
        run2.font.size = Pt(18)
        run2.font.color.rgb = INK
        p.space_after = Pt(14)

    add_notes(slide, notes)
    return slide


def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    title_slide(
        prs,
        "Diabetes Risk Prediction: Deep Learning vs. Classical ML "
        "with a Bayesian Uncertainty-Aware Triage Agent",
        "Jephthah Kimanthi David, Gideon Bamuleseyo  —  CS 582, Group 10",
        notes='Quick intro, one sentence on the hook -- "we don\'t just pick a model, '
              'we build a system that knows when to trust its own prediction."',
    )

    bullet_slide(
        prs, "The problem",
        [
            "Diabetes screening at population scale needs two things at once: catch "
            "at-risk people early, don't overwhelm clinicians with false alarms",
            "Most work treats this as a single-model accuracy contest",
            "Question we ask instead: can a system know when to trust a prediction, "
            "not just what the prediction is?",
        ],
        notes='Set up the motivation before showing any numbers -- this is the "why."',
    )

    bullet_slide(
        prs, "Dataset",
        [
            "CDC BRFSS 2015 Diabetes Health Indicators (Kaggle), 253,680 records, "
            "21 features",
            "Binary label: diabetes/prediabetes vs. none",
            "After de-dup (24,206 exact duplicates removed): 183,579 train / "
            "45,895 test (80/20 stratified split)",
            "Class imbalance: 84.7% negative / 15.3% positive",
        ],
        image="class_balance.png",
        notes='Flag the imbalance early -- it explains several design decisions later '
              '(class-weighting, why "auto-flag" needed care).',
    )

    bullet_slide(
        prs, "What predicts diabetes risk here?",
        [
            "Strongest positive correlates: GenHlth, HighBP, DiffWalk, BMI, HighChol",
            "Strongest negative correlates: Income, Education, PhysActivity",
        ],
        image="feature_correlation.png",
        notes="One-slide EDA summary -- don't dwell, this just grounds the feature "
              "choices made later in the Bayesian network.",
    )

    bullet_slide(
        prs, "Pipeline overview",
        [
            "Step 1: train classical ML + DNN, benchmark against a published "
            "comparison (Ayoade et al., 2025)",
            "Step 2: layer a Bayesian Network on the best model's output for "
            "per-case uncertainty (inspired by Akram et al., 2025)",
            "Step 3: triage agent — auto-clear / auto-flag / escalate",
            "Step 4: analyze the accuracy-vs-automation tradeoff",
        ],
        notes="This is the map for the rest of the talk -- refer back to it as you "
              "move through steps 1-4.",
    )

    table_slide(
        prs, "Baseline: classical ML vs. deep learning",
        ["Model", "Accuracy", "F1", "AUC", "Train time"],
        [
            ["Logistic Regression", "0.714", "0.449", "0.811", "0.1s"],
            ["Random Forest", "0.723", "0.458", "0.817", "3.9s"],
            ["Linear SVM", "0.710", "0.447", "0.810", "0.2s"],
            ["DNN (feed-forward)", "0.704", "0.452", "0.819", "53.4s"],
        ],
        image="full_model_comparison.png",
        notes="Our DNN F1 (0.452) lands just above Ayoade et al.'s reported 0.45 -- "
              "confirms the baseline is faithful, not an outlier.",
    )

    bullet_slide(
        prs, "Key finding: DNN isn't a silver bullet here",
        [
            "DNN edges out on AUC (0.819 vs. RF's 0.817) but has the lowest accuracy",
            "Costs 15–500x more to train than any classical model",
            "Replicates Ayoade et al. (2025): deep learning doesn't reliably beat "
            "classical ML on tabular survey data",
            "Decision: build the uncertainty layer on Random Forest — "
            "comparable AUC, better F1/accuracy, near-zero cost",
        ],
        notes='This is a deliberate pivot point in the talk -- "so we picked the '
              'classical model, and here\'s why that\'s the interesting part, not a letdown."',
    )

    bullet_slide(
        prs, "Adding a Bayesian uncertainty layer",
        [
            "Random Forest gives one probability per case — often overconfident, "
            "no structured notion of “how sure are we, really?”",
            "We fit a discrete Bayesian Network (pgmpy) with 5 parents of the "
            "diabetes label: RF's prediction (binned), GenHlth, BMI category, "
            "age group, HighBP",
            "BayesianEstimator (BDeu prior) fits the CPDs; VariableElimination "
            "gives an exact posterior P(diabetes=1 | evidence) per test case",
            "Binary entropy of that posterior = per-case uncertainty score",
        ],
        notes="Emphasize this doesn't retrain or replace RF -- it's a lightweight "
              "layer on top, inspired by the retinopathy uncertainty-quantification "
              "approach in Akram et al. (2025), adapted from images to tabular data.",
    )

    table_slide(
        prs, "Triage agent: three tiers",
        ["Tier", "Rule", "Share", "Outcome"],
        [
            ["Auto-clear", "posterior < 8%", "44.0%", "FN rate 4.3% (vs. 15.3% base rate)"],
            ["Escalate", "8% ≤ posterior < 35%", "39.9%", "sent to human review"],
            ["Auto-flag", "posterior ≥ 35%", "16.2%", "41.5% precision (2.7x base rate)"],
        ],
        image="triage_tier_breakdown.png",
        notes="Headline number: 60.2% of cases could bypass manual review entirely. "
              "Auto-clear cuts the miss rate to under a third of the population base "
              "rate; auto-flag concentrates true positives at ~2.7x their natural frequency.",
    )

    bullet_slide(
        prs, "Does the uncertainty score actually mean something?",
        [
            "Mean entropy by tier: auto-clear 0.13 (confident) vs. auto-flag 0.97 / "
            "escalate 0.64 (uncertain)",
            "Negative risk can be called with much higher confidence than positive "
            "risk here — a direct consequence of the 85/15 imbalance",
            "Sensible bias for a screening tool: better to escalate a possible case "
            "than silently auto-clear or auto-diagnose it",
        ],
        notes='This slide answers the "why should I trust this triage split" '
              "question before showing the tradeoff curve.",
    )

    bullet_slide(
        prs, "Accuracy vs. automation tradeoff",
        [
            "Swept the escalation band width around the population prevalence, "
            "tracked automation rate against two error signals",
            "FN rate within auto-clear, and share of all diabetics missed by auto-clear",
            "Clear knee point around ~60–62% automation: FN rate stays flat at "
            "~2% below it, then climbs steeply (more than doubles by ~80% automation, "
            "reaches 26% at 100%)",
        ],
        image="triage_tradeoff.png",
        notes='This is the payoff slide -- "automating the easy 60% costs almost '
              'nothing; automating the rest would cost a lot." Pause here for '
              "questions if time allows.",
    )

    novelty_slide(
        prs, "Where's the novelty?",
        [
            ("Not the baseline comparison —",
             "the classical-vs-DNN benchmark is a faithful replication of "
             "Ayoade et al. (2025), useful as a validated starting point, not "
             "the contribution."),
            ("Cross-domain transfer —",
             "Akram et al. (2025) quantify Bayesian uncertainty for a CNN on "
             "retinopathy images; we adapt that pattern to tabular survey data "
             "with a classical base model instead."),
            ("Hybrid evidence fusion —",
             "the Bayesian network fuses Random Forest's opaque prediction with "
             "independent clinical variables (GenHlth, BMI, age, HighBP) as "
             "co-parents — a calibration layer over a black-box model, not "
             "a replacement classifier."),
            ("The agentic contribution —",
             "the triage agent turns that uncertainty score into an autonomous "
             "action policy (auto-clear / auto-flag / escalate) with a "
             "quantified accuracy-vs-automation tradeoff: it doesn't just "
             "predict, it decides when to act versus defer to a human."),
        ],
        notes="This slide directly answers the rubric's \"demonstrate some novelty\" "
              "requirement -- call out explicitly that novelty lives in the "
              "architecture/agentic layer, not the base algorithms. Lead with the "
              "triage agent as the strongest claim.",
    )

    bullet_slide(
        prs, "Conclusions",
        [
            "A well-tuned Random Forest matches/beats a feed-forward DNN on every "
            "practical axis on this tabular health-survey data — deep learning "
            "is not a silver bullet here",
            "A small Bayesian network turns RF's point estimate into a calibrated, "
            "per-case uncertainty score without retraining the base model",
            "That score powers a triage agent automating ~60% of the caseload at a "
            "quarter of the population's natural miss rate, concentrating review "
            "effort where it matters most",
        ],
        notes="Restate the three numbers people should remember: 0.819 AUC, 60.2% "
              "automation, 4.3% FN rate in auto-clear.",
    )

    bullet_slide(
        prs, "Future work",
        [
            "Richer Bayesian network structure (more evidence variables, learned "
            "rather than fixed structure)",
            "Calibration comparison against simpler baselines (raw RF thresholding, "
            "conformal prediction)",
            "Cost-sensitive tuning of the auto-clear/auto-flag bands against a real "
            "clinical review-capacity constraint",
        ],
    )

    bullet_slide(prs, "Questions", ["Thank you — happy to take questions."])

    DOCS_DIR.mkdir(exist_ok=True)
    prs.save(OUT_FILE)
    print(f"Saved {len(prs.slides)} slides to {OUT_FILE}")


if __name__ == "__main__":
    main()
