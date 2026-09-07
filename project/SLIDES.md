# Slide outline — 15th-week presentation

CS 582 Group 10 — Diabetes Risk Prediction: Deep Learning vs. Classical ML with a
Bayesian Uncertainty-Aware Triage Agent. Each slide below is titled, with bullet content,
the figure/table to show, and brief speaker notes. Content pulls directly from `ANALYSIS.md`
and the files in `results/` / `figures/` — swap in screenshots of those PNGs directly.

---

## 1. Title

- Diabetes Risk Prediction: Deep Learning vs. Classical ML with a Bayesian
  Uncertainty-Aware Triage Agent
- Jephthah Kimanthi David, Gideon Bamuleseyo — CS 582, Group 10

**Speaker notes:** Quick intro, one sentence on the hook — "we don't just pick a model,
we build a system that knows when to trust its own prediction."

---

## 2. The problem

- Diabetes screening at population scale needs two things at once: catch at-risk people
  early, don't overwhelm clinicians with false alarms
- Most work treats this as a single-model accuracy contest
- Question we ask instead: can a system know **when to trust** a prediction, not just
  **what** the prediction is?

**Speaker notes:** Set up the motivation before showing any numbers — this is the "why."

---

## 3. Dataset

- CDC BRFSS 2015 Diabetes Health Indicators (Kaggle), 253,680 records, 21 features
- Binary label: diabetes/prediabetes vs. none
- After de-dup (24,206 exact duplicates removed): 183,579 train / 45,895 test
  (80/20 stratified split)
- Class imbalance: 84.7% negative / 15.3% positive

**Visual:** `figures/class_balance.png`

**Speaker notes:** Flag the imbalance early — it explains several design decisions later
(class-weighting, why "auto-flag" needed care).

---

## 4. What predicts diabetes risk here?

- Strongest positive correlates: `GenHlth`, `HighBP`, `DiffWalk`, `BMI`, `HighChol`
- Strongest negative correlates: `Income`, `Education`, `PhysActivity`

**Visual:** `figures/feature_correlation.png` (and optionally `rate_by_genhlth.png` /
`rate_by_age.png` if time allows)

**Speaker notes:** One-slide EDA summary — don't dwell, this just grounds the feature
choices made later in the Bayesian network.

---

## 5. Pipeline overview

- Step 1: train classical ML + DNN, benchmark against a published comparison
  (Ayoade et al., 2025)
- Step 2: layer a Bayesian Network on the best model's output for per-case uncertainty
  (inspired by Akram et al., 2025)
- Step 3: triage agent — auto-clear / auto-flag / escalate
- Step 4: analyze the accuracy-vs-automation tradeoff

**Speaker notes:** This is the map for the rest of the talk — refer back to it as you move
through steps 1–4.

---

## 6. Baseline: classical ML vs. deep learning

| Model | Accuracy | F1 | AUC | Train time |
|---|---|---|---|---|
| Logistic Regression | 0.714 | 0.449 | 0.811 | 0.1s |
| Random Forest | 0.723 | 0.458 | 0.817 | 3.9s |
| Linear SVM | 0.710 | 0.447 | 0.810 | 0.2s |
| DNN (feed-forward) | 0.704 | 0.452 | 0.819 | 53.4s |

**Visual:** `figures/full_model_comparison.png`

**Speaker notes:** Our DNN F1 (0.452) lands just above Ayoade et al.'s reported 0.45 —
confirms the baseline is faithful, not an outlier.

---

## 7. Key finding: DNN isn't a silver bullet here

- DNN edges out on AUC (0.819 vs. RF's 0.817) but has the *lowest* accuracy
- Costs 15–500x more to train than any classical model
- Replicates Ayoade et al. (2025): deep learning doesn't reliably beat classical ML on
  tabular survey data
- **Decision: build the uncertainty layer on Random Forest** — comparable AUC, better
  F1/accuracy, near-zero cost

**Speaker notes:** This is a deliberate pivot point in the talk — "so we picked the
classical model, and here's why that's the interesting part, not a letdown."

---

## 8. Adding a Bayesian uncertainty layer

- Random Forest gives one probability per case — often overconfident, no structured
  notion of "how sure are we, really?"
- We fit a discrete Bayesian Network (pgmpy) with 5 parents of the diabetes label:
  RF's prediction (binned), `GenHlth`, `BMI` category, age group, `HighBP`
- `BayesianEstimator` (BDeu prior) fits the CPDs; `VariableElimination` gives an exact
  posterior P(diabetes=1 | evidence) per test case
- Binary entropy of that posterior = per-case uncertainty score

**Speaker notes:** Emphasize this doesn't retrain or replace RF — it's a lightweight layer
on top, inspired by the retinopathy uncertainty-quantification approach in Akram et al.
(2025), adapted from images to tabular data.

---

## 9. Triage agent: three tiers

| Tier | Rule | Share | Outcome |
|---|---|---|---|
| Auto-clear | posterior < 8% | 44.0% | FN rate 4.3% (vs. 15.3% base rate) |
| Escalate | 8% ≤ posterior < 35% | 39.9% | sent to human review |
| Auto-flag | posterior ≥ 35% | 16.2% | 41.5% precision (2.7x base rate) |

**Visual:** `figures/triage_tier_breakdown.png`

**Speaker notes:** Headline number: **60.2% of cases could bypass manual review
entirely.** Auto-clear cuts the miss rate to under a third of the population base rate;
auto-flag concentrates true positives at ~2.7x their natural frequency.

---

## 10. Does the uncertainty score actually mean something?

- Mean entropy by tier: auto-clear 0.13 (confident) vs. auto-flag 0.97 / escalate 0.64
  (uncertain)
- Negative risk can be called with much higher confidence than positive risk here —
  a direct consequence of the 85/15 imbalance
- Sensible bias for a screening tool: better to escalate a possible case than
  silently auto-clear or auto-diagnose it

**Speaker notes:** This slide answers the "why should I trust this triage split"
question before showing the tradeoff curve.

---

## 11. Accuracy vs. automation tradeoff

- Swept the escalation band width around the population prevalence, tracked automation
  rate against two error signals: FN rate within auto-clear, and share of all diabetics
  missed by auto-clear
- Clear knee point around **~60–62% automation**: FN rate stays flat at ~2% below it,
  then climbs steeply (more than doubles by ~80% automation, reaches 26% at 100%)

**Visual:** `figures/triage_tradeoff.png`

**Speaker notes:** This is the payoff slide — "automating the easy 60% costs almost
nothing; automating the rest would cost a lot." Pause here for questions if time allows.

---

## 12. Where's the novelty?

- **Not** the classical-vs-DNN comparison — that's a faithful replication of
  Ayoade et al. (2025), useful as a validated baseline, not the contribution
- **Cross-domain transfer:** Akram et al. (2025) quantify Bayesian uncertainty for a
  CNN on retinopathy *images* — we adapt that pattern to tabular survey data with a
  classical base model instead
- **Hybrid evidence fusion:** the Bayesian network doesn't just re-derive uncertainty
  from RF alone — it fuses RF's opaque prediction with independent clinical variables
  (GenHlth, BMI, age, HighBP) as co-parents, using the BN as a calibration layer over
  a black-box model, not as the classifier itself
- **The agentic contribution:** the triage agent turns that uncertainty score into an
  autonomous action policy (auto-clear / auto-flag / escalate) with a quantified
  accuracy-vs-automation tradeoff — it doesn't just predict, it decides when to act
  versus defer to a human

**Speaker notes:** This slide directly answers the rubric's "demonstrate some novelty"
requirement — call out explicitly that novelty lives in the *architecture/agentic*
layer, not the base algorithms. Lead with the triage agent as the strongest claim.

---

## 13. Conclusions

1. A well-tuned Random Forest matches/beats a feed-forward DNN on every practical axis
   on this tabular health-survey data — deep learning is not a silver bullet here
2. A small Bayesian network turns RF's point estimate into a calibrated, per-case
   uncertainty score without retraining the base model
3. That score powers a triage agent automating ~60% of the caseload at a quarter of the
   population's natural miss rate, concentrating review effort where it matters most

**Speaker notes:** Restate the three numbers people should remember: **0.819 AUC**,
**60.2% automation**, **4.3% FN rate in auto-clear**.

---

## 14. Future work

- Richer Bayesian network structure (more evidence variables, learned rather than
  fixed structure)
- Calibration comparison against simpler baselines (raw RF thresholding, conformal
  prediction)
- Cost-sensitive tuning of the auto-clear/auto-flag bands against a real clinical
  review-capacity constraint

---

## 15. Questions

- Thank the audience, open the floor
