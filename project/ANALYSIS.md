# Diabetes Risk Prediction: Results & Analysis

**Group 10 — Jephthah Kimanthi David, Gideon Bamuleseyo**
15th-week milestone: end-to-end pipeline, Bayesian uncertainty layer, triage agent, and accuracy-vs-automation tradeoff.

## 1. Dataset

CDC BRFSS 2015 Diabetes Health Indicators dataset (Kaggle), 253,680 records, 21 features, binary
label. After de-duplication (24,206 exact-duplicate rows, ~9.5%), an 80/20 stratified split gives
183,579 train / 45,895 test rows. Class balance is imbalanced: 84.7% no-diabetes vs. 15.3%
diabetes/prediabetes in the training set. Strongest correlates with the label: `GenHlth`,
`HighBP`, `DiffWalk`, `BMI`, `HighChol` (positive); `Income`, `Education`, `PhysActivity`
(negative) — see `figures/feature_correlation.png`.

## 2. Baseline: classical ML vs. deep learning

Following Ayoade et al. (2025), we trained Logistic Regression, Random Forest, and Linear SVM
(class-weight balanced) alongside a feed-forward DNN (21→64→32→1, dropout 0.2, `BCEWithLogitsLoss`
with `pos_weight` for imbalance, 30 epochs on Apple MPS).

| Model | Accuracy | F1 | AUC | Train time |
|---|---|---|---|---|
| Logistic Regression | 0.714 | 0.449 | 0.811 | 0.1s |
| Random Forest | 0.723 | 0.458 | 0.817 | 3.9s |
| Linear SVM | 0.710 | 0.447 | 0.810 | 0.2s |
| DNN (feed-forward) | 0.704 | 0.452 | 0.819 | 53.4s |

*(`results/all_results.csv`, `figures/full_model_comparison.png`)*

**Finding:** this replicates Ayoade et al.'s core result. The DNN does not clearly beat classical
models on this tabular survey data — it edges out on AUC (0.819 vs. RF's 0.817) but has the
*lowest* accuracy and costs 15–500x more to train. Our DNN's F1 (0.452) lands just above their
reported 0.45, confirming our pipeline is a faithful baseline. Random Forest is the best
all-around classical model and the more practical choice to build on: comparable AUC to the DNN,
better F1/accuracy, and near-zero training/inference cost. **We selected Random Forest as the
base model for the uncertainty layer.**

## 3. Bayesian uncertainty layer

Rather than trusting Random Forest's raw probability output directly, we layer a discrete
Bayesian Network (pgmpy) over it, following the uncertainty-quantification approach in
Akram et al. (2025). The network combines five variables as parents of the diabetes label:

- **`pred_bin`** — RF's predicted probability, discretized into tertiles (low/mid/high)
- **`GenHlth`** — self-reported general health (1–5)
- **`BMI_bin`** — BMI category (normal/overweight/obese/severely obese)
- **`Age_bin`** — age group (18–39 / 40–64 / 65+)
- **`HighBP`** — high blood pressure (0/1)

CPDs are fit with a `BayesianEstimator` (BDeu prior, equivalent sample size 10) on the training
set. For each test case, `VariableElimination` gives an exact posterior P(diabetes=1 | evidence);
we also report its **binary entropy** as a per-case uncertainty score (entropy is maximal at
p=0.5 and falls to 0 as the posterior approaches 0 or 1).

This reframes RF's single point prediction as a distribution that also reflects a handful of
independent clinical risk factors, giving the triage agent a principled confidence signal instead
of relying on RF's often overconfident raw probabilities.

## 4. Triage agent

Test cases are routed into three tiers by their BN posterior P(diabetes=1), banded around the
training-set prevalence (15.3%):

| Tier | Rule | Share of cases | Outcome |
|---|---|---|---|
| **Auto-clear** | p < 8% | 44.0% | FN rate 4.3% (vs. 15.3% population base rate) |
| **Escalate** | 8% ≤ p < 35% | 39.9% | sent to human review |
| **Auto-flag** | p ≥ 35% | 16.2% | 41.5% precision (2.7x population base rate) |

*(`results/triage_tiers.csv`, `figures/triage_tier_breakdown.png`)*

**Finding:** **60.2%** of the test population (auto-clear + auto-flag) could bypass manual
review entirely. The auto-clear tier cuts the effective miss rate to under a third of the
population base rate (4.3% vs. 15.3%), and the auto-flag tier concentrates true positives at
nearly 3x their natural frequency — a meaningful triage signal for prioritizing follow-up.

The BN's entropy diagnostic corroborates the routing: mean entropy is low in auto-clear (0.13,
genuinely confident) but much higher in auto-flag (0.97) and escalate (0.64). This shows that,
in this population, *negative* risk can be called with much higher confidence than *positive*
risk — a direct consequence of the 85/15 class imbalance, and a sensible bias for a screening
tool (better to escalate a possible case than silently auto-clear or auto-diagnose it).

## 5. Accuracy-vs-automation tradeoff

We swept the escalation band's half-width around the prevalence point from 0 (maximum automation,
no escalation) to the full prevalence (nothing escalated is impossible — the band covers
everything), tracking automation rate against two error signals:

- **FN rate within auto-clear** — of the cases we auto-clear, what fraction are actually positive?
- **Share of all diabetics missed by auto-clear** — of every true positive in the test set, what
  fraction did auto-clear silently pass through?

*(`results/triage_tradeoff.csv`, `figures/triage_tradeoff.png`)*

**Finding:** there is a clear knee point around **~60–62% automation**. Below that, the
auto-clear false-negative rate stays essentially flat at ~2%. Push automation further and the
error curves — especially the share of all diabetics missed — start climbing steeply, more than
doubling by ~80% automation and reaching 26% by 100% automation (i.e., no escalation at all).
This is the practical argument for the triage design: **automating the "easy" ~60% of cases costs
almost nothing in missed diagnoses, but automating the remaining ~40% would cost a lot.**

## 6. Conclusion

1. On tabular BRFSS survey data, a well-tuned Random Forest matches or beats a feed-forward DNN
   on every practical axis (F1, accuracy, training cost) while nearly matching it on AUC —
   replicating Ayoade et al.'s (2025) finding that deep learning is not a silver bullet for this
   kind of structured health-survey data.
2. Layering a small Bayesian network over Random Forest's output turns a single point estimate
   into a calibrated, per-case uncertainty score, without needing to retrain or replace the base
   model.
3. That uncertainty score powers a triage agent that can safely automate ~60% of the caseload with
   a false-negative rate roughly a quarter of the population base rate, while concentrating
   review effort on the ambiguous middle 40% where errors would otherwise be costliest.

**Future work:** richer BN structure (more evidence variables, learned rather than fixed
structure); calibration comparison against simpler baselines (e.g., raw RF probability
thresholding, conformal prediction); cost-sensitive tuning of the auto-clear/auto-flag bands
against a real clinical review-capacity constraint.
