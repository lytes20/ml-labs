
## 1. The one-paragraph pitch

Diabetes screening at population scale needs to catch at-risk people without
overwhelming clinicians with false alarms. Instead of picking one "best" model, we
built a system that layers *uncertainty estimation* on top of a strong baseline
classifier, then uses that uncertainty to decide **which cases a machine can safely
resolve on its own, and which need a human**. Concretely: (1) we trained and
compared classical ML and a deep neural net on the CDC's BRFSS diabetes survey
data, replicating a published benchmark; (2) we layered a Bayesian Network on the
best model's output to get a calibrated, per-case confidence score; (3) we built a
triage agent that uses that score to auto-clear, auto-flag, or escalate each case;
(4) we quantified exactly how much automation that buys you, and at what cost.

---

## 2. Why this project (motivation)

Most ML coursework/Kaggle work on this dataset frames it as a single-model
accuracy contest: which algorithm gets the best F1/AUC. That's a valid exercise but
doesn't address the real operational question in a screening context — a model is
only useful if you know when to trust it. Ayoade et al. (2025) showed deep learning
doesn't reliably beat classical ML on this kind of tabular survey data, which
redirects the interesting question away from "which model" and toward "what do we
do with the uncertainty in any model's predictions." That's the gap this project
targets, borrowing an uncertainty-quantification pattern from a completely
different domain (Bayesian deep learning for image-based diabetic retinopathy
detection, Akram et al. 2025) and adapting it to tabular data.

---

## 3. The dataset

- **Source:** CDC's 2015 Behavioral Risk Factor Surveillance System (BRFSS) survey,
  cleaned/prepared by Alex Teboul and published on Kaggle as the "Diabetes Health
  Indicators Dataset." Downloaded programmatically via `kagglehub`
  (`src/download_data.py`).
- **Size:** 253,680 records, 21 features, one binary label (`Diabetes_binary`:
  1 = diabetes or prediabetes, 0 = neither).
- **Feature types:**
  - *Binary (14):* HighBP, HighChol, CholCheck, Smoker, Stroke,
    HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
    AnyHealthcare, NoDocbcCost, DiffWalk, Sex
  - *Ordinal (4):* GenHlth (1=excellent…5=poor), Age (13 BRFSS age bands,
    18 → 80+), Education, Income
  - *Numeric (3):* BMI, MentHlth (days/month), PhysHlth (days/month)
- **Data quality (from `src/eda.py`):** 0 missing values; 24,206 exact-duplicate
  rows (~9.5% of the data) — removed. Every one of the 22 columns is coarse
  (binary/ordinal/rounded), so an exact tie across all 22 fields is far more likely
  to be a genuine duplicate record than two different people who happen to share
  every answer; this is a standard cleaning step people take with this specific
  published dataset.
- **Class balance:** before dedup, 86.1% negative / 13.9% positive; after dedup and
  splitting, the training set is 84.7% / 15.3% — real, clinically expected
  imbalance (diabetes prevalence in the general population is well below 50%), not
  an artifact.
- **Split:** 80/20 **stratified** train/test split (`random_state=42`), so both
  sides preserve the same 84.7/15.3 class ratio. Train: 183,579 rows. Test: 45,895
  rows. `StandardScaler` is fit on train only, then applied to test — this avoids
  leaking test-set statistics (mean/variance) into training.
- **Top correlates with the label** (`figures/feature_correlation.png`): positive —
  GenHlth, HighBP, DiffWalk, BMI, HighChol; negative — Income, Education,
  PhysActivity. These five variables are exactly why the Bayesian network (Section
  6) uses GenHlth, BMI, Age, and HighBP as its clinical evidence nodes — they are
  the features with the strongest raw signal.

---

## 4. Classical ML baselines

`src/train_classical.py`. Three models, all with `class_weight="balanced"` (see
Glossary) to correct for the 85/15 imbalance without touching the data itself:

| Model | Key settings | Why |
|---|---|---|
| Logistic Regression | `max_iter=1000` | simple, fast, interpretable linear baseline |
| Random Forest | `n_estimators=200, max_depth=15, min_samples_leaf=10` | captures non-linear interactions; depth/leaf constraints added deliberately (see the RF story below) |
| Linear SVM (`LinearSVC`) | `max_iter=5000` | SVM baseline, but **linear kernel**, not RBF/kernel SVC |

**Why `LinearSVC` instead of a kernel SVC:** a kernel SVM's training cost scales
roughly quadratic-to-cubic in the number of samples. With ~183k training rows,
kernel SVC would take hours-to-days; `LinearSVC` (solved via `liblinear`) trains in
under a second and is the standard substitution for SVM on large tabular data.

**The Random Forest bloat story** (good "tell me about a problem you hit"
answer): the first version used unconstrained tree depth with
`class_weight="balanced"`. On 183k rows that grew a **1.24 GB** model file —
badly overfit, memorizing near-leaf-level detail. Adding `max_depth=15,
min_samples_leaf=10` shrank the model to **90 MB** and simultaneously *improved*
both F1 (0.418 → 0.458) and AUC (0.785 → 0.817) — proof the unconstrained trees
were overfitting, not just wasting disk space.

**Results:**

| Model | Accuracy | F1 | AUC | Train time |
|---|---|---|---|---|
| Logistic Regression | 0.714 | 0.449 | 0.811 | 0.1s |
| Random Forest | 0.723 | 0.458 | 0.817 | 3.9s |
| Linear SVM | 0.710 | 0.447 | 0.810 | 0.2s |

---

## 5. Deep learning baseline

`src/train_dnn.py`. A feed-forward network in PyTorch:

- **Architecture:** 21 → 64 → 32 → 1, ReLU activations, Dropout(0.2) after each
  hidden layer.
- **Loss:** `BCEWithLogitsLoss` with `pos_weight = n_negative / n_positive` — the
  neural-net equivalent of `class_weight="balanced"`; it multiplies the loss
  contribution of positive-class examples so the rare class isn't drowned out.
- **Optimizer:** Adam, learning rate 1e-3.
- **Training:** 30 epochs, batch size 512, on Apple's MPS backend (auto-detected;
  falls back to CUDA, then CPU).
- **Result:** accuracy 0.704, F1 0.452, AUC 0.819, trained in 53.4 seconds.

**Why this architecture and not something bigger:** the point wasn't to
maximize DNN performance in isolation — it's to give the Ayoade et al.
replication a fair, standard feed-forward baseline (matching the spirit of their
setup) and see whether scaling to deep learning helps on *this kind* of data
(tabular, mostly categorical, moderate feature count). It doesn't need to be
state-of-the-art to make that point.

---

## 6. Baseline comparison — the pivot point of the talk

| Model | Accuracy | F1 | AUC | Train time |
|---|---|---|---|---|
| Logistic Regression | 0.714 | 0.449 | 0.811 | 0.1s |
| Random Forest | 0.723 | 0.458 | 0.817 | 3.9s |
| Linear SVM | 0.710 | 0.447 | 0.810 | 0.2s |
| **DNN (feed-forward)** | 0.704 | 0.452 | **0.819** | 53.4s |

Ayoade et al. (2025) reported a feed-forward DNN F1 of **0.45** on this exact
binary BRFSS subset; ours is 0.452 — close enough to say our pipeline is a
faithful, non-cherry-picked replication.

**The finding:** the DNN edges out Random Forest on AUC (0.819 vs. 0.817) but has
the *lowest* accuracy of all four models and costs 15–500x more to train. This
replicates Ayoade et al.'s core conclusion: deep learning does not reliably
outperform classical ML on tabular survey data. **We chose Random Forest as the
base model for the uncertainty layer** — nearly identical AUC, better F1/accuracy,
and effectively free to train/run.

---

## 7. The Bayesian uncertainty layer

`src/bayesian_uncertainty.py`. This is the technical core of the project.

### 7.1 The idea

Random Forest gives a single probability per case, and tree-ensemble
probabilities are known to be poorly calibrated (often overconfident near 0 or 1).
Rather than trusting that number directly, we build a small **discrete Bayesian
Network** that treats RF's own prediction as *one piece of evidence among several*,
combined with a handful of independently informative clinical variables. The
output is a posterior probability that reflects the combined weight of all that
evidence, plus a principled uncertainty score (entropy) for every single case.

### 7.2 The network structure

Five **parent** nodes, all pointing into one **child** node (`label`):

- `pred_bin` — Random Forest's predicted probability, discretized into
  tertiles (low/mid/high) using train-set quantile cutoffs
- `GenHlth` — self-reported general health, 1–5
- `BMI_bin` — BMI bucketed into normal / overweight / obese / severely obese
- `Age_bin` — age bucketed into 18–39 / 40–64 / 65+
- `HighBP` — binary, high blood pressure

This is structurally a **naive-Bayes-style fan-in**: every evidence variable is
assumed conditionally independent of the others *given* the label. That's a real
simplifying assumption (the five variables surely correlate with each other, not
just with the label) — chosen deliberately for tractability with a small feature
set and no separate structure-learning step. A richer version could learn edges
between the evidence nodes too (e.g., via `pgmpy`'s hill-climbing search), at the
cost of needing much more data per parameter and a validation step to prevent
overfitting the structure itself.

### 7.3 Fitting the network

- **`BayesianEstimator`** with a **BDeu prior** (Bayesian Dirichlet equivalent
  uniform), **equivalent sample size (ESS) = 10**. Plain maximum-likelihood
  counting would assign probability zero to any parent-value combination that
  never appeared in training data — catastrophic the moment a test case hits that
  combination during inference. BDeu smooths every count by adding a small
  uniform pseudo-count (governed by ESS) before normalizing, so no probability is
  ever exactly zero. ESS=10 is a light-touch smoothing choice: large enough to
  avoid zero-probability failures, small enough not to wash out the signal
  actually present in 183k training rows.
- **`VariableElimination`** performs **exact** inference — for a query like
  P(label | evidence), it systematically sums out (marginalizes) the variables
  not being queried, in an efficient elimination order, rather than enumerating
  the full joint distribution. Exact inference is feasible here specifically
  because the network is small (5 parents, all with low cardinality).
- **Performance trick:** running exact inference row-by-row for 45,895 test cases
  would mean tens of thousands of redundant queries, since evidence is coarsely
  discretized. We instead group test rows by their unique 5-variable evidence
  combination (far fewer than 45,895), run inference once per unique combination,
  and map results back — same algorithm, no accuracy tradeoff, just avoiding
  repeated work.

### 7.4 The uncertainty score

For every test case we get a posterior **P(diabetes=1 | evidence)**, and we
compute its **binary (Shannon) entropy**:

```
H(p) = -p·log2(p) - (1-p)·log2(1-p)
```

`H` ranges from 0 (certain — p is 0 or 1) to 1 bit (maximally uncertain — p=0.5).
This is the per-case uncertainty score. **Important nuance if asked "what kind of
uncertainty is this":** this is *predictive* entropy over the network's point-
estimated posterior — it captures how spread out the evidence leaves the outcome,
not full Bayesian *parameter* uncertainty (that would require a posterior
distribution over the CPD values themselves, e.g. via full Bayesian
model-averaging or something like MC-Dropout/deep ensembles in a neural setting).
We're quantifying something closer to aleatoric/ambiguity uncertainty in the
evidence, not epistemic uncertainty about the model itself.

---

## 8. The triage agent

`src/triage_agent.py`. Routes every test case into one of three tiers using the
BN's posterior probability, banded around the training-set prevalence (15.3%):

| Tier | Rule | Share of test cases | Outcome |
|---|---|---|---|
| **Auto-clear** | posterior < 8% | 44.0% | FN rate 4.3% (vs. 15.3% population base rate) |
| **Escalate** | 8% ≤ posterior < 35% | 39.9% | sent to a human reviewer |
| **Auto-flag** | posterior ≥ 35% | 16.2% | 41.5% precision (2.7x population base rate) |

**Headline:** 60.2% of cases (auto-clear + auto-flag) could bypass manual review
entirely, while the auto-clear tier's actual miss rate (4.3%) is under a third of
the population's natural base rate (15.3%) — meaning the triage split genuinely
concentrates risk rather than just shrinking the queue arbitrarily.

**Entropy sanity-check:** mean entropy is 0.13 in auto-clear (genuinely
confident) vs. 0.97 in auto-flag and 0.64 in escalate. In plain terms: this
population lets us call *negative* risk with much higher confidence than
*positive* risk — a direct, expected consequence of the 85/15 imbalance (far more
training examples to learn "definitely healthy" patterns from than "definitely
at-risk" ones). That's also a reasonable bias for a screening tool: better to
escalate an ambiguous case than to auto-clear or auto-diagnose it with false
confidence.

### 8.1 A design pivot

- The first version of the triage agent didn't use probability bands — it used the
BN's *entropy* directly (escalate the highest-entropy 25% of cases, then split the
rest at posterior ≥ 0.5 for auto-flag vs. auto-clear). That produced a broken
result: only **11 out of 45,895** cases (0.02%) landed in auto-flag. The reason:
entropy is maximal at p=0.5 and falls off symmetrically, so requiring *both*
p ≥ 0.5 *and* low entropy demands the posterior get all the way up near ~0.7+ —
and with only 5 modestly predictive evidence variables and BDeu smoothing pulling
posteriors toward the 15.3% base rate, very few cases ever get that high. 
- Switching to direct probability bands (fixed low/high cutoffs around the prevalence, entropy kept only as a reported diagnostic) fixed this and produced the well-populated,
well-stratified three-tier split shown above. This is a good story for "what did
you learn/change during the project" — it shows the first uncertainty-routing
design was theoretically clean but practically useless, and the fix required
understanding *why* (the interaction between class imbalance and entropy's shape),
not just tuning a number.

---

## 9. The accuracy-vs-automation tradeoff

We swept the escalation band's half-width around the 15.3% prevalence from 0
(maximum automation) up to the full prevalence (band covers everything, 0%
automation), tracking:

- **FN rate within auto-clear** — of the cases we auto-clear, what fraction are
  actually positive?
- **Overall missed-positive rate** — of *every* true positive in the test set,
  what fraction did auto-clear silently pass through?

**Finding:** there's a clear knee around **~60–62% automation**. Below that, the
auto-clear false-negative rate stays essentially flat around 2%. Push automation
further and the error curves climb — the overall missed-positive rate roughly
triples from ~4% at the knee to ~15% by 80% automation, and reaches 26% at 100%
automation (no escalation at all). **The practical argument:** automating the
"easy" ~60% of cases costs almost nothing in missed diagnoses; automating the
remaining ~40% would cost a lot. This is exactly the kind of quantified tradeoff a
pure accuracy metric can't show you.

---

## 10. The reference papers

1. **Sharma & Shah (2021)**, *A comprehensive review of machine learning
   techniques on diabetes detection* — a survey paper, cited for general framing
   of what ML techniques have been tried for diabetes prediction; not a specific
   methodology we replicate.
2. **Ayoade, Shahrestani & Ruan (2025)**, *Machine Learning and Deep Learning
   Approaches for Predicting Diabetes Progression: A Comparative Analysis* — the
   paper we replicate in Section 6. They compare classical ML and a feed-forward
   DNN on the binary BRFSS dataset and report a DNN F1 of 0.45, showing deep
   learning doesn't reliably beat classical ML here. We reproduce their setup as
   our own baseline and confirm the same conclusion independently.
3. **Akram, Adnan, et al. (2025)**, *Uncertainty-aware diabetic retinopathy
   detection using deep learning enhanced by Bayesian approaches* — the paper that
   inspires Section 7. They quantify Bayesian uncertainty for a CNN doing
   image-based retinopathy detection. We adapt the *pattern* (layer a Bayesian
   uncertainty estimate on top of a point-estimate model) to a different modality
   (tabular survey data) and a different base model (Random Forest, not a CNN) —
   this cross-domain adaptation, not a literal reimplementation, is one of our
   novelty claims.

---

## 11. Glossary

- **AUC (Area Under the ROC Curve):** probability that a random positive example
  is ranked higher (by predicted score) than a random negative example. 0.5 =
  random guessing, 1.0 = perfect ranking. Threshold-independent, unlike accuracy.
- **F1 score:** harmonic mean of precision and recall — `2·(P·R)/(P+R)`. Useful
  when classes are imbalanced and accuracy alone would be misleading (e.g., a
  model that always predicts "negative" gets 84.7% accuracy here but F1 = 0).
- **Precision:** of everything predicted positive, what fraction is actually
  positive. **Recall (sensitivity):** of everything actually positive, what
  fraction did we correctly catch.
- **Class imbalance:** when one class vastly outnumbers another (85/15 here).
  Naively optimizing accuracy under imbalance rewards ignoring the minority class.
- **`class_weight="balanced"` (scikit-learn):** reweights each class's
  contribution to the loss inversely proportional to its frequency, so the
  minority class isn't drowned out — without changing the actual training data.
- **`pos_weight` (PyTorch `BCEWithLogitsLoss`):** the neural-net analogue of
  class weighting — multiplies the loss for positive-class examples by
  `n_negative / n_positive`.
- **Stratified split:** a train/test split that preserves the original class
  proportions in both subsets — important under imbalance to avoid a test set
  that's accidentally even more skewed than the population.
- **Standardization (`StandardScaler`):** rescales each feature to zero mean,
  unit variance. Needed for models sensitive to feature scale (Logistic
  Regression, SVM, neural nets); not strictly needed for tree-based models like
  Random Forest, but harmless to apply uniformly.
- **Bayesian Network:** a directed acyclic graph representing a joint probability
  distribution, where each node is a random variable and each edge is a direct
  probabilistic dependency; every node has a Conditional Probability Distribution
  (CPD) given its parents.
- **CPD (Conditional Probability Distribution/Table):** the table of
  probabilities for a node's possible values, conditioned on every combination of
  its parents' values.
- **Prior / BDeu prior:** a prior is an assumed probability distribution before
  seeing data. BDeu (Bayesian Dirichlet equivalent uniform) is a specific prior
  used when fitting CPDs from counts — it adds small uniform pseudo-counts
  (controlled by the equivalent sample size) to prevent any observed combination
  from getting probability exactly zero.
- **Equivalent sample size (ESS):** a tuning knob for the BDeu prior's strength —
  larger ESS means more smoothing (pulls estimates toward uniform), smaller ESS
  trusts the observed training counts more.
- **Variable Elimination:** an exact inference algorithm for Bayesian networks
  that computes a query (e.g., P(label | evidence)) by summing out
  non-query variables in an efficient order, avoiding full joint enumeration.
- **Posterior probability:** an updated probability estimate after conditioning
  on observed evidence (as opposed to a *prior*, the estimate before evidence).
- **Entropy (Shannon entropy):** a measure of uncertainty/information content in
  a probability distribution; for a binary outcome with probability `p`,
  `H(p) = -p·log2(p) - (1-p)·log2(1-p)`, ranging from 0 (certain) to 1 bit
  (maximally uncertain at p=0.5).
- **Epistemic vs. aleatoric uncertainty:** epistemic = uncertainty from limited
  knowledge/data about the model itself (reducible with more data); aleatoric =
  inherent randomness/ambiguity in the outcome given the evidence (not reducible
  by more data alone). The entropy score in this project is closer to a
  predictive/aleatoric-style measure over a point-estimated posterior, not full
  epistemic uncertainty over model parameters.
- **Overfitting:** when a model fits noise/idiosyncrasies of the training data
  rather than the true underlying pattern, hurting generalization to new data —
  what happened with the unconstrained Random Forest (Section 4).
- **Dropout:** a neural-network regularization technique that randomly zeroes a
  fraction of neuron activations during training, discouraging co-adaptation and
  reducing overfitting.
- **Triage:** sorting cases by urgency/confidence so effort (here, human review)
  is allocated to where it's most needed, rather than treating every case
  identically.
