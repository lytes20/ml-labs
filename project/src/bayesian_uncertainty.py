"""Layer a Bayesian Network on top of the best classical model (Random Forest)
to estimate per-case prediction uncertainty, following the approach in
Akram et al. (2025): combine the model's own confidence with a handful of
clinically meaningful risk factors in a small discrete Bayesian network, then
read off the posterior entropy as an uncertainty score."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from preprocess import load_raw_and_scaled_split

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

BASE_MODEL_FILE = MODELS_DIR / "random_forest.joblib"

AGE_BIN_EDGES = [0, 4, 9, 13]
AGE_BIN_LABELS = ["18-39", "40-64", "65+"]
BMI_BIN_EDGES = [0, 25, 30, 40, 100]
BMI_BIN_LABELS = ["normal", "overweight", "obese", "severely_obese"]

EVIDENCE_COLS = ["pred_bin", "GenHlth", "BMI_bin", "Age_bin", "HighBP"]


def discretize(X_raw, pred_prob, pred_bin_edges):
    df = pd.DataFrame(index=X_raw.index)
    df["pred_bin"] = pd.cut(pred_prob, bins=pred_bin_edges, labels=["low", "mid", "high"])
    df["GenHlth"] = X_raw["GenHlth"].astype(int).astype(str)
    df["BMI_bin"] = pd.cut(X_raw["BMI"], bins=BMI_BIN_EDGES, labels=BMI_BIN_LABELS)
    df["Age_bin"] = pd.cut(X_raw["Age"], bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS)
    df["HighBP"] = X_raw["HighBP"].astype(int).astype(str)
    return df


def binary_entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def query_unique_combos(infer, evidence_df):
    """Run VariableElimination once per unique evidence combination instead
    of once per row -- the discretized evidence has far fewer distinct
    combinations than there are test rows."""
    unique_combos = evidence_df.drop_duplicates()
    posteriors = {}
    for _, row in unique_combos.iterrows():
        evidence = {col: row[col] for col in EVIDENCE_COLS}
        key = tuple(evidence.values())
        result = infer.query(["label"], evidence=evidence, show_progress=False)
        p1 = result.values[list(result.state_names["label"]).index(1)]
        posteriors[key] = p1

    keys = list(zip(*(evidence_df[col] for col in EVIDENCE_COLS)))
    return np.array([posteriors[k] for k in keys])


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    rf_model = joblib.load(BASE_MODEL_FILE)
    X_train_raw, X_test_raw, X_train_scaled, X_test_scaled, y_train, y_test, _ = (
        load_raw_and_scaled_split()
    )

    train_prob = rf_model.predict_proba(X_train_scaled)[:, 1]
    test_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

    pred_bin_edges = [-0.01] + list(np.quantile(train_prob, [1 / 3, 2 / 3])) + [1.01]

    train_evidence = discretize(X_train_raw, train_prob, pred_bin_edges)
    train_evidence["label"] = y_train.values
    test_evidence = discretize(X_test_raw, test_prob, pred_bin_edges)

    print(f"Training Bayesian network on {len(train_evidence)} rows...")
    bn = DiscreteBayesianNetwork(
        [(col, "label") for col in EVIDENCE_COLS]
    )
    estimator = BayesianEstimator(bn, train_evidence)
    cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=10)
    bn.add_cpds(*cpds)

    infer = VariableElimination(bn)
    print(f"Running inference over unique evidence combinations "
          f"(test set has {len(test_evidence)} rows)...")
    posterior_p1 = query_unique_combos(infer, test_evidence)
    entropy = binary_entropy(posterior_p1)

    results_df = pd.DataFrame({
        "y_true": y_test.values,
        "rf_prob": test_prob,
        "bn_posterior_p1": posterior_p1,
        "bn_entropy": entropy,
    }, index=X_test_raw.index)
    results_df = pd.concat([results_df, test_evidence.reset_index(drop=True).set_axis(X_test_raw.index)], axis=1)
    results_df.to_csv(RESULTS_DIR / "bayesian_uncertainty.csv")

    print("\nPosterior P(diabetes=1) summary:")
    print(results_df["bn_posterior_p1"].describe().round(4).to_string())
    print("\nEntropy summary:")
    print(results_df["bn_entropy"].describe().round(4).to_string())
    print(f"\nSaved per-case uncertainty to {RESULTS_DIR / 'bayesian_uncertainty.csv'}")


if __name__ == "__main__":
    main()
