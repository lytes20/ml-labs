"""Live-demo script: walk through a handful of individual test cases end to
end -- raw features -> RF probability -> BN posterior -> entropy -> triage
decision -- instead of just showing aggregate tables.
"""

from pathlib import Path

import pandas as pd

from preprocess import load_raw_and_scaled_split
from triage_agent import CLEAR_THRESHOLD, FLAG_THRESHOLD, assign_tiers

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RAW_FEATURES_TO_SHOW = [
    "GenHlth",
    "BMI",
    "Age",
    "HighBP",
    "HighChol",
    "DiffWalk",
    "PhysActivity",
]

TIER_LABELS = {
    "auto-clear": "AUTO-CLEAR (low risk, no human review)",
    "auto-flag": "AUTO-FLAG (high risk, flagged for action)",
    "escalate": "ESCALATE (ambiguous, sent to human reviewer)",
}


def pick_example(df, tier_name):
    subset = df[df["tier"] == tier_name]
    if tier_name == "escalate":
        target = (CLEAR_THRESHOLD + FLAG_THRESHOLD) / 2
    elif tier_name == "auto-clear":
        target = 0.0
    else:
        target = 1.0
    idx = (subset["bn_posterior_p1"] - target).abs().idxmin()
    return idx


def print_case(case_id, row, raw_row):
    print(f"\n{'=' * 60}")
    print(f"Test case id: {case_id}")
    print(f"{'=' * 60}")
    print("Raw features:")
    for col in RAW_FEATURES_TO_SHOW:
        print(f"  {col:<14} {raw_row[col]}")
    print(f"\n  Random Forest probability  : {row['rf_prob']:.3f}")
    print(f"  BN posterior P(diabetes=1) : {row['bn_posterior_p1']:.3f}")
    print(f"  BN entropy (uncertainty)   : {row['bn_entropy']:.3f}")
    print(f"\n  >>> Triage decision: {TIER_LABELS[row['tier']]}")
    print(f"  (actual label: {'diabetic' if row['y_true'] == 1 else 'not diabetic'})")


def main():
    df = pd.read_csv(RESULTS_DIR / "bayesian_uncertainty.csv", index_col=0)
    df["tier"] = assign_tiers(df, CLEAR_THRESHOLD, FLAG_THRESHOLD)

    _, X_test_raw, _, _, _, _, _ = load_raw_and_scaled_split()

    print("Bayesian uncertainty-aware triage agent -- live case walkthrough")
    print(
        f"(clear threshold p < {CLEAR_THRESHOLD:.0%}, flag threshold p >= {FLAG_THRESHOLD:.0%})"
    )

    for tier_name in ["auto-clear", "escalate", "auto-flag"]:
        case_id = pick_example(df, tier_name)
        print_case(case_id, df.loc[case_id], X_test_raw.loc[case_id])

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
