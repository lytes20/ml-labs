"""Bayesian uncertainty-aware triage agent.

Routes each test case using the Bayesian network's posterior P(diabetes=1)
(bayesian_uncertainty.py) into three tiers:
  - auto-clear:  posterior well below the training-set prevalence -- low risk
  - auto-flag:   posterior well above the training-set prevalence -- high risk
  - escalate:    posterior near the prevalence -- too ambiguous to act on
                 automatically, send to a human reviewer

The BN's entropy is reported per case as a diagnostic (it is highest for
exactly the cases the probability band also routes to escalation), and the
band half-width is swept to trace the accuracy-vs-automation tradeoff.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

TRAIN_PREVALENCE = 0.153  # post-dedup training-set class balance (preprocess.py)
CLEAR_THRESHOLD = 0.08
FLAG_THRESHOLD = 0.35

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
MUTED = "#898781"


def assign_tiers(df, clear_threshold, flag_threshold):
    p = df["bn_posterior_p1"]
    tier = np.where(p < clear_threshold, "auto-clear", np.where(p >= flag_threshold, "auto-flag", "escalate"))
    return pd.Series(tier, index=df.index, name="tier")


def tier_breakdown(df):
    rows = []
    total = len(df)
    for tier_name in ["auto-clear", "auto-flag", "escalate"]:
        subset = df[df["tier"] == tier_name]
        n = len(subset)
        if n == 0:
            continue
        row = {
            "tier": tier_name,
            "count": n,
            "pct_of_total": n / total,
            "mean_bn_entropy": subset["bn_entropy"].mean(),
        }
        if tier_name == "auto-clear":
            row["false_negative_rate"] = (subset["y_true"] == 1).mean()
        elif tier_name == "auto-flag":
            row["false_positive_rate"] = (subset["y_true"] == 0).mean()
        else:
            row["actual_positive_rate"] = (subset["y_true"] == 1).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("tier")


def sweep_tradeoff(df, half_widths, prevalence):
    records = []
    n_total_pos = (df["y_true"] == 1).sum()
    for w in half_widths:
        clear_threshold = max(0.0, prevalence - w)
        flag_threshold = min(1.0, prevalence + w)
        tier = assign_tiers(df, clear_threshold, flag_threshold)
        auto_clear = df[tier == "auto-clear"]
        escalated = df[tier == "escalate"]

        automation_rate = (tier != "escalate").mean()
        missed_in_auto_clear = (auto_clear["y_true"] == 1).sum()
        fn_rate_auto_clear = missed_in_auto_clear / len(auto_clear) if len(auto_clear) else 0.0
        overall_missed_positive_rate = missed_in_auto_clear / n_total_pos if n_total_pos else 0.0
        pct_positives_escalated = (
            (escalated["y_true"] == 1).sum() / n_total_pos if n_total_pos else 0.0
        )

        records.append({
            "band_half_width": w,
            "automation_rate": automation_rate,
            "fn_rate_within_auto_clear": fn_rate_auto_clear,
            "overall_missed_positive_rate": overall_missed_positive_rate,
            "pct_positives_caught_by_escalation": pct_positives_escalated,
        })
    return pd.DataFrame(records)


def plot_tier_breakdown(breakdown_df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"auto-clear": AQUA, "auto-flag": ORANGE, "escalate": MUTED}
    bars = ax.bar(
        breakdown_df.index, breakdown_df["pct_of_total"],
        color=[colors[t] for t in breakdown_df.index],
    )
    for bar, tier in zip(bars, breakdown_df.index):
        pct = breakdown_df.loc[tier, "pct_of_total"]
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.015, f"{pct:.1%}",
                ha="center", color=INK, fontsize=10)

    ax.set_ylim(0, breakdown_df["pct_of_total"].max() * 1.25)
    ax.set_ylabel("Share of test cases")
    ax.set_title(
        f"Triage tier breakdown (clear < {CLEAR_THRESHOLD:.0%}, flag >= {FLAG_THRESHOLD:.0%})",
        color=INK, loc="left", pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "triage_tier_breakdown.png", dpi=150)
    plt.close(fig)


def plot_tradeoff(tradeoff_df, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    order = tradeoff_df.sort_values("automation_rate")
    ax.plot(
        order["automation_rate"], order["fn_rate_within_auto_clear"],
        color=BLUE, marker="o", markersize=4, linewidth=2,
        label="FN rate within auto-clear tier",
    )
    ax.plot(
        order["automation_rate"], order["overall_missed_positive_rate"],
        color=RED, marker="o", markersize=4, linewidth=2,
        label="Share of all diabetics missed by auto-clear",
    )
    ax.set_xlabel("Automation rate (share of cases not escalated)")
    ax.set_ylabel("Error rate")
    ax.set_title("Accuracy vs. automation tradeoff", color=INK, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "triage_tradeoff.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "bayesian_uncertainty.csv", index_col=0)

    df["tier"] = assign_tiers(df, CLEAR_THRESHOLD, FLAG_THRESHOLD)

    breakdown = tier_breakdown(df)
    breakdown.to_csv(RESULTS_DIR / "triage_tiers.csv")
    print(f"Clear threshold: p < {CLEAR_THRESHOLD}, flag threshold: p >= {FLAG_THRESHOLD}\n")
    print("Tier breakdown:")
    print(breakdown.round(4).to_string())

    half_widths = np.linspace(0.0, TRAIN_PREVALENCE, 21)
    tradeoff = sweep_tradeoff(df, half_widths, TRAIN_PREVALENCE)
    tradeoff.to_csv(RESULTS_DIR / "triage_tradeoff.csv", index=False)

    plot_tier_breakdown(breakdown, FIG_DIR)
    plot_tradeoff(tradeoff, FIG_DIR)

    print(f"\nSaved tier breakdown to {RESULTS_DIR / 'triage_tiers.csv'}, "
          f"tradeoff sweep to {RESULTS_DIR / 'triage_tradeoff.csv'}, "
          f"figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
