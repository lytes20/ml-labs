"""Combine classical + DNN results into one comparison table and chart,
benchmarked against Ayoade et al. (2025)'s reported DNN F1 on binary BRFSS."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

AYOADE_DNN_F1 = 0.45

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
MUTED = "#898781"


def plot_comparison(results_df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(results_df))
    width = 0.25
    colors = [BLUE, ORANGE, AQUA]

    for i, metric in enumerate(["accuracy", "f1", "auc"]):
        offset = (i - 1) * width
        ax.bar(
            [p + offset for p in x], results_df[metric], width=width,
            label=metric.upper(), color=colors[i],
        )

    ax.axhline(
        AYOADE_DNN_F1, color=RED, linewidth=1.5, linestyle="--",
        label=f"Ayoade et al. (2025) DNN F1 = {AYOADE_DNN_F1}",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(results_df.index, rotation=10, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Classical vs. deep learning: full comparison", color=INK, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "full_model_comparison.png", dpi=150)
    plt.close(fig)


def main():
    classical = pd.read_csv(RESULTS_DIR / "classical_results.csv", index_col=0)
    dnn = pd.read_csv(RESULTS_DIR / "dnn_results.csv", index_col=0)
    combined = pd.concat([classical, dnn])
    combined.to_csv(RESULTS_DIR / "all_results.csv")

    print("Combined results:")
    print(combined.round(4).to_string())

    print(f"\nFor reference, Ayoade et al. (2025) reported a feed-forward DNN F1 of {AYOADE_DNN_F1} "
          f"on the binary BRFSS subset.")
    print(f"Our DNN F1: {combined.loc['DNN (Feed-forward)', 'f1']:.4f} "
          f"({'above' if combined.loc['DNN (Feed-forward)', 'f1'] > AYOADE_DNN_F1 else 'below'} their reported number)")

    plot_comparison(combined, FIG_DIR)
    print(f"\nSaved combined results to {RESULTS_DIR / 'all_results.csv'}, figure to {FIG_DIR}")


if __name__ == "__main__":
    main()
