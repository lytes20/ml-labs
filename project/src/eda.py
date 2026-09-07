"""Exploratory data analysis on the Diabetes Health Indicators Dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_FILE = DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
MUTED = "#898781"

BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]
NUMERIC_FEATURES = ["BMI", "MentHlth", "PhysHlth"]


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_class_balance(df, out_dir):
    counts = df["Diabetes_binary"].value_counts().sort_index()
    labels = ["No diabetes/prediabetes", "Diabetes/prediabetes"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=[BLUE, ORANGE], width=0.6)
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / counts.sum()
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", color=INK, fontsize=9)

    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_title("Class balance", color=INK, loc="left", pad=12)
    ax.set_ylabel("Records")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=150)
    plt.close(fig)


def plot_bmi_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, value, color in [("No diabetes", 0, BLUE), ("Diabetes/prediabetes", 1, ORANGE)]:
        subset = df.loc[df["Diabetes_binary"] == value, "BMI"]
        ax.hist(subset, bins=40, range=(10, 60), density=True, alpha=0.55, color=color, label=label)

    ax.set_title("BMI distribution by class", color=INK, loc="left")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "bmi_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df, out_dir):
    corr = df.corr(numeric_only=True)["Diabetes_binary"].drop("Diabetes_binary").sort_values()

    colors = [ORANGE if v < 0 else BLUE for v in corr.values]
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_title("Feature correlation with diabetes label", color=INK, loc="left")
    ax.set_xlabel("Pearson correlation")
    style_axes(ax)
    ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_correlation.png", dpi=150)
    plt.close(fig)


def plot_rate_by_category(df, feature, out_dir):
    rate = df.groupby(feature)["Diabetes_binary"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rate.index.astype(str), rate.values, color=BLUE, width=0.6)
    ax.set_title(f"Diabetes rate by {feature}", color=INK, loc="left")
    ax.set_xlabel(feature)
    ax.set_ylabel("Diabetes rate")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"rate_by_{feature.lower()}.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_FILE)

    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nClass balance:")
    print(df["Diabetes_binary"].value_counts(normalize=True))

    plot_class_balance(df, FIG_DIR)
    plot_bmi_distribution(df, FIG_DIR)
    plot_correlation_heatmap(df, FIG_DIR)
    plot_rate_by_category(df, "GenHlth", FIG_DIR)
    plot_rate_by_category(df, "Age", FIG_DIR)

    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
