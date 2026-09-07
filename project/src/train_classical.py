"""Train and compare classical baselines: Logistic Regression, Random Forest, Linear SVM.

Note on SVM: a kernel SVC does not scale to ~180k training rows in reasonable
time, so we use LinearSVC (linear-kernel SVM solved via liblinear) as the SVM
baseline, matching common practice for large tabular datasets.
"""

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC

from preprocess import load_and_split

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
MUTED = "#898781"


def evaluate(name, model, X_test, y_test, scores):
    preds = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.decision_function(X_test)

    scores[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs),
    }


def plot_comparison(results_df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(results_df))
    width = 0.25

    colors = [BLUE, ORANGE, AQUA]
    for i, metric in enumerate(["accuracy", "f1", "auc"]):
        offset = (i - 1) * width
        ax.bar(
            [p + offset for p in x], results_df[metric], width=width,
            label=metric.upper(), color=colors[i],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(results_df.index)
    ax.set_ylim(0, 1)
    ax.set_title("Classical model comparison", color=INK, loc="left", pad=12)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "classical_model_comparison.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, scaler = load_and_split()

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Linear SVM": LinearSVC(class_weight="balanced", max_iter=5000, random_state=42),
    }

    scores = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        evaluate(name, model, X_test, y_test, scores)
        scores[name]["train_time_s"] = elapsed
        joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        print(f"{name}: trained in {elapsed:.1f}s")

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    results_df = pd.DataFrame(scores).T
    results_df = results_df[["accuracy", "f1", "auc", "train_time_s"]]
    results_df.to_csv(RESULTS_DIR / "classical_results.csv")

    print("\nComparative results:")
    print(results_df.round(4).to_string())

    plot_comparison(results_df, FIG_DIR)
    print(f"\nSaved results to {RESULTS_DIR}, models to {MODELS_DIR}, figure to {FIG_DIR}")


if __name__ == "__main__":
    main()
