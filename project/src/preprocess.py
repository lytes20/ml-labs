"""Shared preprocessing: load, de-duplicate, split, and scale the diabetes dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
LABEL = "Diabetes_binary"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw_and_scaled_split():
    """Same dedup + stratified split as load_and_split, but also returns the
    raw (unscaled) train/test features, so downstream steps that need
    human-readable feature values (e.g. discretizing BMI or GenHlth for a
    Bayesian network) can line up with the scaled features by index."""
    df = pd.read_csv(DATA_FILE)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=[LABEL])
    y = df[LABEL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def load_and_split():
    _, _, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_raw_and_scaled_split()
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = load_and_split()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
