"""Load the Breast Cancer Wisconsin (Diagnostic) dataset."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def load_data():
    """Return (X, y). y = 1 for malignant (positive class), 0 for benign.

    scikit-learn encodes malignant as 0, so the target is flipped here.
    """
    data = load_breast_cancer(as_frame=True)
    y = (1 - data.target).rename("malignant")
    return data.data, y


def split_data(X, y, test_size=0.2, seed=SEED):
    """Stratified train/test split with a fixed seed (deterministic)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def export_csv(path=ROOT / "data" / "breast_cancer_wisconsin.csv"):
    X, y = load_data()
    pd.concat([X, y], axis=1).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print("Wrote", export_csv())
