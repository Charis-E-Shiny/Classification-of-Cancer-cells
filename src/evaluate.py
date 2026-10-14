"""Evaluate the saved model on the held-out test set.  Run:  python -m src.evaluate"""
import json

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, brier_score_loss,
                             classification_report)

from src.data_loader import ROOT, load_data, split_data

FIG = ROOT / "reports" / "figures"


def load_model():
    model = joblib.load(ROOT / "models" / "model.joblib")
    meta = json.load(open(ROOT / "models" / "metadata.json"))
    return model, meta


def false_negatives(model, X_test, y_test, X_train, y_train, threshold=0.5):
    """Missed malignant cases with the features that deviate most from the malignant average."""
    p = model.predict_proba(X_test)[:, 1]
    mask = (y_test.values == 1) & (p < threshold)
    mu, sd = X_train[y_train == 1].mean(), X_train[y_train == 1].std()
    rows = []
    for idx, score in zip(X_test.index[mask], p[mask]):
        z = ((X_test.loc[idx] - mu) / sd).abs().sort_values(ascending=False)
        rows.append({"sample": idx, "malignant_probability": round(float(score), 3),
                     "most_atypical_features": ", ".join(z.index[:3])})
    return pd.DataFrame(rows)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    model, meta = load_model()
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    p = model.predict_proba(X_test)[:, 1]
    labels = ["benign", "malignant"]

    for tag, t in [("default_0.5", 0.5), ("tuned", meta["threshold"])]:
        pred = (p >= t).astype(int)
        print(f"\n=== {meta['model']} @ threshold {t:.3f} ({tag}) ===")
        print(classification_report(y_test, pred, target_names=labels, digits=3))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=labels, colorbar=False)
        plt.title(f"{meta['model']} - threshold {t:.2f}"); plt.tight_layout()
        plt.savefig(FIG / f"confusion_matrix_{tag}.png", dpi=130); plt.close()

    RocCurveDisplay.from_predictions(y_test, p, name=meta["model"]); plt.tight_layout()
    plt.savefig(FIG / "roc_curve.png", dpi=130); plt.close()
    CalibrationDisplay.from_predictions(y_test, p, n_bins=6, name=meta["model"]); plt.tight_layout()
    plt.savefig(FIG / "calibration.png", dpi=130); plt.close()
    print(f"Brier score: {brier_score_loss(y_test, p):.4f}")

    fn = false_negatives(model, X_test, y_test, X_train, y_train, 0.5)
    fn.to_csv(ROOT / "reports" / "false_negatives.csv", index=False)
    print("\nMissed malignant cases at 0.5:\n", fn.to_string(index=False) if len(fn) else "none")


if __name__ == "__main__":
    main()
