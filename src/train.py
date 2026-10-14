"""Train and compare models. Run from the repo root:  python -m src.train

Model selection uses cross-validated (out-of-fold) PR-AUC on the TRAINING set only.
The decision threshold is also chosen on out-of-fold training scores. The test set is
used once, for the final reported numbers.
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data_loader import ROOT, SEED, load_data, split_data
from src.preprocessing import get_models

MIN_RECALL = 0.98


def pick_threshold(y_true, scores, min_recall=MIN_RECALL):
    """Highest threshold whose recall on the given scores is >= min_recall."""
    _, rec, thr = precision_recall_curve(y_true, scores)
    ok = np.where(rec[:-1] >= min_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5


def row_at(y_true, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return tp, fn, fp, tn, recall_score(y_true, pred), tp / max(tp + fp, 1)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, model in get_models(SEED).items():
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thr = pick_threshold(y_train, oof)
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        tp, fn, fp, tn, rec, prec = row_at(y_test, p, 0.5)
        tp2, fn2, fp2, tn2, rec2, prec2 = row_at(y_test, p, thr)
        rows.append({
            "model": name, "cv_pr_auc": average_precision_score(y_train, oof),
            "cv_roc_auc": roc_auc_score(y_train, oof),
            "test_pr_auc": average_precision_score(y_test, p), "test_roc_auc": roc_auc_score(y_test, p),
            "threshold_tuned": thr,
            "recall@0.5": rec, "precision@0.5": prec, "FN@0.5": fn, "FP@0.5": fp,
            "recall@tuned": rec2, "precision@tuned": prec2, "FN@tuned": fn2, "FP@tuned": fp2,
        })
        fitted[name] = (model, p, thr)
        print(f"{name:20s} CV PR-AUC {rows[-1]['cv_pr_auc']:.4f} | test PR-AUC {rows[-1]['test_pr_auc']:.4f} "
              f"| FN@0.5={fn} FN@tuned={fn2} (FP {fp}->{fp2})")

    df = pd.DataFrame(rows).round(4)
    (ROOT / "reports").mkdir(exist_ok=True)
    df.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

    best = df.sort_values("cv_pr_auc", ascending=False).iloc[0]["model"]   # chosen on CV, not test
    model, _, thr = fitted[best]
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "model.joblib")
    json.dump({"model": best, "threshold": thr, "min_recall_target": MIN_RECALL,
               "features": list(X.columns), "positive_class": "malignant", "seed": SEED},
              open(ROOT / "models" / "metadata.json", "w"), indent=2)
    print(f"\nSelected (best CV PR-AUC): {best}  | tuned threshold = {thr:.3f}")

    fig_dir = ROOT / "reports" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4.5))
    for name, (_, p, _) in fitted.items():
        pr, rc, _ = precision_recall_curve(y_test, p)
        plt.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_test, p):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall (test)")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(fig_dir / "pr_curves_all_models.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
