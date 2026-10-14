"""Model explanations via permutation importance (no extra dependencies).
Run:  python -m src.explain"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from src.data_loader import ROOT, SEED, load_data, split_data
from src.evaluate import load_model


def main(top_n=10):
    model, meta = load_model()
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    r = permutation_importance(model, X_test, y_test, scoring="average_precision",
                               n_repeats=30, random_state=SEED)
    imp = pd.DataFrame({"feature": X.columns, "mean_drop_in_pr_auc": r.importances_mean,
                        "std": r.importances_std}).sort_values("mean_drop_in_pr_auc", ascending=False)
    imp.to_csv(ROOT / "reports" / "permutation_importance.csv", index=False)
    top = imp.head(top_n).iloc[::-1]
    plt.figure(figsize=(7, 4.5))
    plt.barh(top["feature"], top["mean_drop_in_pr_auc"], xerr=top["std"])
    plt.xlabel("Drop in PR-AUC when feature is shuffled"); plt.title(f"{meta['model']}: top features (test set)")
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "figures" / "permutation_importance.png", dpi=130)
    print(imp.head(top_n).round(4).to_string(index=False))
    print("\nNote: many features are highly correlated (radius/perimeter/area), so importance is shared "
          "between them and individual rankings should be read with care.")


if __name__ == "__main__":
    main()
