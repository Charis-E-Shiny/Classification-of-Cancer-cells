"""Preprocessing and model definitions. Scaling lives inside the pipeline so it is
fit on training folds only (no leakage) and is saved together with the model."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def get_models(seed=42):
    return {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=5000, class_weight="balanced")),
        "Random Forest": build_pipeline(
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed)),
        "Gradient Boosting": build_pipeline(GradientBoostingClassifier(random_state=seed)),
    }
