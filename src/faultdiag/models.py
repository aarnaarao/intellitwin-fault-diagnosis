"""Baseline machine-learning models."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_NAMES = ["random_forest", "xgboost", "svm"]


def make_model(name, seed=42):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "xgboost":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                             n_jobs=-1, random_state=seed, eval_metric="mlogloss")
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    raise ValueError(f"unknown model: {name}")
