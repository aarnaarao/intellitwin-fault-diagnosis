"""Evaluation metrics. Macro F1 is the main metric because accuracy hides rare classes."""
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from faultdiag.data import CLASSES

LABELS = list(range(len(CLASSES)))


def metrics(y_true, y_pred):
    out = {"macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
           "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0),
           "accuracy": accuracy_score(y_true, y_pred)}
    rec = recall_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    pre = precision_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    for c, r, p in zip(CLASSES, rec, pre):
        out[f"recall_{c}"], out[f"precision_{c}"] = r, p
    return {k: round(float(v), 4) for k, v in out.items()}
