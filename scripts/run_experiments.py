"""5 imbalance strategies x 3 models. Run: python scripts/run_experiments.py   (add --fast to skip the slow SVM)"""
import sys; sys.path.insert(0, "src")
import os, time, warnings
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from faultdiag.data import load_data, derive_labels
from faultdiag.evaluate import metrics
from faultdiag.features import build_features
from faultdiag.imbalance import STRATEGIES, resample
from faultdiag.models import MODEL_NAMES, make_model
from faultdiag.split import split_pumps

warnings.filterwarnings("ignore")
models = [m for m in MODEL_NAMES if not (m == "svm" and "--fast" in sys.argv)]
df, X = build_features(load_data()); y = derive_labels(df)
masks = {k: df.asset_id.isin(p).values for k, p in zip(["train", "val", "test"], split_pumps(df))}
scaler = StandardScaler().fit(X[masks["train"]])               # fitted on training pumps only
Z = {k: scaler.transform(X[m]) for k, m in masks.items()}
Y = {k: y[m] for k, m in masks.items()}

rows = []
for strategy in STRATEGIES:
    Xs, ys, w = resample(strategy, Z["train"], Y["train"])      # training data only
    for name in models:
        t0 = time.time(); model = make_model(name)
        model.fit(Xs, ys, sample_weight=w) if w is not None else model.fit(Xs, ys)
        row = {"strategy": strategy, "model": name, "fit_seconds": round(time.time() - t0, 1)}
        for split in ("val", "test"):
            for k, v in metrics(Y[split], model.predict(Z[split])).items():
                row[f"{split}_{k}"] = v
        rows.append(row); print(f"{strategy:13s} {name:13s} val macro F1 {row['val_macro_f1']:.3f} | test macro F1 {row['test_macro_f1']:.3f} | {row['fit_seconds']}s", flush=True)

res = pd.DataFrame(rows); os.makedirs("results", exist_ok=True); res.to_csv("results/results.csv", index=False)
print("\nTest macro F1:"); print(res.pivot(index="strategy", columns="model", values="test_macro_f1").to_string())
best = res.loc[res["val_macro_f1"].idxmax()]
print(f"\nBest by VALIDATION macro F1: {best.strategy} + {best.model}  (test macro F1 {best.test_macro_f1})")
# Retrain the best model using the selected training strategy
best_X, best_y, best_weights = resample(
    best.strategy, Z["train"], Y["train"]
)

best_model = make_model(best.model)

if best_weights is not None:
    best_model.fit(best_X, best_y, sample_weight=best_weights)
else:
    best_model.fit(best_X, best_y)

# Save model and preprocessing information
os.makedirs("results", exist_ok=True)

joblib.dump({
    "model": best_model,
    "scaler": scaler,
    "feature_columns": X.columns.tolist(),
    "classes": ["Normal", "Overload", "Pressure_drop", "Shutdown"],
    "best_strategy": best.strategy,
    "best_model": best.model
}, "results/intellitwin_model.joblib")

print("Trained model saved successfully!")

import make_report
make_report.main()                                   # builds results/report.html and opens it in your browser
if os.environ.get("TERM_PROGRAM") != "vscode":       # keep the window open if launched outside VS Code
    input("\nPress Enter to close...")
