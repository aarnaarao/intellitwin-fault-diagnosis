"""Build results/report.html from results/results.csv and open it in the browser.
Run alone any time (no retraining):  python scripts/make_report.py"""
import base64, io, sys, webbrowser
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ["Normal", "Overload", "Pressure_drop", "Shutdown"]
ROOT = Path(__file__).resolve().parent.parent


def _img(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return '<img src="data:image/png;base64,%s">' % base64.b64encode(buf.getvalue()).decode()


def main(open_browser=True):
    res = pd.read_csv(ROOT / "results" / "results.csv")
    res["combo"] = res.strategy + " + " + res.model
    best = res.loc[res.val_macro_f1.idxmax()]

    # 1. heatmap: test macro F1
    piv = res.pivot(index="strategy", columns="model", values="test_macro_f1")
    fig, ax = plt.subplots(figsize=(8, 3.6)); im = ax.imshow(piv.values, cmap="YlGn", vmin=0.6, vmax=0.95)
    ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]): ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=11)
    ax.set_title("Test macro F1 (higher is better)"); heat = _img(fig)

    # 2. accuracy vs macro F1
    r = res.sort_values("test_macro_f1"); y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(y + 0.2, r.test_accuracy, 0.4, label="Accuracy", color="#9ca3af")
    ax.barh(y - 0.2, r.test_macro_f1, 0.4, label="Macro F1", color="#0f9d8a")
    ax.set_yticks(y, r.combo); ax.set_xlim(0.5, 1.0); ax.legend(loc="lower right")
    ax.set_title("Accuracy looks high for every model; macro F1 tells them apart"); acc = _img(fig)

    # 3. per-class recall and precision of the best model
    rec = [best[f"test_recall_{c}"] for c in CLASSES]; pre = [best[f"test_precision_{c}"] for c in CLASSES]
    x = np.arange(4); fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(x - 0.2, rec, 0.4, label="Recall", color="#0f9d8a"); ax.bar(x + 0.2, pre, 0.4, label="Precision", color="#6366f1")
    for i in range(4):
        ax.text(i - 0.2, rec[i] + 0.01, f"{rec[i]:.2f}", ha="center", fontsize=9); ax.text(i + 0.2, pre[i] + 0.01, f"{pre[i]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, CLASSES); ax.set_ylim(0, 1.12); ax.legend(loc="lower left"); ax.set_title(f"Per-class results: {best.combo}"); cls = _img(fig)

    tbl = res.sort_values("val_macro_f1", ascending=False)[["strategy", "model", "val_macro_f1", "test_macro_f1", "test_weighted_f1", "test_accuracy", "fit_seconds"]]
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Fault Diagnosis Results</title><style>
body{{font-family:Calibri,Segoe UI,sans-serif;max-width:900px;margin:30px auto;padding:0 16px;color:#111}}h1{{margin-bottom:4px}}
h2{{margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:4px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:6px 10px;text-align:right}}
td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){{text-align:left}}th{{background:#f3f4f6}}tr:first-child td{{background:#ecfdf5;font-weight:600}}
.box{{background:#ecfdf5;border-left:4px solid #0f9d8a;padding:10px 14px;margin:14px 0}}.note{{color:#555;font-size:14px}}img{{max-width:100%}}</style></head><body>
<h1>Fault Diagnosis: Model Comparison</h1><div class="note">5 imbalance strategies x 3 models, evaluated on pumps never seen in training</div>
<div class="box"><b>Best on validation data:</b> {best.strategy} + {best.model}<br>Test macro F1 <b>{best.test_macro_f1:.3f}</b> | test accuracy {best.test_accuracy:.3f}</div>
<h2>Test macro F1 by strategy and model</h2>{heat}<h2>Why accuracy is not enough</h2>{acc}
<h2>Best model, class by class</h2>{cls}<p class="note">Overload has only a few dozen rows in the test pumps, so differences of one or two points between top models are within noise.</p>
<h2>All results (sorted by validation macro F1)</h2>{tbl.round(3).to_html(index=False)}</body></html>"""
    out = ROOT / "results" / "report.html"; out.write_text(html, encoding="utf-8")
    print(f"Report saved: {out}")
    if open_browser: webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
