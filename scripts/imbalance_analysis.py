"""Quantify the imbalance and show why accuracy misleads. Run: python scripts/imbalance_analysis.py"""
import sys; sys.path.insert(0, "src")
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from faultdiag.data import load_data, derive_labels
from faultdiag.imbalance import class_table, majority_baseline_accuracy
from faultdiag.split import split_pumps

df = load_data(); y = derive_labels(df)
os.makedirs("results", exist_ok=True)
print("ALL DATA"); t = class_table(y); print(t.to_string())
print(f"\nA model that always predicts 'Normal' is {majority_baseline_accuracy(y):.1%} accurate and finds 0 faults.")
for name, pumps in zip(["train", "val", "test"], split_pumps(df)):
    print(f"\n{name.upper()} ({len(pumps)} pumps)"); print(class_table(y[df.asset_id.isin(pumps).values]).rows.to_string())
ax = t.rows.plot.barh(color="#2dd4bf"); ax.set_xscale("log"); ax.set_xlabel("rows (log scale)"); ax.set_title("Class distribution")
plt.tight_layout(); plt.savefig("results/class_distribution.png", dpi=150); print("\nsaved results/class_distribution.png")
