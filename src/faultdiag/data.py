"""Data loading and fault-label definition."""
import numpy as np
import pandas as pd

CLASSES = ["Normal", "Overload", "Pressure_drop", "Shutdown"]


def load_data(path="data/pump_predictive_maintenance_dataset1.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)


def derive_labels(df):
    """Class index per row. Priority when flags overlap: Shutdown > Overload > Pressure_drop."""
    return np.select(
        [df.shutdown_event == 1, df.overload_event == 1, df.pressure_drop_event == 1],
        [3, 1, 2],
        default=0,
    )