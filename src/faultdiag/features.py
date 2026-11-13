"""Feature engineering. Uses only the current reading and the pump's PAST readings."""
import numpy as np
import pandas as pd

SENSORS = ["flow_rate_bpd", "pressure_psi", "temperature_c", "vibration_mm_s",
           "rotation_rpm", "power_kw", "current_a", "voltage_v", "efficiency_pct"]
CONTEXT = ["operating_hours", "depth_m", "water_cut_pct", "gor_scf_bbl", "sand_rate_ppm",
           "fluid_viscosity_cp", "fluid_density_kg_m3", "co2_ppm", "h2s_ppm",
           "ambient_temp_c", "days_since_maintenance"]
CATEGORICAL = ["pump_type", "manufacturer"]
WINDOW = 7  # each reading is compared with the pump's previous 7 readings


def build_features(df, columns=None):
    """Return (sorted dataframe, feature matrix).

    Pass `columns` (saved at training time) to guarantee the same column order later.
    The event flags, rul_days and health_index are deliberately NOT used.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    df["pressure_per_flow"] = df.pressure_psi / df.flow_rate_bpd
    df["power_per_flow"] = df.power_kw / df.flow_rate_bpd

    grouped = df.groupby("asset_id")
    for c in SENSORS:
        # shift(1) excludes the current reading, so no information from today or the future
        prev = grouped[c].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=3).mean())
        df[c + "_dev7"] = ((df[c] - prev) / prev.replace(0, np.nan)).fillna(0)

    numeric = (SENSORS + CONTEXT + ["pressure_per_flow", "power_per_flow"]
               + [c + "_dev7" for c in SENSORS])
    X = pd.concat([df[numeric], pd.get_dummies(df[CATEGORICAL], dtype=int)], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return df, X