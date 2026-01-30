# src/preprocessing.py`
# Cleans the data (handles missing values, sorts by time, prepares robot groups, normalizes features).
# Creates the target label: **time_to_failure_days** by detecting failure events and calculating the time until the next one.
# Outputs training-ready data so Linear Regression can **predict the next failure timing**.


from __future__ import annotations

import pandas as pd
import numpy as np


def parse_time(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    return out


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Drop rows with missing time; fill numeric NaNs with column median
    if "Time" in out.columns:
        out = out.dropna(subset=["Time"])

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        out[col] = out[col].fillna(out[col].median())

    # Fill any remaining NaNs in non-numeric with empty string
    for col in out.columns:
        if col not in numeric_cols:
            out[col] = out[col].fillna("")

    return out


def _rolling_zscore(series: pd.Series, window: int = 120) -> pd.Series:
    """
    Rolling z-score to detect abnormal behavior.
    Window ~ 120 points; tune if your sampling frequency differs.
    """
    s = series.astype(float)
    mean = s.rolling(window=window, min_periods=max(10, window // 4)).mean()
    std = s.rolling(window=window, min_periods=max(10, window // 4)).std().replace(0, np.nan)
    z = (s - mean) / std
    return z.fillna(0.0)


def engineer_failure_and_target(
    df: pd.DataFrame,
    robot_id_col: str,
    time_col: str,
    feature_col: str,
    z_threshold: float = 3.0,
    min_gap_minutes_between_failures: int = 60,
    target_name: str = "time_to_failure_days",
) -> pd.DataFrame:
    """
    Your dataset does not include a failure label, so we engineer it:
      - Failure event occurs when rolling |zscore(feature)| >= z_threshold
      - Then target = days until NEXT failure event (per robot)
    """
    out = df.copy()
    out = out.sort_values([robot_id_col, time_col]).reset_index(drop=True)

    targets = []
    failure_flags = []

    for robot_id, g in out.groupby(robot_id_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)

        z = _rolling_zscore(g[feature_col])
        raw_fail = (z.abs() >= z_threshold).astype(int)

        # Enforce minimum gap between failures (avoid bursts)
        fail_times = g.loc[raw_fail == 1, time_col].tolist()
        filtered = []
        last_t = None
        for t in fail_times:
            if last_t is None:
                filtered.append(t)
                last_t = t
            else:
                delta_min = (t - last_t).total_seconds() / 60.0
                if delta_min >= min_gap_minutes_between_failures:
                    filtered.append(t)
                    last_t = t

        failure_event = g[time_col].isin(filtered).astype(int)
        failure_flags.append(failure_event)

        # Compute time until next failure
        next_fail_time = []
        fail_idx = [i for i, v in enumerate(failure_event.tolist()) if v == 1]

        if len(fail_idx) == 0:
            # No failures detected; set large constant target
            next_fail_time = [np.nan] * len(g)
        else:
            # For each row, find next failure time
            fail_times_filtered = g.loc[failure_event == 1, time_col].tolist()
            j = 0
            for i in range(len(g)):
                while j < len(fail_times_filtered) and fail_times_filtered[j] <= g.loc[i, time_col]:
                    j += 1
                if j < len(fail_times_filtered):
                    delta_days = (fail_times_filtered[j] - g.loc[i, time_col]).total_seconds() / 86400.0
                    next_fail_time.append(delta_days)
                else:
                    next_fail_time.append(np.nan)

        targets.append(pd.Series(next_fail_time))
    out["failure_event"] = pd.concat(failure_flags, ignore_index=True)
    out[target_name] = pd.concat(targets, ignore_index=True)

    # If target is missing at the tail (after last failure), fill with median of known targets
    if out[target_name].isna().all():
        out[target_name] = 9999.0
    else:
        med = out[target_name].dropna().median()
        out[target_name] = out[target_name].fillna(med)

    return out


def select_univariate_xy(df: pd.DataFrame, feature_col: str, target_col: str):
    X = df[[feature_col]].astype(float).values
    y = df[target_col].astype(float).values
    return X, y
