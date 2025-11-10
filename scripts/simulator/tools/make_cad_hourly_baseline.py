#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import timezone

# Inputs
CALLS_PARQUET = Path("data/processed/medical_calls_lemsa.parquet")
OUT_CSV = Path("reports/cad_hourly_baseline.csv")

# Candidate column names in your CAD export
RECV_CANDS = [
    "t_received","received_ts","call_received_ts","opened_ts","eventOpened_ts",
    "created_ts","time_received","call_time"
]
ARRIVE_CANDS = [
    "first_on_scene_ts","arrived_ts","onscene_ts","first_unit_arrived_ts",
    "time_arrived","arrive_on_scene_ts"
]

def pick_col(df: pd.DataFrame, cands: list[str]) -> str:
    # case-insensitive pick
    cols_lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    # fallback: try any column containing the key token
    for c in df.columns:
        lc = c.lower()
        if "receive" in lc or "open" in lc:
            if cands is RECV_CANDS:
                return c
        if "scene" in lc or "arriv" in lc:
            if cands is ARRIVE_CANDS:
                return c
    raise KeyError(f"Missing any of columns: {cands}")

def to_dt(s: pd.Series) -> pd.Series:
    # tolerate int epoch, str, datetime
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, utc=True)
    # try numeric epoch seconds or ms
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        # guess seconds vs ms by magnitude
        is_ms = s.dropna().astype(float).gt(10**12).mean() > 0.5
        factor = 1000 if is_ms else 1
        return pd.to_datetime(s.astype("Int64") * (1000 // factor), unit="ms", utc=True, errors="coerce")
    # generic parse
    return pd.to_datetime(s, utc=True, errors="coerce")

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CALLS_PARQUET)

    recv_col = pick_col(df, RECV_CANDS)
    arrv_col = pick_col(df, ARRIVE_CANDS)

    t_recv = to_dt(df[recv_col])
    t_arrv = to_dt(df[arrv_col])

    rt_min = (t_arrv - t_recv).dt.total_seconds() / 60.0

    # Clean: keep 0–60 min window, drop NaN/negatives
    rt_min = rt_min.replace([np.inf, -np.inf], np.nan)
    mask = rt_min.notna() & (rt_min >= 0) & (rt_min <= 60)
    cleaned = pd.DataFrame({
        "t_recv": t_recv[mask],
        "rt_min": rt_min[mask],
    }).dropna()

    if cleaned.empty:
        raise RuntimeError("No valid response times after cleaning. Check arrival/received columns.")

    # Hour-of-day in local-like sense (UTC OK for shape unless you need exact local)
    hours = cleaned["t_recv"].dt.hour

    by_hour = (
        pd.DataFrame({"hour": hours, "rt_min": cleaned["rt_min"]})
        .groupby("hour", as_index=False)
        .agg(p50=("rt_min", lambda x: np.percentile(x, 50)),
             p90=("rt_min", lambda x: np.percentile(x, 90)))
        .sort_values("hour")
    )

    # Ensure all 24 hours exist
    all_hours = pd.DataFrame({"hour": list(range(24))})
    by_hour = all_hours.merge(by_hour, on="hour", how="left")
    # Simple forward/backward fill for missing hours; or leave NaN if you prefer
    by_hour[["p50","p90"]] = by_hour[["p50","p90"]].interpolate(limit_direction="both")

    by_hour.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV.resolve()} with {len(by_hour)} rows.")

if __name__ == "__main__":
    main()