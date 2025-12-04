#!/usr/bin/env python3
"""
Full preprocessing pipeline for EMS project (Task 0):
1. Filter to medical calls
2. Keep only Pennsylvania incidents
3. Parse timestamps to UTC
4. Deduplicate by incidentID (keep latest)
5. Save as medical_calls_pa_clean.parquet

Usage:
  python filter_medical_pa.py
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

# ===================== CONFIG =====================
INPUT_JSON  = Path("data/raw/all_incidents.json")         # large JSON input
OUTPUT_PARQ = Path("data/processed/medical_calls_pa_clean.parquet")
OUTPUT_CSV  = Path("data/processed/medical_calls_pa_clean_preview.csv")
CHUNKSIZE   = 100_000

# --- Keyword rules (expand if professor approves later) ---
MEDICAL_KEYWORDS = re.compile(
    r"\b("
    r"medic|medical|ems|emergency|injur|accident|unconsc|cardiac|"
    r"breath|overdose|bleed|trauma|seizure|stroke|"
    r"chest pain|respiratory|diabetic|collapse|pain"
    r")\b",
    re.IGNORECASE,
)

# --- Pennsylvania bounding box ---
PA_LAT_MIN, PA_LAT_MAX = 39.5, 42.3
PA_LON_MIN, PA_LON_MAX = -80.6, -74.7

# ===========================================================
def normalize_state(s):
    if not isinstance(s, str):
        return None
    s = s.strip().upper()
    if s in {"PA", "PENNSYLVANIA"}:
        return "PA"
    return s

def normalize_timezone(s):
    if not isinstance(s, str):
        return s
    return re.sub(r"\s+(UTC|GMT)$", " +0000", s.strip())

def extract_epoch_ms(x):
    if isinstance(x, dict) and "$date" in x:
        return float(x["$date"])
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x if x > 1e12 else x * 1000)
    if isinstance(x, pd.Timestamp):
        return float(x.value // 1e6)
    if isinstance(x, str):
        t = pd.to_datetime(x, utc=True, errors="coerce")
        return float(t.value // 1e6) if pd.notna(t) else None
    return None

def best_update_ts(row):
    for col in ("_updated_at", "_created_at", "t_received"):
        if col in row and pd.notna(row[col]):
            if col == "t_received":
                ts = pd.to_datetime(row[col], utc=True, errors="coerce")
                if pd.notna(ts):
                    return float(ts.value // 1e6)
            else:
                ms = extract_epoch_ms(row[col])
                if ms is not None:
                    return ms
    return -1.0

# ===========================================================
# 1. Stream load + filter medical
rows_kept, rows_total = 0, 0
filtered_chunks = []

print("🔹 Reading chunks & filtering medical calls...")
for chunk in pd.read_json(INPUT_JSON, lines=True, chunksize=CHUNKSIZE):
    rows_total += len(chunk)
    mask = chunk["description"].astype(str).str.contains(MEDICAL_KEYWORDS, na=False)
    filtered = chunk.loc[mask].copy()
    rows_kept += len(filtered)
    filtered_chunks.append(filtered)

df = pd.concat(filtered_chunks, ignore_index=True)
print(f"✅ Filtered {rows_kept:,} medical rows from {rows_total:,} total.")

# ===========================================================
# 2. Keep only Pennsylvania incidents
df["state_norm"] = df["state"].map(normalize_state)

mask_pa = df["state_norm"].eq("PA")
coord_ok = (
    df["latitude"].between(PA_LAT_MIN, PA_LAT_MAX, inclusive="both")
    & df["longitude"].between(PA_LON_MIN, PA_LON_MAX, inclusive="both")
    & df["latitude"].notna()
    & df["longitude"].notna()
    & (~np.isclose(df["latitude"], 0))
    & (~np.isclose(df["longitude"], 0))
)
mask_keep = mask_pa | (df["state_norm"].isna() & coord_ok)
df = df.loc[mask_keep].copy()
print(f"✅ Kept {len(df):,} Pennsylvania rows.")

# ===========================================================
# 3. Parse timestamps to UTC
ser = df["incidentTime"].astype(str).map(normalize_timezone)
df["t_received"] = pd.to_datetime(ser, utc=True, errors="coerce")
print(f"⏱️ Parsed timestamps. Missing: {(df['t_received'].isna().mean()*100):.2f}%")

# ===========================================================
# 4. Deduplicate by incidentID
if "incidentID" in df.columns:
    print("🔹 Deduplicating by incidentID...")
    n_before = len(df)

    # compute a numeric "last updated" key
    df["updated_ms"] = df.apply(best_update_ts, axis=1)
    df["_ix"] = np.arange(len(df))  # stable tie-breaker

    # stable sort so the final (kept) row per incidentID is the latest
    df = df.sort_values(
        ["incidentID", "updated_ms", "_ix"],
        kind="mergesort"  # guarantees stability
    )

    # keep the last (latest) row per incidentID
    df = df.drop_duplicates(subset="incidentID", keep="last").copy()

    # clean up
    df.drop(columns=["updated_ms", "_ix"], inplace=True, errors="ignore")

    n_after = len(df)
    print(f"✅ Deduplicated: {n_before:,} → {n_after:,} rows (removed {n_before - n_after:,}).")

    # quick residual check
    dup_rate = df["incidentID"].astype(str).duplicated(keep=False).mean() * 100
    print(f"Residual duplicate incidentID rate: {dup_rate:.2f}%")
else:
    print("⚠️ No incidentID column found — skipping deduplication.")

# ===========================================================
# 5. Save outputs
df.to_parquet(OUTPUT_PARQ, index=False)
df.head(10_000).to_csv(OUTPUT_CSV, index=False)
print(f"💾 Saved {OUTPUT_PARQ} and {OUTPUT_CSV}")

# Quick post-check
if "county" in df.columns:
    print("Top counties:", df["county"].astype(str).str.strip().value_counts().head(5).to_dict())
if "unitsString" in df.columns:
    pct_units = df["unitsString"].astype(str).str.strip().replace({"": np.nan}).notna().mean()*100
    print(f"Units present in {pct_units:.2f}% of rows")
