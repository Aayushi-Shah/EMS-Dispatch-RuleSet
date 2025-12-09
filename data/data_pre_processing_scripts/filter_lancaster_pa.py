#!/usr/bin/env python3
"""
Extract Lancaster-only incidents from the cleaned PA medical dataset
and run a structured missing-field audit.

Usage:
  python extract_lancaster_and_audit.py \
    --in medical_calls_pa_clean.parquet \
    --out-parquet medical_calls_lancaster.parquet \
    --out-csv medical_calls_lancaster_preview.csv
"""
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.ems_config import settings

PA_LAT_MIN, PA_LAT_MAX = 39.5, 42.3
PA_LON_MIN, PA_LON_MAX = -80.6, -74.7

def norm_text(s):
    return re.sub(r"\s+", " ", str(s).strip()).lower() if pd.notna(s) else ""

def main():
    src = Path(settings.PA_MEDICAL)
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_parquet(src)
    n_all = len(df)

    # ---------- Lancaster-only filter ----------
    # Prefer explicit county; fall back to bbox + known city names if county missing.
    county_norm = df.get("county").astype(str).map(norm_text) if "county" in df.columns else pd.Series(index=df.index, dtype="object")
    is_lancaster_county = county_norm.str.contains(r"\blancaster\b") & county_norm.str.contains("county")

    # Coordinate-based safety net (Lancaster County bbox is ~ within PA bounds; keep broad PA bbox)
    lat = pd.to_numeric(df.get("latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("longitude"), errors="coerce")
    in_pa_bbox = (
        lat.between(PA_LAT_MIN, PA_LAT_MAX, inclusive="both")
        & lon.between(PA_LON_MIN, PA_LON_MAX, inclusive="both")
        & lat.notna() & lon.notna()
        & ~np.isclose(lat, 0) & ~np.isclose(lon, 0)
    )

    # If county missing, allow municipality/city hints (very light heuristic)
    muni_norm = df.get("municipality").astype(str).map(norm_text) if "municipality" in df.columns else pd.Series(index=df.index, dtype="object")
    looks_like_lancaster_city = muni_norm.str.contains(r"\blancaster\b")

    mask_keep = is_lancaster_county | ((county_norm == "") & in_pa_bbox & looks_like_lancaster_city)

    df_lanc = df.loc[mask_keep].copy()
    n_lanc = len(df_lanc)

    # ---------- Audit ----------
    def pct(x): return f"{100.0*float(x):.2f}%"

    # Core fields presence
    fields = ["incidentID","t_received","description","county","municipality","latitude","longitude","unitsString"]
    missing = {c: pct(df_lanc[c].isna().mean()) if c in df_lanc.columns else "N/A" for c in fields}

    # Units presence
    if "unitsString" in df_lanc.columns:
        has_units = df_lanc["unitsString"].astype(str).str.strip().replace({"": np.nan}).notna()
        units_pct = pct(has_units.mean())
        units_examples = df_lanc.loc[has_units, "unitsString"].astype(str).head(5).tolist()
    else:
        units_pct, units_examples = "N/A", []

    # Time sanity
    if "t_received" in df_lanc.columns:
        t = pd.to_datetime(df_lanc["t_received"], utc=True, errors="coerce")
        time_na = pct(t.isna().mean())
        is_utc = getattr(t.dtype, "tz", None) is not None
        t_min, t_max = str(t.min()), str(t.max())
    else:
        time_na, is_utc, t_min, t_max = "N/A", False, "NA", "NA"

    # Duplicates
    dup_pct = pct(df_lanc["incidentID"].astype(str).duplicated(keep=False).mean()) if "incidentID" in df_lanc.columns else "N/A"

    # Coordinates quality
    bad_coords = lat.loc[df_lanc.index].isna() | lon.loc[df_lanc.index].isna() | np.isclose(lat.loc[df_lanc.index], 0) | np.isclose(lon.loc[df_lanc.index], 0)
    in_bbox = (
        lat.loc[df_lanc.index].between(PA_LAT_MIN, PA_LAT_MAX, inclusive="both")
        & lon.loc[df_lanc.index].between(PA_LON_MIN, PA_LON_MAX, inclusive="both")
        & ~bad_coords
    )
    coord_bad_pct = pct(bad_coords.mean())
    coord_in_bbox_pct = pct(in_bbox.mean())

    # Top municipalities (helps spot location coverage)
    top_muni = {}
    if "municipality" in df_lanc.columns:
        top_muni = df_lanc["municipality"].astype(str).str.strip().value_counts().head(10).to_dict()

    # ---------- Save outputs ----------
    Path(settings.LANCASTER_MEDICAL).parent.mkdir(parents=True, exist_ok=True)
    df_lanc.to_parquet(settings.LANCASTER_MEDICAL, index=False)

    # small preview for quick eyeballing
    n_prev = min(10_000, n_lanc)
    df_lanc.head(n_prev).to_csv(settings.LANCASTER_MEDICAL_CSV, index=False)

    # ---------- Print summary ----------
    print("\n=== Lancaster Subset Audit ===")
    print(f"Input rows (PA medical, deduped): {n_all:,}")
    print(f"Kept Lancaster County rows      : {n_lanc:,}")
    print("\nMissing (%) among Lancaster subset:")
    for k in fields:
        print(f"  {k:12s}: {missing.get(k,'N/A')}")
    print(f"\nunitsString present             : {units_pct}")
    if units_examples:
        print(f"unitsString examples            : {units_examples}")
    print(f"\nTime parsed NA                 : {time_na}")
    print(f"t_received is UTC              : {is_utc}")
    print(f"time range                     : {t_min}  →  {t_max}")
    print(f"\nDuplicate incidentID           : {dup_pct}")
    print(f"Bad coords (0/NaN)             : {coord_bad_pct}")
    print(f"In PA bbox (sanity)            : {coord_in_bbox_pct}")
    if top_muni:
        print(f"\nTop municipalities (10): {top_muni}")

    print(f"\nSaved → {settings.LANCASTER_MEDICAL}")
    print(f"Preview → {settings.LANCASTER_MEDICAL_CSV} (first {n_prev} rows)")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
