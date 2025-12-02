#!/usr/bin/env python3
"""
Aggregate and normalize KPIs per variant for Pareto analysis.

Inputs:
  - --kpis: CSV with per-run KPIs (e.g., reference/variant_kpis_full.csv)
    Expected columns (at minimum):
      - variant_id              (frozen variant id; e.g., ff_coverage, ff_coverage_2)
      - run_id                  (per simulation run)
      - policy                  (policy name)
      - kwargs                  (stringified kwargs or dict)
      - complexity              (numeric policy complexity) [optional]
      - note                    [optional]
      - base_variant_id         (original id before freezing) [optional]
      - Various numeric KPI columns:
          mean_resp_time, p50_resp_time, p90_resp_time,
          coverage_loss_mean, coverage_loss_p90,
          fairness_gap, als_mismatch_rate, unit_area_mismatch_rate,
          plus any numeric columns from summary.csv.

Outputs:
  - --out:
      One row per variant_id, with:
        variant_id, (optional metadata), and *_norm columns for each KPI.
  - --debug-out:
      Same as aggregated table (raw KPI means, n_runs, metadata)
      plus *_norm columns so you can verify scaling.

Usage:
  python scripts/policy_frontier/kpi_aggregate_normalize.py \
    --kpis reference/variant_kpis_full.csv \
    --out reference/variant_metrics_normalized.csv \
    --debug-out reference/variant_metrics_scaled.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# KPI direction configuration
#   - Explicit overrides for known KPIs.
#   - All other numeric columns default to "higher_is_better".
# ----------------------------------------------------------------------
KPI_DIRECTION_OVERRIDES: Dict[str, str] = {
    # Response times: lower is better
    "mean_resp_time": "lower_is_better",
    "p50_resp_time": "lower_is_better",
    "p90_resp_time": "lower_is_better",

    # Coverage loss: lower is better
    "coverage_loss_mean": "lower_is_better",
    "coverage_loss_p90": "lower_is_better",

    # Fairness gap: lower is better (gap between urban/rural)
    "fairness_gap": "lower_is_better",

    # Mismatch rates: lower is better
    "als_mismatch_rate": "lower_is_better",
    "unit_area_mismatch_rate": "lower_is_better",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kpis",
        required=True,
        help="Input CSV with per-run KPIs (e.g., reference/variant_kpis_full.csv)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV with per-variant normalized KPIs",
    )
    ap.add_argument(
        "--debug-out",
        required=True,
        help="Output CSV with aggregated raw KPIs + normalized columns",
    )
    ap.add_argument(
        "--id-col",
        default="variant_id",
        help="Column name to treat as variant identity (default: variant_id)",
    )
    return ap.parse_args()


def normalize_series(s: pd.Series, direction: str) -> pd.Series:
    """Normalize a numeric series to [0,1] given direction."""
    arr = s.to_numpy(dtype=float, copy=True)
    mask = ~np.isnan(arr)
    if not mask.any():
        # all NaN; return as-is
        return pd.Series(arr, index=s.index)

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))

    if vmin == vmax:
        # No variation; assign neutral 0.5 for non-NaN entries
        out = np.full_like(arr, np.nan, dtype=float)
        out[mask] = 0.5
        return pd.Series(out, index=s.index)

    if direction == "lower_is_better":
        out = (vmax - arr) / (vmax - vmin)
    elif direction == "higher_is_better":
        out = (arr - vmin) / (vmax - vmin)
    else:
        raise ValueError(f"Unknown direction '{direction}'")

    return pd.Series(out, index=s.index)


def main():
    args = parse_args()

    kpi_path = Path(args.kpis)
    if not kpi_path.exists():
        raise FileNotFoundError(f"Input KPI file not found: {kpi_path}")

    df = pd.read_csv(kpi_path)
    if df.empty:
        print("WARNING: KPI table is empty. Writing empty outputs.")
        pd.DataFrame().to_csv(args.out, index=False)
        pd.DataFrame().to_csv(args.debug_out, index=False)
        return

    id_col = args.id_col
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in KPI file. Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # 1. Aggregate per variant_id
    # ------------------------------------------------------------------
    # Define which columns are treated as metadata (not normalized)
    meta_cols = [
        id_col,
        "base_variant_id",
        "policy",
        "kwargs",
        "complexity",
        "note",
    ]
    meta_cols_existing: List[str] = [c for c in meta_cols if c in df.columns]

    # Group spec:
    # - For metadata: take the first value per variant (they should be constant).
    # - For numeric KPI columns: take the mean across runs.
    # - Also compute n_runs per variant.
    # Identify numeric columns in the raw df
    numeric_cols_raw = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Exclude things we don't want to normalize/average like run_id
    exclude_from_kpis = {id_col, "run_id", "complexity"}
    kpi_numeric_cols = [c for c in numeric_cols_raw if c not in exclude_from_kpis]

    if not kpi_numeric_cols:
        raise RuntimeError(
            "No numeric KPI columns found to aggregate. "
            f"Numeric cols in file: {numeric_cols_raw}"
        )

    # Build aggregation spec
    agg_spec: Dict[str, str] = {}
    for c in meta_cols_existing:
        if c != id_col:
            agg_spec[c] = "first"
    for c in kpi_numeric_cols:
        agg_spec[c] = "mean"

    grouped = df.groupby(id_col, as_index=False).agg(agg_spec)

    # Add n_runs per variant
    counts = df.groupby(id_col).size().rename("n_runs")
    grouped = grouped.merge(counts, on=id_col, how="left")

    # ------------------------------------------------------------------
    # 2. Normalize KPI columns to [0,1]
    # ------------------------------------------------------------------
    # Recompute numeric KPI columns from the aggregated table,
    # excluding meta and count columns.
    exclude_norm = {
        id_col,
        "base_variant_id",
        "policy",
        "kwargs",
        "complexity",
        "note",
        "n_runs",
    }
    kpi_cols_for_norm = [
        c for c in grouped.columns
        if c not in exclude_norm and pd.api.types.is_numeric_dtype(grouped[c])
    ]

    if not kpi_cols_for_norm:
        raise RuntimeError(
            "No KPI columns found to normalize after aggregation. "
            f"Aggregated columns: {list(grouped.columns)}"
        )

    norm_cols: Dict[str, pd.Series] = {}
    for col in kpi_cols_for_norm:
        direction = KPI_DIRECTION_OVERRIDES.get(col, "higher_is_better")
        norm_col_name = f"{col}_norm"
        norm_cols[norm_col_name] = normalize_series(grouped[col], direction=direction)

    # ------------------------------------------------------------------
    # 3. Build output tables
    # ------------------------------------------------------------------
    # Debug table: aggregated KPIs + normalized columns
    debug_df = grouped.copy()
    for col_name, series in norm_cols.items():
        debug_df[col_name] = series

    # Out table: variant_id (+ optional complexity/meta) + only normalized KPIs
    out_cols = [id_col]
    if "complexity" in grouped.columns:
        out_cols.append("complexity")
    if "n_runs" in grouped.columns:
        out_cols.append("n_runs")

    norm_df = grouped[out_cols].copy()
    for col_name, series in norm_cols.items():
        norm_df[col_name] = series

    # ------------------------------------------------------------------
    # 4. Write outputs
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    debug_out_path = Path(args.debug_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_out_path.parent.mkdir(parents=True, exist_ok=True)

    norm_df.to_csv(out_path, index=False)
    debug_df.to_csv(debug_out_path, index=False)

    # Logging
    n_variants = grouped[id_col].nunique()
    n_rows_raw = len(df)
    print("✔ KPI aggregation and normalization complete.")
    print(f"→ Raw rows (per-run): {n_rows_raw}")
    print(f"→ Aggregated variants: {n_variants}")
    print(f"→ Normalized KPI columns: {kpi_cols_for_norm}")
    print(f"→ Normalized table: {out_path}")
    print(f"→ Debug table (raw+normalized): {debug_out_path}")


if __name__ == "__main__":
    main()