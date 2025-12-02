#!/usr/bin/env python3
"""
Build Pareto frontier over policy complexity vs performance.

Inputs:
  --metrics: per-variant normalized KPI table
             e.g. reference/variant_kpis_full_normalized.csv

Expected columns:
  - variant_id
  - complexity
  - (optional) n_runs
  - mean_resp_time_norm
  - p50_resp_time_norm
  - p90_resp_time_norm
  - coverage_loss_mean_norm
  - coverage_loss_p90_norm
  - fairness_gap_norm
  - als_mismatch_rate_norm
  - unit_area_mismatch_rate_norm
  - shifts_norm
  - total_calls_norm
  - w_avg_p50_resp_min_norm
  - w_avg_p90_resp_min_norm
  - w_avg_avg_resp_min_norm
  - units_norm

Outputs:
  --out: CSV with one row per variant:
    variant_id, complexity, n_runs (if available),
    performance_score,
    rank (0 = non-dominated frontier),
    is_pareto (True/False),
    plus all *_norm KPI columns.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# KPI weights for performance_score
# All these columns are assumed to be normalized in [0,1] with higher = better.
# Category weights:
#   - Response times (6): 0.4 total
#   - Coverage (2):       0.2 total
#   - Fairness+appropr.:  0.2 total
#   - Resource efficiency (2): 0.2 total
#   - total_calls_norm: 0 (constant / context only)
# ----------------------------------------------------------------------
BASE_METRIC_WEIGHTS: Dict[str, float] = {
    # Response-time behaviour (0.4 total)
    "mean_resp_time_norm":           0.4 / 6.0,
    "p50_resp_time_norm":            0.4 / 6.0,
    "p90_resp_time_norm":            0.4 / 6.0,
    "w_avg_p50_resp_min_norm":       0.4 / 6.0,
    "w_avg_p90_resp_min_norm":       0.4 / 6.0,
    "w_avg_avg_resp_min_norm":       0.4 / 6.0,

    # Coverage (0.2 total)
    "coverage_loss_mean_norm":       0.2 / 2.0,
    "coverage_loss_p90_norm":        0.2 / 2.0,

    # Fairness & appropriateness (0.2 total)
    "fairness_gap_norm":             0.2 / 3.0,
    "als_mismatch_rate_norm":        0.2 / 3.0,
    "unit_area_mismatch_rate_norm":  0.2 / 3.0,

    # Resource efficiency (0.2 total)
    "shifts_norm":                   0.2 / 2.0,
    "units_norm":                    0.2 / 2.0,

    # Volume (likely constant; treat as context → 0)
    "total_calls_norm":              0.0,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        required=True,
        help="Input CSV with per-variant normalized KPIs",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV with Pareto frontier annotations",
    )
    return ap.parse_args()


def compute_performance_score(df: pd.DataFrame) -> pd.Series:
    # Use all *_norm columns that we have weights for and that exist in df
    available_metrics: List[str] = [
        c for c in BASE_METRIC_WEIGHTS.keys() if c in df.columns
    ]
    if not available_metrics:
        raise ValueError(
            "No overlap between BASE_METRIC_WEIGHTS and metrics table columns. "
            f"Table columns: {list(df.columns)}"
        )

    # Extract weights for the available metrics
    raw_weights = {m: BASE_METRIC_WEIGHTS[m] for m in available_metrics}
    total_w = sum(raw_weights.values())
    if total_w <= 0:
        raise ValueError(
            "Sum of weights for available metrics is non-positive. "
            f"Available metrics: {available_metrics}, weights: {raw_weights}"
        )

    # Normalize weights to sum to 1 over the used metrics
    weights = {m: raw_weights[m] / total_w for m in available_metrics}

    score = np.zeros(len(df), dtype=float)
    for m in available_metrics:
        score += weights[m] * df[m].to_numpy(dtype=float)

    return pd.Series(score, index=df.index, name="performance_score")


def pareto_rank(df: pd.DataFrame, perf_col: str, complexity_col: str) -> pd.DataFrame:
    """
    Compute Pareto ranks in 2D:
      - maximize perf_col
      - minimize complexity_col

    rank 0 = non-dominated frontier,
    rank 1 = dominated only by rank 0, etc.
    """

    perf = df[perf_col].to_numpy(dtype=float)
    comp = df[complexity_col].to_numpy(dtype=float)
    n = len(df)

    ranks = np.full(n, -1, dtype=int)
    remaining = np.arange(n)
    current_rank = 0

    while remaining.size > 0:
        frontier = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                # j dominates i if:
                # perf_j >= perf_i and comp_j <= comp_i
                # and at least one strict inequality
                if (
                    perf[j] >= perf[i]
                    and comp[j] <= comp[i]
                    and (perf[j] > perf[i] or comp[j] < comp[i])
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(i)

        frontier = np.array(frontier, dtype=int)
        if frontier.size == 0:
            # Everything left is mutually dominated; assign same rank
            ranks[remaining] = current_rank
            break

        ranks[frontier] = current_rank
        remaining = np.array([i for i in remaining if i not in frontier], dtype=int)
        current_rank += 1

    out = df.copy()
    out["rank"] = ranks
    out["is_pareto"] = out["rank"] == 0
    return out


def main():
    args = parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    if df.empty:
        raise RuntimeError("Metrics table is empty.")

    # Basic required columns
    for col in ["variant_id", "complexity"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from metrics table.")

    # 1) Compute performance_score using all KPIs with defined weights
    df["performance_score"] = compute_performance_score(df)

    # 2) Pareto ranking: maximize performance_score, minimize complexity
    ranked = pareto_rank(df, perf_col="performance_score", complexity_col="complexity")

    # 3) Build output table
    keep_cols: List[str] = ["variant_id", "complexity"]
    if "n_runs" in ranked.columns:
        keep_cols.append("n_runs")
    keep_cols.append("performance_score")
    keep_cols.append("rank")
    keep_cols.append("is_pareto")

    # Also include all *_norm metrics for inspection
    norm_cols = [c for c in ranked.columns if c.endswith("_norm")]
    keep_cols.extend(norm_cols)

    # Deduplicate while preserving order
    seen = set()
    final_keep_cols = []
    for c in keep_cols:
        if c not in seen and c in ranked.columns:
            seen.add(c)
            final_keep_cols.append(c)

    out = ranked[final_keep_cols].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    n_total = len(out)
    n_frontier = int(out["is_pareto"].sum())
    print("✔ Pareto frontier construction complete.")
    print(f"→ Total variants: {n_total}")
    print(f"→ Non-dominated frontier size (rank 0): {n_frontier}")
    print(f"→ Output table: {out_path}")


if __name__ == "__main__":
    main()