#!/usr/bin/env python3
"""
analysis/kpi_pareto.py

Multi-objective Pareto / tradeoff analysis for EMS policies.

Modes:

  1) mode=complexity  (default)
     X = policy complexity (number of rules)
     Y = perf_metric (e.g. high_p90)
     Pareto objectives: [complexity, perf_metric, *extra_metrics]

  2) mode=r1_low_als
     X = als_share_on_low_calls
     Y = high_p90
     Pareto objectives: [als_share_on_low_calls, high_p90]
     Color = complexity
     Infinite-fleet row (policy == 'nearest_eta_infleet'), if present, is shown
     only as a horizontal lower-bound line (not part of the frontier).

  3) mode=r2_strip_muni
     X = pct_low_calls_stripping_muni
     Y = high_p90
     Pareto objectives: [pct_low_calls_stripping_muni, high_p90]
     Color = complexity
     Infinite-fleet row handled the same way as in R1 mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Policy complexity mapping
# ----------------------------------------------------------------------

def add_complexity_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a simple 'complexity' score to each policy:
      - nearest_eta                  -> 0 rules
      - nearest_eta_r1               -> 1 rule (R1)
      - nearest_eta_r2               -> 1 rule (R2)
      - nearest_eta_r3               -> 1 rule (R3)
      - nearest_eta_r1_r2            -> 2 rules (R1+R2)
      - nearest_eta_r1_r3            -> 2 rules (R1+R3)
      - nearest_eta_r2_r3            -> 2 rules (R2+R3)
      - nearest_eta_r1_r2_r3         -> 3 rules (R1+R2+R3)
      - nearest_eta_infleet          -> treated as baseline, complexity NaN
    """
    policy_complexity = {
        "nearest_eta": 0,
        "nearest_eta_r1": 1,
        "nearest_eta_r2": 1,
        "nearest_eta_r3": 1,
        "nearest_eta_r1_r2": 2,
        "nearest_eta_r1_r3": 2,
        "nearest_eta_r2_r3": 2,
        "nearest_eta_r1_r2_r3": 3,
        "nearest_eta_infleet": np.nan,
    }

    if "policy" not in df.columns:
        raise ValueError("Expected a 'policy' column in KPI CSV")

    df = df.copy()
    df["complexity"] = df["policy"].map(policy_complexity)

    if df["complexity"].isna().any():
        missing = df.loc[df["complexity"].isna(), "policy"].unique()
        print(
            "WARNING: No complexity mapping for policies "
            f"(treated as baseline/ignored in Pareto): {missing}"
        )

    return df


# ----------------------------------------------------------------------
# 2. General N-dimensional Pareto (all objectives minimized)
# ----------------------------------------------------------------------

def compute_pareto_min_nd(points: np.ndarray) -> np.ndarray:
    """
    Multi-objective Pareto:

    points: shape (N, D), all D objectives are to be MINIMIZED.

    Returns:
        is_pareto: boolean array of length N, True if point is non-dominated.
    """
    n = points.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        pi = points[i]
        dominates_i = (
            np.all(points <= pi, axis=1) &
            np.any(points < pi, axis=1)
        )
        dominates_i[i] = False

        if np.any(dominates_i):
            is_pareto[i] = False

    return is_pareto


# ----------------------------------------------------------------------
# 3. Plotting – complexity vs perf
# ----------------------------------------------------------------------

def plot_pareto_for_scenario_complexity(
    df_agg: pd.DataFrame,
    scenario: str,
    perf_metric: str = "high_p90",
    extra_metrics: List[str] | None = None,
    coverage_metric_for_color: str | None = "pct_no_idle_units_in_critical_muni_overall",
    out_path: str | None = None,
) -> None:
    """
    For a given scenario:
      - Build multi-objective frontier using:
          [complexity, perf_metric, *extra_metrics]
      - Plot 2D slice: X = complexity, Y = perf_metric
      - Highlight frontier points in red.

    Infinite-fleet policies are dropped from Pareto; you can overlay them
    separately if desired.
    """
    extra_metrics = extra_metrics or []

    sub = df_agg[df_agg["scenario"] == scenario].copy()
    if sub.empty:
        raise ValueError(f"No rows for scenario={scenario} in aggregated KPIs")

    if perf_metric not in sub.columns:
        raise ValueError(f"Performance metric '{perf_metric}' not found in columns")

    if "complexity" not in sub.columns:
        sub = add_complexity_column(sub)

    # Drop any policies with NaN complexity (e.g., nearest_eta_infleet baseline)
    sub = sub.dropna(subset=["complexity"])

    cols_needed = ["complexity", perf_metric] + extra_metrics
    sub = sub.dropna(subset=[c for c in cols_needed if c in sub.columns])
    if sub.empty:
        raise ValueError("No valid rows after dropping NA for objectives")

    objectives = [
        sub["complexity"].to_numpy(dtype=float),
        sub[perf_metric].to_numpy(dtype=float),
    ]
    for m in extra_metrics:
        if m not in sub.columns:
            raise ValueError(f"Extra metric '{m}' not found in columns")
        objectives.append(sub[m].to_numpy(dtype=float))

    pts = np.column_stack(objectives)

    pareto_mask = compute_pareto_min_nd(pts)

    x = pts[:, 0]
    y = pts[:, 1]

    if coverage_metric_for_color and coverage_metric_for_color in sub.columns:
        cvals = sub[coverage_metric_for_color].astype(float)
    else:
        cvals = None

    fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(
        x,
        y,
        c=cvals if cvals is not None else "tab:blue",
        s=60,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.scatter(
        x[pareto_mask],
        y[pareto_mask],
        s=110,
        facecolors="none",
        edgecolors="red",
        linewidth=2.0,
        label="Multi-objective Pareto frontier",
    )

    frontier_df = sub.loc[pareto_mask, ["policy", "complexity", perf_metric]].copy()
    frontier_df = frontier_df.sort_values(by="complexity")
    ax.plot(
        frontier_df["complexity"].to_numpy(),
        frontier_df[perf_metric].to_numpy(),
        linestyle="--",
        color="red",
        alpha=0.7,
    )

    for _, row in sub.iterrows():
        ax.annotate(
            row["policy"],
            (row["complexity"], row[perf_metric]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    ax.set_xlabel("Policy complexity (number of rules)")
    ax.set_ylabel(f"{perf_metric} (minutes) – lower is better")
    title = f"Complexity vs {perf_metric} – scenario: {scenario}"
    if extra_metrics:
        title += f"\nPareto on: complexity + {perf_metric} + {', '.join(extra_metrics)}"
    ax.set_title(title)

    if cvals is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(
            f"{coverage_metric_for_color} (fraction time critical muni uncovered)"
        )

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200)
        print(f"Saved Pareto plot to {out_file}")
    else:
        plt.show()


# ----------------------------------------------------------------------
# 4. R1 tradeoff: ALS share on low calls vs high_p90
# ----------------------------------------------------------------------

def plot_r1_low_als_vs_high_p90(
    df_agg: pd.DataFrame,
    scenario: str,
    out_path: str | None = None,
) -> None:
    """
    R1 view:

      X = als_share_on_low_calls  (ALS share on low/medium-priority calls)
      Y = high_p90                (90th percentile response for high severity)

    Finite-fleet policies define the Pareto frontier.
    nearest_eta_infleet (same scenario) is shown only as a horizontal line.
    """
    sub_all = df_agg[df_agg["scenario"] == scenario].copy()
    if sub_all.empty:
        raise ValueError(f"No rows for scenario={scenario} in aggregated KPIs")

    is_infleet = sub_all["policy"] == "nearest_eta_infleet"
    sub_infleet = sub_all[is_infleet].copy()
    sub = sub_all[~is_infleet].copy()

    for col in ["als_share_on_low_calls", "high_p90"]:
        if col not in sub.columns and (sub_infleet.empty or col not in sub_infleet.columns):
            raise ValueError(f"Column '{col}' not found in KPI CSV")

    if "complexity" not in sub.columns:
        sub = add_complexity_column(sub)

    sub = sub.dropna(subset=["als_share_on_low_calls", "high_p90", "complexity"])
    if sub.empty:
        raise ValueError("No valid finite-fleet rows for R1 tradeoff after dropping NA")

    x = sub["als_share_on_low_calls"].to_numpy(dtype=float)
    y = sub["high_p90"].to_numpy(dtype=float)
    pts = np.column_stack([x, y])

    pareto_mask = compute_pareto_min_nd(pts)

    fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(
        x,
        y,
        c=sub["complexity"].to_numpy(dtype=float),
        cmap="viridis",
        s=60,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.scatter(
        x[pareto_mask],
        y[pareto_mask],
        s=110,
        facecolors="none",
        edgecolors="red",
        linewidth=2.0,
        label="Pareto frontier (ALS share vs high_p90)",
    )

    frontier_df = sub.loc[
        pareto_mask, ["policy", "als_share_on_low_calls", "high_p90"]
    ].copy()
    frontier_df = frontier_df.sort_values(by="als_share_on_low_calls")
    ax.plot(
        frontier_df["als_share_on_low_calls"].to_numpy(),
        frontier_df["high_p90"].to_numpy(),
        linestyle="--",
        color="red",
        alpha=0.7,
    )

    for _, row in sub.iterrows():
        ax.annotate(
            row["policy"],
            (row["als_share_on_low_calls"], row["high_p90"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    # Infinite-fleet lower bound as horizontal line
    if not sub_infleet.empty and "high_p90" in sub_infleet.columns:
        y_inf = float(sub_infleet["high_p90"].iloc[0])
        ax.axhline(
            y_inf,
            linestyle=":",
            color="black",
            linewidth=1.5,
            label="Infinite-fleet lower bound",
        )
        ax.annotate(
            "nearest_eta_infleet",
            (ax.get_xlim()[0], y_inf),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("ALS share on low/medium-priority calls – lower is better")
    ax.set_ylabel("high_p90 (minutes) – lower is better")
    ax.set_title(f"R1 tradeoff: ALS use on low calls vs high_p90 – scenario: {scenario}")

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        location="bottom",
        orientation="horizontal",
        pad=0.15,
        fraction=0.05,
    )
    cbar.set_label("Policy complexity (rules)")

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200)
        print(f"Saved R1 tradeoff plot to {out_file}")
    else:
        plt.show()


# ----------------------------------------------------------------------
# 5. R2 tradeoff: muni stripping on low calls vs high_p90
# ----------------------------------------------------------------------

def plot_r2_strip_muni_vs_high_p90(
    df_agg: pd.DataFrame,
    scenario: str,
    out_path: str | None = None,
) -> None:
    """
    R2 view:

      X = pct_low_calls_stripping_muni   (fraction of low/medium calls
                                          that strip a muni of its last idle unit)
      Y = high_p90

    Finite-fleet policies define the Pareto frontier.
    nearest_eta_infleet (same scenario) is shown only as a horizontal line.
    """
    sub_all = df_agg[df_agg["scenario"] == scenario].copy()
    if sub_all.empty:
        raise ValueError(f"No rows for scenario={scenario} in aggregated KPIs")

    is_infleet = sub_all["policy"] == "nearest_eta_infleet"
    sub_infleet = sub_all[is_infleet].copy()
    sub = sub_all[~is_infleet].copy()

    for col in ["pct_low_calls_stripping_muni", "high_p90"]:
        if col not in sub.columns and (sub_infleet.empty or col not in sub_infleet.columns):
            raise ValueError(f"Column '{col}' not found in KPI CSV")

    if "complexity" not in sub.columns:
        sub = add_complexity_column(sub)

    sub = sub.dropna(subset=["pct_low_calls_stripping_muni", "high_p90", "complexity"])
    if sub.empty:
        raise ValueError("No valid finite-fleet rows for R2 tradeoff after dropping NA")

    x = sub["pct_low_calls_stripping_muni"].to_numpy(dtype=float)
    y = sub["high_p90"].to_numpy(dtype=float)
    pts = np.column_stack([x, y])

    pareto_mask = compute_pareto_min_nd(pts)

    fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(
        x,
        y,
        c=sub["complexity"].to_numpy(dtype=float),
        cmap="viridis",
        s=60,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.scatter(
        x[pareto_mask],
        y[pareto_mask],
        s=110,
        facecolors="none",
        edgecolors="red",
        linewidth=2.0,
        label="Pareto frontier (muni stripping vs high_p90)",
    )

    frontier_df = sub.loc[
        pareto_mask, ["policy", "pct_low_calls_stripping_muni", "high_p90"]
    ].copy()
    frontier_df = frontier_df.sort_values(by="pct_low_calls_stripping_muni")
    ax.plot(
        frontier_df["pct_low_calls_stripping_muni"].to_numpy(),
        frontier_df["high_p90"].to_numpy(),
        linestyle="--",
        color="red",
        alpha=0.7,
    )

    for _, row in sub.iterrows():
        ax.annotate(
            row["policy"],
            (row["pct_low_calls_stripping_muni"], row["high_p90"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    if not sub_infleet.empty and "high_p90" in sub_infleet.columns:
        y_inf = float(sub_infleet["high_p90"].iloc[0])
        ax.axhline(
            y_inf,
            linestyle=":",
            color="black",
            linewidth=1.5,
            label="Infinite-fleet lower bound",
        )
        ax.annotate(
            "nearest_eta_infleet",
            (ax.get_xlim()[0], y_inf),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Pct low/medium calls stripping muni – lower is better")
    ax.set_ylabel("high_p90 (minutes) – lower is better")
    ax.set_title(
        f"R2 tradeoff: muni stripping on low calls vs high_p90 – scenario: {scenario}"
    )

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        location="bottom",
        orientation="horizontal",
        pad=0.15,
        fraction=0.05,
    )
    cbar.set_label("Policy complexity (rules)")

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200)
        print(f"Saved R2 tradeoff plot to {out_file}")
    else:
        plt.show()


# ----------------------------------------------------------------------
# 6. Rebuild kpi_runs_agg from kpi_runs_all
# ----------------------------------------------------------------------

def rebuild_agg_from_runs(
    runs_csv: str,
    agg_out: str,
    group_cols: List[str] | None = None,
) -> None:
    group_cols = group_cols or ["policy", "scenario"]
    df_runs = pd.read_csv(runs_csv)
    numeric_cols = df_runs.select_dtypes(include="number").columns.tolist()

    df_agg = (
        df_runs[group_cols + numeric_cols]
        .groupby(group_cols, as_index=False)
        .mean()
    )

    out_path = Path(agg_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(out_path, index=False)
    print(f"Rebuilt aggregated KPIs to {out_path} (rows={len(df_agg)})")


# ----------------------------------------------------------------------
# 7. CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pareto / tradeoff plots from EMS KPI CSVs."
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="complexity",
        choices=["complexity", "r1_low_als", "r2_strip_muni"],
        help="Plot mode: 'complexity' (default), 'r1_low_als', or 'r2_strip_muni'.",
    )
    ap.add_argument(
        "--agg-csv",
        type=str,
        help="Path to aggregated KPI CSV (e.g. kpi_runs_agg.csv).",
    )
    ap.add_argument(
        "--scenario",
        type=str,
        help="Scenario to filter on (e.g. S2_demand_2x). Required for plotting.",
    )
    ap.add_argument(
        "--perf-metric",
        type=str,
        default="high_p90",
        help="Performance metric column for Y axis in complexity mode.",
    )
    ap.add_argument(
        "--extra-metrics",
        type=str,
        nargs="*",
        default=[],
        help="Extra objectives (minimized) for complexity mode.",
    )
    ap.add_argument(
        "--coverage-metric-for-color",
        type=str,
        default="pct_no_idle_units_in_critical_muni_overall",
        help="Metric used only for coloring points in complexity mode.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. If omitted, shows interactively.",
    )
    ap.add_argument(
        "--rebuild-agg-from",
        type=str,
        default=None,
        help="Optional: path to kpi_runs_all.csv to rebuild aggregated KPIs.",
    )
    ap.add_argument(
        "--agg-out",
        type=str,
        default="kpi_runs_agg_rebuilt.csv",
        help="Output path if using --rebuild-agg-from.",
    )

    args = ap.parse_args()

    if args.rebuild_agg_from:
        rebuild_agg_from_runs(args.rebuild_agg_from, args.agg_out)

    if not args.agg_csv or not args.scenario:
        if args.rebuild_agg_from:
            return
        ap.error("For plotting, you must provide --agg-csv and --scenario")

    agg_path = Path(args.agg_csv)
    if not agg_path.exists():
        raise SystemExit(f"Aggregated KPI CSV not found: {agg_path}")

    df_agg = pd.read_csv(agg_path)
    df_agg = add_complexity_column(df_agg)

    if args.mode == "complexity":
        plot_pareto_for_scenario_complexity(
            df_agg=df_agg,
            scenario=args.scenario,
            perf_metric=args.perf_metric,
            extra_metrics=args.extra_metrics,
            coverage_metric_for_color=args.coverage_metric_for_color,
            out_path=args.out,
        )
    elif args.mode == "r1_low_als":
        plot_r1_low_als_vs_high_p90(
            df_agg=df_agg,
            scenario=args.scenario,
            out_path=args.out,
        )
    elif args.mode == "r2_strip_muni":
        plot_r2_strip_muni_vs_high_p90(
            df_agg=df_agg,
            scenario=args.scenario,
            out_path=args.out,
        )
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()