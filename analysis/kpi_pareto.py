#!/usr/bin/env python3
"""
generate_all_paretos_from_runs.py

From kpi_runs_all.csv, rebuild aggregated KPIs and generate, for every scenario:

  - R1: als_share_on_low_calls vs high_p90
  - R2: pct_low_calls_stripping_muni vs high_p90

Outputs:
  out_dir/pareto_<scenario>_r1_low_als.png
  out_dir/pareto_<scenario>_r2_strip_muni.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Policy complexity mapping (same as your kpi_pareto.py)
# ----------------------------------------------------------------------

def add_complexity_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a simple 'complexity' score to each policy:
      - nearest_eta                  -> 0 rules
      - nearest_eta_r1               -> 1 rule (R1)
      - nearest_eta_r2               -> 1 rule (R2)
      - nearest_eta_r1_r2            -> 2 rules (R1+R2)
      - nearest_eta_infleet          -> treated as baseline, complexity NaN
    """
    policy_complexity = {
        "nearest_eta": 0,
        "nearest_eta_r1": 1,
        "nearest_eta_r2": 1,
        "nearest_eta_r1_r2": 2,
        "nearest_eta_reserve_als": 1,
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
# 2. General 2D Pareto (all objectives minimized)
# ----------------------------------------------------------------------

def compute_pareto_min_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    2D Pareto frontier (minimize both x and y).
    Returns a boolean mask of non-dominated points.
    """
    pts = np.column_stack([x, y])
    n = pts.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        pi = pts[i]
        dominates_i = (
            np.all(pts <= pi, axis=1) &
            np.any(pts < pi, axis=1)
        )
        dominates_i[i] = False

        if np.any(dominates_i):
            is_pareto[i] = False

    return is_pareto


# ----------------------------------------------------------------------
# 3. R1 tradeoff: als_share_on_low_calls vs high_p90
# ----------------------------------------------------------------------

def plot_r1_low_als_vs_high_p90(
    df_agg: pd.DataFrame,
    scenario: str,
    out_path: Path,
) -> None:
    """
    R1 view:

      X = als_share_on_low_calls  (ALS share on low/medium-priority calls)
      Y = high_p90                (90th percentile response for high severity)

    Finite-fleet policies define the Pareto frontier.
    nearest_eta_infleet is shown only as a horizontal line.
    """
    sub_all = df_agg[df_agg["scenario"] == scenario].copy()
    if sub_all.empty:
        print(f"[R1] No rows for scenario={scenario}, skipping.")
        return

    is_infleet = sub_all["policy"] == "nearest_eta_infleet"
    sub_infleet = sub_all[is_infleet].copy()
    sub = sub_all[~is_infleet].copy()

    for col in ["als_share_on_low_calls", "high_p90"]:
        if col not in sub.columns and (sub_infleet.empty or col not in sub_infleet.columns):
            print(f"[R1] Column '{col}' not found for scenario={scenario}, skipping.")
            return

    if "complexity" not in sub.columns:
        sub = add_complexity_column(sub)

    sub = sub.dropna(subset=["als_share_on_low_calls", "high_p90", "complexity"])
    if sub.empty:
        print(f"[R1] No valid finite-fleet rows for scenario={scenario}, skipping.")
        return

    x = sub["als_share_on_low_calls"].to_numpy(dtype=float)
    y = sub["high_p90"].to_numpy(dtype=float)

    pareto_mask = compute_pareto_min_2d(x, y)

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
        xmin, xmax = ax.get_xlim()
        ax.axhline(
            y_inf,
            linestyle=":",
            color="black",
            linewidth=1.5,
            label="Infinite-fleet lower bound",
        )
        ax.annotate(
            "nearest_eta_infleet",
            (xmin, y_inf),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
        )
        ax.set_xlim(xmin, xmax)

    ax.set_xlabel("ALS share on low/medium-priority calls – lower is better")
    ax.set_ylabel("high_p90 (minutes) – lower is better")
    ax.set_title(f"R1 tradeoff: ALS use on low calls vs high_p90 – {scenario}")

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[R1] Saved {out_path}")


# ----------------------------------------------------------------------
# 4. R2 tradeoff: pct_low_calls_stripping_muni vs high_p90
# ----------------------------------------------------------------------

def plot_r2_strip_muni_vs_high_p90(
    df_agg: pd.DataFrame,
    scenario: str,
    out_path: Path,
) -> None:
    """
    R2 view:

      X = pct_low_calls_stripping_muni   (fraction of low/medium calls
                                          that strip a muni of its last idle unit)
      Y = high_p90

    Finite-fleet policies define the Pareto frontier.
    nearest_eta_infleet is shown only as a horizontal line.
    """
    sub_all = df_agg[df_agg["scenario"] == scenario].copy()
    if sub_all.empty:
        print(f"[R2] No rows for scenario={scenario}, skipping.")
        return

    is_infleet = sub_all["policy"] == "nearest_eta_infleet"
    sub_infleet = sub_all[is_infleet].copy()
    sub = sub_all[~is_infleet].copy()

    for col in ["pct_low_calls_stripping_muni", "high_p90"]:
        if col not in sub.columns and (sub_infleet.empty or col not in sub_infleet.columns):
            print(f"[R2] Column '{col}' not found for scenario={scenario}, skipping.")
            return

    if "complexity" not in sub.columns:
        sub = add_complexity_column(sub)

    sub = sub.dropna(subset=["pct_low_calls_stripping_muni", "high_p90", "complexity"])
    if sub.empty:
        print(f"[R2] No valid finite-fleet rows for scenario={scenario}, skipping.")
        return

    x = sub["pct_low_calls_stripping_muni"].to_numpy(dtype=float)
    y = sub["high_p90"].to_numpy(dtype=float)

    pareto_mask = compute_pareto_min_2d(x, y)

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

    # Infinite-fleet lower bound as horizontal line (same as R1)
    if not sub_infleet.empty and "high_p90" in sub_infleet.columns:
        y_inf = float(sub_infleet["high_p90"].iloc[0])
        xmin, xmax = ax.get_xlim()
        ax.axhline(
            y_inf,
            linestyle=":",
            color="black",
            linewidth=1.5,
            label="Infinite-fleet lower bound",
        )
        ax.annotate(
            "nearest_eta_infleet",
            (xmin, y_inf),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
        )
        ax.set_xlim(xmin, xmax)

    ax.set_xlabel("Pct low/medium calls stripping muni – lower is better")
    ax.set_ylabel("high_p90 (minutes) – lower is better")
    ax.set_title(
        f"R2 tradeoff: muni stripping on low calls vs high_p90 – {scenario}"
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[R2] Saved {out_path}")


# ----------------------------------------------------------------------
# 4b. R3 view: n_failures vs high_p90 (both minimized)
# ----------------------------------------------------------------------

def plot_failures_vs_high_p90(
    df_agg: pd.DataFrame,
    scenario: str,
    out_path: Path,
) -> None:
    """
    R3 view:
      X = n_failures (per policy/scenario average)
      Y = high_p90 (queue-inclusive)
    """
    sub_all = df_agg[df_agg["scenario"] == scenario].copy()
    if sub_all.empty:
        print(f"[R3] No rows for scenario={scenario}, skipping.")
        return

    for col in ["n_failures", "high_p90"]:
        if col not in sub_all.columns:
            print(f"[R3] Column '{col}' not found for scenario={scenario}, skipping.")
            return

    is_infleet = sub_all["policy"] == "nearest_eta_infleet"
    sub_infleet = sub_all[is_infleet].copy()
    sub = sub_all[~is_infleet].copy()

    if "complexity" not in sub.columns:
        sub = add_complexity_column(sub)

    sub = sub.dropna(subset=["n_failures", "high_p90", "complexity"])
    if sub.empty:
        print(f"[R3] No valid finite rows for scenario={scenario}, skipping.")
        return

    x = sub["n_failures"].to_numpy(dtype=float)
    y = sub["high_p90"].to_numpy(dtype=float)

    pareto_mask = compute_pareto_min_2d(x, y)

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
        label="Pareto frontier (failures vs high_p90)",
    )

    frontier_df = sub.loc[pareto_mask, ["policy", "n_failures", "high_p90"]].copy()
    frontier_df = frontier_df.sort_values(by="n_failures")
    ax.plot(
        frontier_df["n_failures"].to_numpy(),
        frontier_df["high_p90"].to_numpy(),
        linestyle="--",
        color="red",
        alpha=0.7,
    )

    for _, row in sub.iterrows():
        ax.annotate(
            row["policy"],
            (row["n_failures"], row["high_p90"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    # Infinite-fleet lower bound as horizontal line
    if not sub_infleet.empty and "high_p90" in sub_infleet.columns:
        y_inf = float(sub_infleet["high_p90"].iloc[0])
        xmin, xmax = ax.get_xlim()
        ax.axhline(
            y_inf,
            linestyle=":",
            color="black",
            linewidth=1.5,
            label="Infinite-fleet lower bound",
        )
        ax.annotate(
            "nearest_eta_infleet",
            (xmin, y_inf),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
        )
        ax.set_xlim(xmin, xmax)

    ax.set_xlabel("Failures (calls not served) – lower is better")
    ax.set_ylabel("high_p90 (minutes, incl. queue) – lower is better")
    ax.set_title(f"R3: failures vs high_p90 – {scenario}")

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[R3] Saved {out_path}")

# ----------------------------------------------------------------------
# 5. Rebuild agg from kpi_runs_all and drive all scenarios
# ----------------------------------------------------------------------

def rebuild_agg_from_runs(
    runs_csv: str,
    group_cols: List[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or ["policy", "scenario"]
    df_runs = pd.read_csv(runs_csv)
    numeric_cols = df_runs.select_dtypes(include="number").columns.tolist()

    df_agg = (
        df_runs[group_cols + numeric_cols]
        .groupby(group_cols, as_index=False)
        .mean()
    )

    return df_agg


def main():
    ap = argparse.ArgumentParser(
        description="Generate R1/R2 Pareto plots for all scenarios from kpi_runs_all.csv"
    )
    ap.add_argument(
        "--runs-csv",
        type=str,
        required=True,
        help="Path to kpi_runs_all.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="pareto_plots",
        help="Output directory for Pareto PNGs (default: pareto_plots)",
    )
    args = ap.parse_args()

    runs_path = Path(args.runs_csv)
    if not runs_path.exists():
        raise SystemExit(f"kpi_runs_all.csv not found: {runs_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_agg = rebuild_agg_from_runs(str(runs_path))
    df_agg = add_complexity_column(df_agg)

    if "scenario" not in df_agg.columns:
        raise SystemExit("Expected 'scenario' column in aggregated KPIs")

    scenarios = sorted(df_agg["scenario"].unique())
    print(f"Found scenarios: {scenarios}")

    for scen in scenarios:
        r1_path = out_dir / f"pareto_{scen}_r1_low_als.png"
        r2_path = out_dir / f"pareto_{scen}_r2_strip_muni.png"
        r3_path = out_dir / f"pareto_{scen}_r3_failures_high.png"

        plot_r1_low_als_vs_high_p90(df_agg, scen, r1_path)
        plot_r2_strip_muni_vs_high_p90(df_agg, scen, r2_path)
        plot_failures_vs_high_p90(df_agg, scen, r3_path)


if __name__ == "__main__":
    main()
