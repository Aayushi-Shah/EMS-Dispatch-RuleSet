# experiments/rulelist_policies/kpi_analysis.py

from __future__ import annotations

import argparse
from typing import Dict, Any, List
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core KPI blocks
# ---------------------------------------------------------------------------

def compute_high_risk_response_stats(
    df: pd.DataFrame,
    threshold_min: float,
) -> Dict[str, Any]:
    """
    High-risk = call_is_high == True (boolean from DES).

    Only consider rows where a unit was actually assigned and resp_min is finite.
    Queue delay is added to response to reflect total time from call to arrival.
    """
    if "unit" not in df.columns or "resp_min" not in df.columns:
        return {
            "high_n": 0,
            "high_mean": np.nan,
            "high_p50": np.nan,
            "high_p90": np.nan,
            "high_p99": np.nan,
            "high_pct_over_threshold": np.nan,
        }

    mask_success = df["unit"].notna() & np.isfinite(df["resp_min"])
    df_succ = df.loc[mask_success]

    if "call_is_high" not in df_succ.columns:
        return {
            "high_n": 0,
            "high_mean": np.nan,
            "high_p50": np.nan,
            "high_p90": np.nan,
            "high_p99": np.nan,
            "high_pct_over_threshold": np.nan,
        }

    mask_high = df_succ["call_is_high"] == True  # noqa: E712
    df_high = df_succ.loc[mask_high]

    if df_high.empty:
        return {
            "high_n": 0,
            "high_mean": np.nan,
            "high_p50": np.nan,
            "high_p90": np.nan,
        "high_p99": np.nan,
        "high_pct_over_threshold": np.nan,
    }

    # Total response = queue delay + travel/scene response
    queue_delay = df_high["queue_delay_min"] if "queue_delay_min" in df_high.columns else 0.0
    resp_total = df_high["resp_min"] + queue_delay.fillna(0.0)
    stats = {
        "high_n": int(len(resp_total)),
        "high_mean": float(resp_total.mean()),
        "high_p50": float(resp_total.quantile(0.50)),
        "high_p90": float(resp_total.quantile(0.90)),
        "high_p99": float(resp_total.quantile(0.99)),
        "high_pct_over_threshold": float((resp_total > threshold_min).mean()),
    }
    return stats


def compute_coverage_health(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Coverage-oriented metrics based on idle ALS/BLS counts.
    """
    # Guard against missing columns; return NaNs if not present.
    required = [
        "call_zone",
        "idle_als_total_before",
        "idle_bls_total_before",
        "idle_als_in_call_zone_before",
        "idle_bls_in_call_zone_before",
    ]
    if any(c not in df.columns for c in required):
        return {
            "pct_no_als_global": np.nan,
            "pct_no_bls_global": np.nan,
            "pct_no_als_in_als_zone": np.nan,
            "pct_no_bls_in_bls_zone": np.nan,
        }

    zone = df["call_zone"].astype(str).str.upper()
    n_rows = len(df)

    # Global
    if n_rows > 0:
        mask_no_als_global = df["idle_als_total_before"] == 0
        pct_no_als_global = float(mask_no_als_global.mean())
        mask_no_bls_global = df["idle_bls_total_before"] == 0
        pct_no_bls_global = float(mask_no_bls_global.mean())
    else:
        pct_no_als_global = np.nan
        pct_no_bls_global = np.nan

    # ALS zone
    df_als_zone = df.loc[zone == "ALS"]
    if df_als_zone.empty:
        pct_no_als_in_als_zone = np.nan
    else:
        pct_no_als_in_als_zone = float(
            (df_als_zone["idle_als_in_call_zone_before"] == 0).mean()
        )

    # BLS zone
    df_bls_zone = df.loc[zone == "BLS"]
    if df_bls_zone.empty:
        pct_no_bls_in_bls_zone = np.nan
    else:
        pct_no_bls_in_bls_zone = float(
            (df_bls_zone["idle_bls_in_call_zone_before"] == 0).mean()
        )

    return {
        "pct_no_als_global": pct_no_als_global,
        "pct_no_bls_global": pct_no_bls_global,
        "pct_no_als_in_als_zone": pct_no_als_in_als_zone,
        "pct_no_bls_in_bls_zone": pct_no_bls_in_bls_zone,
    }


def compute_dangerous_last_als(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Dangerous pattern:

      - call_is_low == True
      - idle_als_total_before == 1    → this is the last ALS
      - idle_bls_total_before == 0    → no BLS cushion

    This is where nearestETA is about to send the only ALS to a low-priority call
    while there is no idle BLS left.
    """
    if "call_is_low" not in df.columns:
        return {
            "low_n": 0,
            "danger_last_als_low_n": 0,
            "danger_last_als_low_rate": np.nan,
        }

    mask_low = df["call_is_low"] == True  # noqa: E712
    df_low = df.loc[mask_low]

    if df_low.empty or (
        "idle_als_total_before" not in df_low.columns
        or "idle_bls_total_before" not in df_low.columns
    ):
        return {
            "low_n": int(len(df_low)),
            "danger_last_als_low_n": 0,
            "danger_last_als_low_rate": np.nan,
        }

    mask_danger_global = (
        (df_low["idle_als_total_before"] == 1)
        & (df_low["idle_bls_total_before"] == 0)
    )

    danger_n = int(mask_danger_global.sum())
    low_n = int(len(df_low))

    rate = float(danger_n / low_n) if low_n > 0 else np.nan

    return {
        "low_n": low_n,
        "danger_last_als_low_n": danger_n,
        "danger_last_als_low_rate": rate,
    }


def compute_critical_zone_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Coverage health specifically for critical municipalities.

    We look at decision rows corresponding to calls in critical municipalities
    (if a flag is present) and measure the fraction of rows where there were
    zero idle units available in that critical municipality at dispatch time.

    Expected / optional columns:
      - idle_units_in_critical_muni_before  (int or float)
      - optional call-level flag, one of:
          * call_in_critical_muni
          * call_is_critical_municipality
          * call_is_in_critical_muni
      - optional municipality label:
          * call_municipality_std / call_municipality / call_muni
    """
    # 1) Find the idle-units column we can use.
    idle_col = None
    for c in [
        "idle_units_in_critical_muni_before",
        "idle_units_in_critical_municipality_before",
        "idle_units_in_critical_zone_before",
    ]:
        if c in df.columns:
            idle_col = c
            break

    if idle_col is None:
        print(
            "No critical-municipality idle-units column found; "
            "skipping critical-zone coverage."
        )
        return {
            "pct_no_idle_units_in_critical_muni_overall": np.nan,
        }

    # 2) Determine which rows are “critical-muni calls”.
    crit_flag_col = None
    for c in [
        "call_in_critical_muni",
        "call_is_critical_municipality",
        "call_is_in_critical_muni",
    ]:
        if c in df.columns:
            crit_flag_col = c
            break

    if crit_flag_col is not None:
        mask = df[crit_flag_col] == True  # noqa: E712
    else:
        # Fallback: use all rows where idle_col is not null.
        mask = df[idle_col].notna()

    sub = df[mask].copy()
    n_rows = len(sub)
    if n_rows == 0:
        return {
            "pct_no_idle_units_in_critical_muni_overall": np.nan,
        }

    # 3) Fraction of “critical rows” where there were zero idle units
    frac_uncovered = (sub[idle_col] == 0).mean()

    out: Dict[str, Any] = {
        # fraction in [0,1]; multiply by 100 only when printing if you want %
        "pct_no_idle_units_in_critical_muni_overall": float(frac_uncovered),
    }

    # 4) Optional per-municipality breakdown if label is available.
    muni_col = None
    for c in ["call_municipality_std", "call_municipality", "call_muni"]:
        if c in df.columns:
            muni_col = c
            break

    if muni_col is not None:
        sub["_uncovered"] = (sub[idle_col] == 0)
        per = sub.groupby(muni_col)["_uncovered"].mean()
        for muni, frac in per.items():
            key = f"pct_no_idle_units_in_critical_muni_{muni}"
            out[key] = float(frac)

    return out


def compute_als_usage_on_low_priority(df: pd.DataFrame) -> Dict[str, Any]:
    """
    R1 metric: ALS share on low/medium-priority calls.

      als_share_on_low_calls = fraction of low-priority decisions
                               that used an ALS unit.
    """
    required = ["call_is_low", "unit", "unit_type", "resp_min"]
    if any(c not in df.columns for c in required):
        return {"als_share_on_low_calls": np.nan}

    mask = (
        (df["call_is_low"] == True)  # noqa: E712
        & df["unit"].notna()
        & np.isfinite(df["resp_min"])
    )
    sub = df.loc[mask]
    if sub.empty:
        return {"als_share_on_low_calls": np.nan}

    is_als = sub["unit_type"].astype(str).str.upper() == "ALS"
    share = float(is_als.mean())

    return {"als_share_on_low_calls": share}


def compute_muni_stripping_on_low_priority(df: pd.DataFrame) -> Dict[str, Any]:
    """
    R2 metric: how often low/medium priority calls strip a municipality
    of its last idle unit.

    Requires:
      - call_is_low
      - idle_units_in_call_muni_before (integer)
      - unit, resp_min
    """
    required = ["call_is_low", "idle_units_in_call_muni_before", "unit", "resp_min"]
    if any(c not in df.columns for c in required):
        return {
            "pct_low_calls_stripping_muni": np.nan,
            "low_calls_stripping_muni_n": 0,
            "low_calls_stripping_muni_base_n": 0,
        }

    mask_low_success = (
        (df["call_is_low"] == True)  # noqa: E712
        & df["unit"].notna()
        & np.isfinite(df["resp_min"])
    )
    df_low = df.loc[mask_low_success]
    base_n = int(len(df_low))
    if df_low.empty:
        return {
            "pct_low_calls_stripping_muni": np.nan,
            "low_calls_stripping_muni_n": 0,
            "low_calls_stripping_muni_base_n": 0,
        }

    # "Stripping muni" = before dispatch, there was exactly 1 idle unit in that muni
    mask_strip = df_low["idle_units_in_call_muni_before"] == 1
    n_strip = int(mask_strip.sum())
    rate = float(n_strip / base_n) if base_n > 0 else np.nan

    return {
        "pct_low_calls_stripping_muni": rate,
        "low_calls_stripping_muni_n": n_strip,
        "low_calls_stripping_muni_base_n": base_n,
    }


# ---------------------------------------------------------------------------
# Single-run analysis
# ---------------------------------------------------------------------------

def analyze_decisions(
    decisions_path: Path,
    high_threshold_min: float,
) -> Dict[str, Any]:
    df = pd.read_parquet(decisions_path)

    n_rows = len(df)
    if "unit" in df.columns:
        n_failures = int(df["unit"].isna().sum())
    else:
        n_failures = 0

    # Core KPI blocks
    high_stats = compute_high_risk_response_stats(df, high_threshold_min)
    cov_stats = compute_coverage_health(df)
    crit_cov_stats = compute_critical_zone_coverage(df)
    danger_stats = compute_dangerous_last_als(df)
    als_usage_stats = compute_als_usage_on_low_priority(df)
    muni_strip_stats = compute_muni_stripping_on_low_priority(df)

    # Pack into a single flat dict for CSV
    out: Dict[str, Any] = {
        "decisions_path": str(decisions_path),
        "n_rows": n_rows,
        "n_failures": n_failures,
        "high_threshold_min": high_threshold_min,
    }

    out.update(high_stats)
    out.update(cov_stats)
    out.update(crit_cov_stats)
    out.update(danger_stats)
    out.update(als_usage_stats)
    out.update(muni_strip_stats)
    return out


def print_single_run_summary(summary: Dict[str, Any]) -> None:
    """
    Pretty console summary for a single decisions file.
    """
    print("\n=== KPI SUMMARY ===")
    print(f"decisions_path           : {summary.get('decisions_path')}")
    print(f"rows (decisions)         : {summary.get('n_rows')}")
    print(f"failures (no unit)       : {summary.get('n_failures')}")
    print(f"high-risk threshold (min): {summary.get('high_threshold_min')}")
    print()

    # High-risk block
    high_n = summary.get("high_n", 0)
    print("High-risk response (call_is_high == True):")
    print(f"  n_high                 : {high_n}")
    if high_n > 0:
        print(f"  mean_resp_high         : {summary['high_mean']:.3f}")
        print(f"  p50_resp_high          : {summary['high_p50']:.3f}")
        print(f"  p90_resp_high          : {summary['high_p90']:.3f}")
        print(f"  p99_resp_high          : {summary['high_p99']:.3f}")
        print(
            f"  pct_high_resp_over_T   : {summary['high_pct_over_threshold']*100:.2f}%"
        )
    else:
        print("  mean_resp_high         : NA")
        print("  p50_resp_high          : NA")
        print("  p90_resp_high          : NA")
        print("  p99_resp_high          : NA")
        print("  pct_high_resp_over_T   : NA")
    print()

    # Coverage health
    print("Coverage health:")
    for key, label in [
        ("pct_no_als_global", "pct_no_als_global"),
        ("pct_no_bls_global", "pct_no_bls_global"),
        ("pct_no_als_in_als_zone", "pct_no_als_in_als_zone"),
        ("pct_no_bls_in_bls_zone", "pct_no_bls_in_bls_zone"),
    ]:
        val = summary.get(key, np.nan)
        if isinstance(val, (int, float)) and not np.isnan(val):
            print(f"  {label:24}: {val*100:.2f}%")
        else:
            print(f"  {label:24}: NA")
    print()

    # Dangerous last ALS on low priority
    low_n = summary.get("low_n", 0)
    print("Dangerous 'last ALS on low priority' (no BLS cushion):")
    print(f"  low_n                  : {low_n}")
    if low_n > 0:
        print(f"  danger_last_als_low_n  : {summary.get('danger_last_als_low_n')}")
        rate = summary.get("danger_last_als_low_rate")
        if rate is not None and not np.isnan(rate):
            print(f"  danger_last_als_low_rate: {rate*100:.2f}%")
        else:
            print("  danger_last_als_low_rate: NA")
    else:
        print("  danger_last_als_low_n  : NA")
        print("  danger_last_als_low_rate: NA")
    print()

    # Critical muni coverage (just top-level metric)
    crit = summary.get("pct_no_idle_units_in_critical_muni_overall", np.nan)
    print("Critical-municipality coverage:")
    if isinstance(crit, (int, float)) and not np.isnan(crit):
        print(
            f"  pct_no_idle_units_in_critical_muni_overall: {crit*100:.2f}%"
        )
    else:
        print("  pct_no_idle_units_in_critical_muni_overall: NA")
    print()

    # R1: ALS share on low/medium-priority calls
    als_share = summary.get("als_share_on_low_calls", np.nan)
    print("ALS usage on low/medium-priority calls:")
    if isinstance(als_share, (int, float)) and not np.isnan(als_share):
        print(f"  als_share_on_low_calls : {als_share*100:.2f}%")
    else:
        print("  als_share_on_low_calls : NA")
    print()

    # R2: muni stripping on low/medium calls
    strip_pct = summary.get("pct_low_calls_stripping_muni", np.nan)
    print("Muni stripping on low/medium-priority calls:")
    base_n = summary.get("low_calls_stripping_muni_base_n", 0)
    n_strip = summary.get("low_calls_stripping_muni_n", 0)
    print(f"  base_n_low_calls       : {base_n}")
    print(f"  stripping_events_n     : {n_strip}")
    if isinstance(strip_pct, (int, float)) and not np.isnan(strip_pct):
        print(f"  pct_low_calls_stripping_muni: {strip_pct*100:.2f}%")
    else:
        print("  pct_low_calls_stripping_muni: NA")
    print()


# ---------------------------------------------------------------------------
# Batch KPI runner
# ---------------------------------------------------------------------------

def _iter_decision_files(results_root: Path) -> List[Path]:
    """
    Find all decisions_*.parquet files under results_root.
    Layout assumed:
        results/{policy}/{scenario}/rep_*/decisions_*.parquet  (legacy)
    or:
        results/{policy}/{scenario}/decisions_*.parquet        (no reps)
    """
    return sorted(results_root.rglob("decisions_*.parquet"))


def _infer_ids(decisions_path: Path) -> Dict[str, str]:
    """
    Infer policy / scenario / rep from paths like:
      - results/{policy}/{scenario}/rep_*/decisions_*.parquet (legacy)
      - results/{policy}/{scenario}/decisions_*.parquet       (no reps)
    """
    run_dir = decisions_path.parent
    scenario_dir = run_dir
    if run_dir.name.startswith("rep_"):
        rep = run_dir.name
        scenario_dir = run_dir.parent
    else:
        rep = "single"

    try:
        scenario = scenario_dir.name
        policy = scenario_dir.parent.name
    except Exception:
        scenario = "unknown_scenario"
        policy = "unknown_policy"

    return {
        "policy": policy,
        "scenario": scenario,
        "rep": rep,
    }


def run_batch_kpi_analysis(
    results_root: str = "results",
    high_threshold_min: float = 20.0,
    out_runs_csv: str = "kpi_runs_all.csv",
    out_agg_csv: str = "kpi_runs_agg.csv",
) -> None:
    """
    Batch driver:
      - walks all decisions_*.parquet under results_root,
      - calls analyze_decisions() on each,
      - writes:
          * out_runs_csv:    one row per (policy, scenario, rep)
          * out_agg_csv:     averaged over reps per (policy, scenario)
    """
    root = Path(results_root)
    if not root.exists():
        raise SystemExit(f"results_root not found: {root}")

    decision_files = _iter_decision_files(root)
    if not decision_files:
        raise SystemExit(f"No decisions_*.parquet files found under {root}")

    rows: List[Dict[str, Any]] = []
    print(f"Found {len(decision_files)} decisions_*.parquet files under {root}")

    for path in decision_files:
        ids = _infer_ids(path)
        print(
            f"Analyzing {path} "
            f"(policy={ids['policy']}, scenario={ids['scenario']}, rep={ids['rep']})"
        )

        try:
            summary = analyze_decisions(path, high_threshold_min=high_threshold_min)
        except Exception as e:
            print(f"  !! ERROR analyzing {path}: {e}")
            continue

        # Attach ids so you can group/compare later
        summary.update(ids)
        rows.append(summary)

    if not rows:
        raise SystemExit("No KPI rows collected; all analyses failed?")

    df_runs = pd.DataFrame(rows)

    out_runs = Path(out_runs_csv)
    df_runs.to_csv(out_runs, index=False)
    print(f"✅ Wrote per-run KPIs to {out_runs} (rows={len(df_runs)})")

    # Aggregate across reps: group by (policy, scenario) and average numeric cols
    numeric_cols = df_runs.select_dtypes(include="number").columns.tolist()
    group_cols = ["policy", "scenario"]

    df_agg = (
        df_runs[group_cols + numeric_cols]
        .groupby(group_cols, as_index=False)
        .mean()
    )

    out_agg = Path(out_agg_csv)
    df_agg.to_csv(out_agg, index=False)
    print(f"✅ Wrote aggregated KPIs to {out_agg} (rows={len(df_agg)})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="KPI analysis for EMS dispatch experiments."
    )
    ap.add_argument(
        "--decisions",
        type=str,
        help="Path to a single decisions_*.parquet file (single-run mode).",
    )
    ap.add_argument(
        "--batch",
        action="store_true",
        help="Run batch KPI analysis over results/{policy}/{scenario}/rep_*.",
    )
    ap.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory for batch mode (default: results).",
    )
    ap.add_argument(
        "--high-threshold-min",
        type=float,
        default=20.0,
        help="Threshold (minutes) for 'very late' high-priority responses (default: 20.0).",
    )
    ap.add_argument(
        "--out-runs-csv",
        type=str,
        default="kpi_runs_all.csv",
        help="Per-run KPI CSV output for batch mode (default: kpi_runs_all.csv).",
    )
    ap.add_argument(
        "--out-agg-csv",
        type=str,
        default="kpi_runs_agg.csv",
        help="Aggregated KPI CSV output for batch mode (default: kpi_runs_agg.csv).",
    )
    args = ap.parse_args()

    if args.batch:
        run_batch_kpi_analysis(
            results_root=args.results_root,
            high_threshold_min=args.high_threshold_min,
            out_runs_csv=args.out_runs_csv,
            out_agg_csv=args.out_agg_csv,
        )
    elif args.decisions:
        summary = analyze_decisions(
            Path(args.decisions),
            high_threshold_min=args.high_threshold_min,
        )
        print_single_run_summary(summary)
    else:
        ap.error("Specify either --decisions <file> or --batch")
