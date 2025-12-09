# find_k_tipping_point.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from simulator.runner import run_simulation
from analysis.kpi_analysis import analyze_decisions

# Define the k values to test
K_VALUES = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 17.0, 19.0, 20.0]

# Define the stress scenario
SCENARIO = {
    "name": "S2_demand_2x",
    "als_frac": 1.0,
    "bls_frac": 1.0,
    "demand_factor": 2.0,
}

def main():
    ap = argparse.ArgumentParser(
        description="Find the tipping point for k_minutes in nearest_eta_r1 policy."
    )
    ap.add_argument(
        "--results-root",
        type=str,
        default="results_tipping_point",
        help="Root directory for simulation outputs.",
    )
    ap.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base RNG seed.",
    )
    ap.add_argument(
        "--high-threshold-min",
        type=float,
        default=20.0,
        help="Threshold for high-risk response time.",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    results = []

    # Run baseline nearest_eta
    print("--- Running baseline: nearest_eta ---")
    policy_name = "nearest_eta"
    out_dir = results_root / policy_name / SCENARIO["name"]
    metrics = run_simulation(
        policy_name=policy_name,
        out_dir=out_dir,
        seed=args.base_seed,
        als_frac=SCENARIO["als_frac"],
        bls_frac=SCENARIO["bls_frac"],
        demand_factor=SCENARIO["demand_factor"],
        scenario_name=SCENARIO["name"],
    )
    decisions_path = out_dir / f"decisions_{policy_name}.parquet"
    kpi_summary = analyze_decisions(decisions_path, args.high_threshold_min)
    
    results.append({
        "policy": policy_name,
        "k_minutes": "N/A",
        "high_p90": kpi_summary.get("high_p90"),
        "missed_calls": metrics.get("missed_calls"),
        "als_share_on_low_calls": kpi_summary.get("als_share_on_low_calls"),
        "pct_no_als_global": kpi_summary.get("pct_no_als_global"),
    })
    print(f"  -> high_p90: {kpi_summary.get('high_p90'):.3f}, "
          f"missed_calls: {metrics.get('missed_calls')}, "
          f"als_share_on_low_calls: {kpi_summary.get('als_share_on_low_calls'):.3f}, "
          f"pct_no_als_global: {kpi_summary.get('pct_no_als_global'):.3f}")


    # Run nearest_eta_r1 for each k value
    for k in K_VALUES:
        print(f"--- Running nearest_eta_r1 with k_minutes = {k} ---")
        policy_name = "nearest_eta_r1"
        out_dir = results_root / f"{policy_name}_k{k}" / SCENARIO["name"]
        metrics = run_simulation(
            policy_name=policy_name,
            out_dir=out_dir,
            seed=args.base_seed,
            als_frac=SCENARIO["als_frac"],
            bls_frac=SCENARIO["bls_frac"],
            demand_factor=SCENARIO["demand_factor"],
            scenario_name=SCENARIO["name"],
            k_minutes=k,
        )
        decisions_path = out_dir / f"decisions_{policy_name}.parquet"
        kpi_summary = analyze_decisions(decisions_path, args.high_threshold_min)
        
        results.append({
            "policy": policy_name,
            "k_minutes": k,
            "high_p90": kpi_summary.get("high_p90"),
            "missed_calls": metrics.get("missed_calls"),
            "als_share_on_low_calls": kpi_summary.get("als_share_on_low_calls"),
            "pct_no_als_global": kpi_summary.get("pct_no_als_global"),
        })
        print(f"  -> high_p90: {kpi_summary.get('high_p90'):.3f}, "
              f"missed_calls: {metrics.get('missed_calls')}, "
              f"als_share_on_low_calls: {kpi_summary.get('als_share_on_low_calls'):.3f}, "
              f"pct_no_als_global: {kpi_summary.get('pct_no_als_global'):.3f}")

    print("\n--- Summary ---")
    df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
