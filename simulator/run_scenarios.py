# analysis/run_scenarios.py
from __future__ import annotations

import argparse
from pathlib import Path

from simulator.runner import run_simulation


# All policies we want to evaluate
POLICIES = [
    "nearest_eta",          # baseline
    "nearest_eta_r1",       # R1: BLS for low-priority
    "nearest_eta_r2",       # R2: protect last ALS in critical muni
    "nearest_eta_r1_r2",
    "nearest_eta_reserve_als",  # R3: reserve ALS for highs
]

# Scenario definitions
# (scenario_name, als_frac, bls_frac, demand_factor)
SCENARIOS = [
    ("S0_baseline",    1.0, 1.0, 1.0),
    ("S1_demand_1.5x", 1.0, 1.0, 1.5),
    ("S2_demand_2x",   1.0, 1.0, 2.0),
    ("S3_supply_0.7x_als", 0.7, 1.0, 1.0),
    ("S4_supply_0.5x_als", 0.5, 1.0, 1.0),
    ("S5_supply_0.7x_bls", 1.0, 0.7, 1.0),
    ("S6_supply_0.5x_bls", 1.0, 0.5, 1.0)
]


def main():
    ap = argparse.ArgumentParser(
        description="Run all EMS policies across all scenarios in batch."
    )
    ap.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory for simulation outputs.",
    )
    ap.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of stochastic repetitions per (policy, scenario).",
    )
    ap.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base RNG seed; per-run seeds are derived from this.",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    for policy in POLICIES:
        for scen_idx, (scenario_name, als_frac, bls_frac, demand_factor) in enumerate(SCENARIOS):
            for rep in range(args.reps):
                seed = args.base_seed + rep

                out_dir = (
                    results_root
                    / policy
                    / scenario_name
                    / f"rep_{rep}"
                )

                print(
                    f"Running policy={policy}, scenario={scenario_name}, "
                    f"rep={rep}, seed={seed}, "
                    f"als_frac={als_frac}, bls_frac={bls_frac}, "
                    f"demand_factor={demand_factor}"
                )

                metrics = run_simulation(
                    policy_name=policy,
                    out_dir=out_dir,
                    seed=seed,
                    als_frac=als_frac,
                    bls_frac=bls_frac,
                    demand_factor=demand_factor,
                    scenario_name=scenario_name,
                )

                print(
                    f"  -> n_calls={metrics['n_calls']}, "
                    f"missed_calls={metrics['missed_calls']}, "
                    f"n_decisions={len(metrics['decisions'])}"
                )


if __name__ == "__main__":
    main()
