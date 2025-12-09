# simulator/run_infleet_baselines.py

from __future__ import annotations

from pathlib import Path
import argparse

from .runner import run_simulation

SCENARIOS = [
    {"scenario": "S0_baseline",       "demand_factor": 1.0},
    {"scenario": "S1_demand_1.5x",    "demand_factor": 1.5},
    {"scenario": "S2_demand_2x",      "demand_factor": 2.0},
    {"scenario": "S3_supply_0.7x",    "demand_factor": 1.0},
    {"scenario": "S4_supply_0.5x",    "demand_factor": 1.0},
]


def main():
    ap = argparse.ArgumentParser(
        description="Run infinite-fleet baselines (nearest_eta) for all scenarios."
    )
    ap.add_argument(
        "--base-out",
        type=str,
        default="results",
        help="Base output directory (default: results)",
    )
    ap.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of repetitions per scenario (default: 3)",
    )
    ap.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed; rep index will be added to get per-rep seed.",
    )
    ap.add_argument(
        "--fleet-factor",
        type=int,
        default=20,  # or 50 if you want “more infinite”
        help="Replication factor for infinite-fleet baseline (default: 20).",
    )
    args = ap.parse_args()

    base_out = Path(args.base_out)

    for scen in SCENARIOS:
        scen_name = scen["scenario"]
        demand_factor = scen["demand_factor"]

        for r in range(args.reps):
            seed = args.base_seed + r

            out_dir = (
                base_out
                / "nearest_eta_infleet"
                / scen_name
                / f"rep_{r}"
            )

            print(
                f"[infleet] scenario={scen_name}, rep={r}, "
                f"demand_factor={demand_factor}, seed={seed}, out_dir={out_dir}, "
                f"fleet_factor={args.fleet_factor}"
            )

            run_simulation(
                policy_name="nearest_eta",        # logic
                out_dir=out_dir,
                seed=seed,
                als_frac=1.0,                     # NO SUPPLY CUT
                bls_frac=1.0,
                demand_factor=demand_factor,
                scenario_name=scen_name,
                fleet_factor=args.fleet_factor,   # THIS is the “infinite” part
            )


if __name__ == "__main__":
    main()