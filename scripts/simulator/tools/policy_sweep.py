#!/usr/bin/env python3
from __future__ import annotations
import json, time
import pandas as pd
from pathlib import Path

from .. import config
from ..policies import NearestETA, StationBiasETA, CoverageAwareETA
from ..io import load_calls, load_units, segment_calls_by_shift, load_unit_duty
from ..kpis import weighted_aggregate
from ..logging_utils import new_run_id, write_artifacts
from ..runner import _build_initial_unit_state, run_one_segment  # reuse

def run_with_policy(policy_obj):
    calls = load_calls()
    units = load_units()
    duty_map = load_unit_duty() if getattr(config, "DUTY_ENFORCEMENT", False) else {}
    groups, seg_start_abs = segment_calls_by_shift(calls)

    unit_state = _build_initial_unit_state(units)
    rows = []
    for key, seg in sorted(groups.items()):
        row, unit_state = run_one_segment(seg, unit_state, seg_start_abs, key, duty_map=duty_map, policy_obj=policy_obj)
        row["segment"] = key
        rows.append(row)

    per_shift_df = pd.DataFrame(rows).sort_values("segment")
    summary_df = weighted_aggregate(per_shift_df, units_count=len(units))
    return per_shift_df, summary_df, {"calls": len(calls), "units": len(units), "segments": len(groups)}

def main():
    sweep = []
    # Grid: 3 simple policies with small parameter sweeps
    configs = [
        ("NearestETA", NearestETA()),
        ("StationBiasETA_beta1", StationBiasETA(beta_min=1.0)),
        ("StationBiasETA_beta3", StationBiasETA(beta_min=3.0)),
        ("CoverageAwareETA_g1", CoverageAwareETA(gamma_min=1.0)),
        ("CoverageAwareETA_g3", CoverageAwareETA(gamma_min=3.0)),
    ]

    out_rows = []
    run_id = new_run_id()
    out_dir = config.RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, policy_obj in configs:
        per_shift_df, summary_df, counts = run_with_policy(policy_obj)
        # Save per-policy artifacts
        pdir = out_dir / name
        pdir.mkdir(parents=True, exist_ok=True)
        per_shift_df.to_csv(pdir / "per_shift.csv", index=False)
        summary_df.to_csv(pdir / "summary.csv", index=False)

        # Flatten summary
        s = summary_df.iloc[0].to_dict()
        out_rows.append({
            "policy": name,
            "complexity": getattr(policy_obj, "complexity", lambda: 1.0)(),
            "w_avg_p50_resp_min": s.get("w_avg_p50_resp_min"),
            "w_avg_p90_resp_min": s.get("w_avg_p90_resp_min"),
            "w_avg_avg_resp_min": s.get("w_avg_avg_resp_min"),
            "missed_calls": s.get("missed_calls", 0),
            "calls": counts["calls"],
            "units": counts["units"],
            "segments": counts["segments"],
        })

    sweep_df = pd.DataFrame(out_rows)
    sweep_path = out_dir / "policy_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)

    # Minimal Pareto pick (complexity vs p50 response)
    # keep points that are not dominated on both metrics
    pareto = []
    for i, ri in sweep_df.iterrows():
        dominated = False
        for j, rj in sweep_df.iterrows():
            if j == i: continue
            if (rj["complexity"] <= ri["complexity"]) and (rj["w_avg_p50_resp_min"] <= ri["w_avg_p50_resp_min"]) and \
               ((rj["complexity"] < ri["complexity"]) or (rj["w_avg_p50_resp_min"] < ri["w_avg_p50_resp_min"])):
                dominated = True
                break
        if not dominated:
            pareto.append(ri)
    pareto_df = pd.DataFrame(pareto).sort_values(["complexity", "w_avg_p50_resp_min"])
    pareto_df.to_csv(out_dir / "pareto_frontier.csv", index=False)

    # Write a meta stub so run is traceable
    meta = {
        "run_id": run_id,
        "policies": [c[0] for c in configs],
        "artifacts": {
            "sweep_csv": str(sweep_path),
            "pareto_csv": str(out_dir / "pareto_frontier.csv")
        }
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"✅ Sweep complete → {sweep_path}")
    print(f"✅ Pareto frontier → {out_dir/'pareto_frontier.csv'}")

if __name__ == "__main__":
    main()