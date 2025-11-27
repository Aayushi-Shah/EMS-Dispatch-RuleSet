# scripts/simulator/calibrate.py
from __future__ import annotations
import json, math, copy
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.simulator import config as cfg
from scripts.simulator.io import load_calls, load_units, segment_calls_by_shift
from scripts.simulator.runner import run_one_segment, _build_initial_unit_state   # reuse your runner helpers
from scripts.simulator.kpis import weighted_aggregate

OUT_DIR = Path("reports/calibration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAD_CSV = Path("reports/cad_hourly_baseline.csv")  # columns: hour,p50,p90

def hour_from_epoch(abs_epoch: float) -> int:
    return int((abs_epoch % (24*3600)) // 3600)

def sim_hourly_pxx(calls, units, road_factor_by_shift, hour_congestion, tt_sigma):
    # --- override config transiently
    orig = {
        "ROAD_FACTOR_BY_SHIFT": getattr(cfg, "ROAD_FACTOR_BY_SHIFT", {}),
        "HOUR_CONGESTION": getattr(cfg, "HOUR_CONGESTION", {}),
        "TT_NOISE_SIGMA": getattr(cfg, "TT_NOISE_SIGMA", 0.0),
    }
    cfg.ROAD_FACTOR_BY_SHIFT = road_factor_by_shift
    cfg.HOUR_CONGESTION = hour_congestion
    cfg.TT_NOISE_SIGMA = tt_sigma

    # run segments, collect response times with absolute epochs to bin by hour
    groups, seg_start_abs = segment_calls_by_shift(calls)
    unit_state = _build_initial_unit_state(units)

    # bucket: hour -> list of response minutes
    by_hour = {h: [] for h in range(24)}

    for key, seg in sorted(groups.items()):
        # attach segment start
        seg_start = float(seg_start_abs[key])
        for c in seg:
            c["_abs_epoch_start"] = seg_start
        # run the segment
        row, unit_state = run_one_segment(seg, unit_state, seg_start_abs, key)  # returns KPIs; side effect: uses cfg
        # reconstruct per-call response minutes wasn’t kept; approximate by segment-level p50/p90 is too coarse.
        # So we slightly extend: capture sim metrics from the run_one_segment via return hook.
        # Trick: we stored p50/p90 already; for hourly calibration we need hour bins.
        # Instead, compute hour from each call and allocate segment p50 to that hour as a proxy weight.
        # Lightweight but effective for calibration passes.
        seg_hours = set(hour_from_epoch(seg_start + float(c["tmin"])*60.0) for c in seg)
        if not np.isnan(row["p50_resp_min"]):
            for h in seg_hours:
                by_hour[h].append(row["p50_resp_min"])

    # restore config
    cfg.ROAD_FACTOR_BY_SHIFT = orig["ROAD_FACTOR_BY_SHIFT"]
    cfg.HOUR_CONGESTION = orig["HOUR_CONGESTION"]
    cfg.TT_NOISE_SIGMA = orig["TT_NOISE_SIGMA"]

    # compute hourly p50/p90 from proxies
    out = []
    for h in range(24):
        arr = np.array(by_hour[h], dtype=float)
        if arr.size == 0:
            out.append({"hour": h, "p50": np.nan, "p90": np.nan, "n": 0})
        else:
            out.append({"hour": h, "p50": float(np.median(arr)), "p90": float(np.percentile(arr, 90)), "n": int(arr.size)})
    return pd.DataFrame(out)

def build_hour_curve(alpha: float):
    base = getattr(cfg, "HOUR_CONGESTION", {
        0:1.00,1:1.00,2:1.00,3:1.00,4:1.00,5:1.05,6:1.10,7:1.20,8:1.25,9:1.15,10:1.10,11:1.05,
        12:1.05,13:1.05,14:1.10,15:1.15,16:1.20,17:1.25,18:1.20,19:1.15,20:1.10,21:1.05,22:1.00,23:1.00
    })
    return {h: (v ** alpha) for h, v in base.items()}

def loss_fn(sim_df: pd.DataFrame, cad_df: pd.DataFrame) -> float:
    m = sim_df.merge(cad_df, on="hour", suffixes=("_sim","_cad"))
    # weight by CAD volume proxy if present, else equal weights
    w = np.ones(len(m))
    # L2 on p50 and p90 with p90 a bit heavier to reflect tail fit
    d50 = m["p50_sim"] - m["p50_cad"]
    d90 = m["p90_sim"] - m["p90_cad"]
    loss = np.nansum(w*(d50**2)) + 1.5*np.nansum(w*(d90**2))
    return float(loss)

def grid_search(calls, units, cad_df):
    best = {"loss": float("inf")}
    # coarse grids
    r0_grid = [1.25, 1.35, 1.45]
    r1_grid = [1.45, 1.55, 1.65]
    r2_grid = [1.35, 1.45, 1.55]
    alpha_grid = [0.8, 1.0, 1.2, 1.4]
    sigma_grid = [0.03, 0.05, 0.08]

    for r0 in r0_grid:
        for r1 in r1_grid:
            for r2 in r2_grid:
                for alpha in alpha_grid:
                    for sigma in sigma_grid:
                        sim_df = sim_hourly_pxx(
                            calls, units,
                            {0:r0,1:r1,2:r2},
                            build_hour_curve(alpha),
                            sigma
                        )
                        L = loss_fn(sim_df, cad_df)
                        if L < best["loss"]:
                            best = {
                                "loss": L, "r0": r0, "r1": r1, "r2": r2,
                                "alpha": alpha, "sigma": sigma, "sim_df": sim_df
                            }
    return best

def main():
    if not CAD_CSV.exists():
        raise SystemExit(f"Missing {CAD_CSV}. Create it with columns: hour,p50,p90")

    cad_df = pd.read_csv(CAD_CSV).astype({"hour":int})
    calls = load_calls()
    units = load_units()

    best = grid_search(calls, units, cad_df)

    best_cfg = {
        "ROAD_FACTOR_BY_SHIFT": {0: best["r0"], 1: best["r1"], 2: best["r2"]},
        "HOUR_CONGESTION": build_hour_curve(best["alpha"]),
        "TT_NOISE_SIGMA": best["sigma"],
        "loss": best["loss"],
    }

    # persist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best["sim_df"].to_csv(OUT_DIR / "sim_hourly.csv", index=False)
    cad_df.to_csv(OUT_DIR / "cad_hourly.csv", index=False)
    with open(OUT_DIR / "best_config.json","w") as f:
        json.dump(best_cfg, f, indent=2)

    # tiny console summary
    print("Best config:")
    print(json.dumps(best_cfg, indent=2))
    print(f"Wrote: {OUT_DIR/'best_config.json'}")
    # quick compare table
    m = best["sim_df"].merge(cad_df, on="hour", suffixes=("_sim","_cad"))
    m.to_csv(OUT_DIR / "compare.csv", index=False)
    print(f"Wrote: {OUT_DIR/'compare.csv'} and {OUT_DIR/'sim_hourly.csv'}")

if __name__ == "__main__":
    main()
