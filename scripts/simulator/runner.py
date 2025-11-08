# scripts/simulator/runner.py
from __future__ import annotations

import random
import numpy as np
import pandas as pd

from . import config, traffic
from .des import DES, Unit
from .policies import nearest_unit_policy
from .io import load_calls, load_units, segment_calls_by_shift
from .kpis import weighted_aggregate
from .logging_utils import new_run_id, write_artifacts


# ---------- small local diag helpers ----------
def _hav_miles(lon1, lat1, lon2, lat2) -> float:
    import math
    R = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------- segment runner ----------
def _build_initial_unit_state(units: list[Unit]) -> dict[str, dict]:
    return {
        u.name: {
            "abs_free_epoch": -float("inf"),
            "lon": u.lon, "lat": u.lat,
            "station_lon": u.station_lon, "station_lat": u.station_lat,
            "utype": u.utype, "station": u.station,
        }
        for u in units
    }


def run_one_segment(
    calls_segment: list[dict],
    unit_state: dict[str, dict],
    seg_start_abs_map: dict[str, float],
    seg_key: str,
) -> tuple[dict, dict]:
    """Run one shift segment and return (row_kpis, updated_unit_state)."""
    traffic.reset_stats()

    sim = DES(select_unit_fn=nearest_unit_policy)
    new_units: list[Unit] = []

    seg_start_abs = float(seg_start_abs_map[seg_key])

    # materialize units for this segment
    for name, s in unit_state.items():
        remaining_min = max(0.0, (s["abs_free_epoch"] - seg_start_abs) / 60.0)
        if remaining_min > 0:
            lon, lat = s["lon"], s["lat"]
        else:
            lon, lat = s["station_lon"], s["station_lat"]

        u = Unit(
            name=name,
            utype=s["utype"],
            station=s["station"],
            lon=lon, lat=lat,
            station_lon=s["station_lon"], station_lat=s["station_lat"],
            busy_until=(remaining_min if remaining_min > 0 else 0.0),
        )
        new_units.append(u)
        sim.add_unit(u)
        if u.busy_until > 0.0:
            sim.schedule(u.busy_until, "unit_free", unit=u, end_lon=u.lon, end_lat=u.lat)

    # attach segment absolute start and schedule calls
    for c in calls_segment:
        c["_abs_epoch_start"] = seg_start_abs
        sim.schedule(c["tmin"], "call", **c)

    while sim.advance():
        pass

    # updated unit state to absolute epochs
    updated_state = {}
    for u in new_units:
        abs_free = seg_start_abs + max(u.busy_until, 0.0) * 60.0
        updated_state[u.name] = {
            "abs_free_epoch": abs_free,
            "lon": u.lon,
            "lat": u.lat,
            "station_lon": u.station_lon,
            "station_lat": u.station_lat,
            "utype": u.utype,
            "station": u.station,
        }

    # KPIs
    resp = np.array(sim.metrics["resp_times"]) if sim.metrics["resp_times"] else np.array([])
    row = {
        "n_calls": sim.metrics["n_calls"],
        "missed_calls": sim.metrics["missed_calls"],
        "p50_resp_min": np.percentile(resp, 50) if len(resp) else np.nan,
        "p90_resp_min": np.percentile(resp, 90) if len(resp) else np.nan,
        "avg_resp_min": float(resp.mean()) if len(resp) else np.nan,
        "p50_wait_min": np.percentile(sim.metrics["wait_minutes"], 50) if sim.metrics["wait_minutes"] else np.nan,
        "p90_wait_min": np.percentile(sim.metrics["wait_minutes"], 90) if sim.metrics["wait_minutes"] else np.nan,
        "avg_onscene_min": np.mean(sim.metrics["on_scene"]) if sim.metrics["on_scene"] else np.nan,
        "avg_transport_min": np.mean(sim.metrics["transport"]) if sim.metrics["transport"] else np.nan,
        "avg_turnaround_min": np.mean(sim.metrics["turnaround"]) if sim.metrics["turnaround"] else np.nan,
        "units": len(new_units),
    }

    # per-segment traffic diagnostics
    tstats = traffic.snapshot_stats()
    row.update({
        "legs": tstats["legs"],
        "avg_road_factor": tstats["avg_road_factor"],
        "avg_hour_mult": tstats["avg_hour_mult"],
        "avg_zone_mult": tstats["avg_zone_mult"],
    })

    return row, updated_state


# ---------- main ----------
def main():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    calls = load_calls()
    units = load_units()

    print(f"[CFG] SCENE_SPEED_MPH={config.SCENE_SPEED_MPH} "
          f"HOSPITAL_SPEED_MPH={config.HOSPITAL_SPEED_MPH} "
          f"MAX_QUEUE_RETRIES={config.MAX_QUEUE_RETRIES}")

    # diagnostics: nearest-station distances for first 1000 calls
    calls_df = pd.DataFrame(calls)
    units_df = pd.DataFrame([{"lon": u.lon, "lat": u.lat} for u in units])
    if len(calls_df) and len(units_df):
        u_mat = units_df[["lon", "lat"]].to_numpy()
        c_mat = calls_df.head(1000)[["lon", "lat"]].to_numpy()
        dists = []
        for cl in c_mat:
            d = np.min([_hav_miles(u[0], u[1], cl[0], cl[1]) for u in u_mat])
            dists.append(d)
        if dists:
            dists = np.array(dists)
            print(f"[DIAG] nearest-station miles p50={np.percentile(dists,50):.2f} "
                  f"p90={np.percentile(dists,90):.2f}")
            est_p50 = 60.0 * np.percentile(dists, 50) / max(config.SCENE_SPEED_MPH, 1e-6)
            est_p90 = 60.0 * np.percentile(dists, 90) / max(config.SCENE_SPEED_MPH, 1e-6)
            print(f"[DIAG] naive resp est p50={est_p50:.2f} min p90={est_p90:.2f} min")

    groups, seg_start_abs = segment_calls_by_shift(calls)

    unit_state = _build_initial_unit_state(units)
    rows = []
    for key, seg in sorted(groups.items()):
        traffic.reset_stats()
        row, unit_state = run_one_segment(seg, unit_state, seg_start_abs, key)
        # already contains per-segment traffic stats
        row["segment"] = key
        rows.append(row)

    per_shift_df = pd.DataFrame(rows).sort_values("segment")

    # optional aggregate traffic diag from per-shift rows
    if "legs" in per_shift_df.columns and per_shift_df["legs"].sum() > 0:
        L = per_shift_df["legs"].sum()
        agg_avg_road = (per_shift_df["avg_road_factor"] * per_shift_df["legs"]).sum() / L
        agg_avg_hour = (per_shift_df["avg_hour_mult"] * per_shift_df["legs"]).sum() / L
        agg_avg_zone = (per_shift_df["avg_zone_mult"] * per_shift_df["legs"]).sum() / L
        print(f"[TRAFFIC][aggregate] legs={int(L)} avg_road={agg_avg_road:.3f} "
              f"avg_hour={agg_avg_hour:.3f} avg_zone={agg_avg_zone:.3f}")

    summary_df = weighted_aggregate(per_shift_df, units_count=len(units))

    # write artifacts (this creates the CSVs and returns their paths)
    run_id = new_run_id()
    meta = {
        "run_id": run_id,
        "config": {
            "SHIFT_WINDOWS": config.SHIFT_WINDOWS,
            "SEGMENT_BY_SHIFT": config.SEGMENT_BY_SHIFT,
            "RANDOM_SEED": config.RANDOM_SEED,
            "SCENE_SPEED_MPH": config.SCENE_SPEED_MPH,
            "HOSPITAL_SPEED_MPH": config.HOSPITAL_SPEED_MPH,
            "DISPATCH_DELAY_MIN": config.DISPATCH_DELAY_MIN,
            "ONSCENE_MIN": config.ONSCENE_MIN,
            "ONSCENE_SCALE": config.ONSCENE_SCALE,
            "TURNAROUND_MIN": config.TURNAROUND_MIN,
            "TURNAROUND_SCALE": config.TURNAROUND_SCALE,
            "MAX_QUEUE_RETRIES": config.MAX_QUEUE_RETRIES,
            "ROAD_FACTOR_BY_SHIFT": getattr(config, "ROAD_FACTOR_BY_SHIFT", {}),
            "HOUR_CONGESTION": getattr(config, "HOUR_CONGESTION", {}),
            "ZONES": getattr(config, "ZONES", []),
            "ZONE_MULTIPLIER": getattr(config, "ZONE_MULTIPLIER", {}),
            "TT_NOISE_SIGMA": getattr(config, "TT_NOISE_SIGMA", 0.0),
        },
        "inputs": {
            "CALLS_PARQUET": str(config.CALLS_PARQUET),
            "STATIONS_CSV": str(config.STATIONS_CSV),
            "UNITS_CSV": str(config.UNITS_CSV),
        },
        "counts": {
            "calls": len(calls),
            "units": len(units),
            "segments": len(groups),
        },
    }
    paths = write_artifacts(run_id, per_shift_df, summary_df, meta)

    # console summary
    print(f"📦 calls loaded: {len(calls)}  units: {len(units)}")
    print(f"SEGMENT_BY_SHIFT={config.SEGMENT_BY_SHIFT}")
    print(f"🗂 segments: {len(groups)}  sample: {list(sorted(groups.keys())[:3])}")
    print(summary_df.to_string(index=False))
    print(f"📝 Run ID: {run_id}")
    print(f"→ per-shift: {paths['per_shift_csv']}")
    print(f"→ summary:   {paths['summary_csv']}")
    print(f"→ meta:      {paths['meta_json']}")
    print(f"→ runlog:    {paths['runlog_csv']}")


if __name__ == "__main__":
    main()