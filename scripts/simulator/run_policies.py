# scripts/simulator/run_policies.py
from __future__ import annotations
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from . import config, traffic
from .des import DES, Unit
from .io import load_calls, load_units, segment_calls_by_shift, load_unit_duty
from .kpis import weighted_aggregate
from .logging_utils import new_run_id, write_artifacts
from .policies import POLICY_REGISTRY

def _build_initial_unit_state(units: List[Unit]) -> Dict[str, dict]:
    return {
        u.name: {
            "abs_free_epoch": -float("inf"),
            "lon": u.lon, "lat": u.lat,
            "station_lon": u.station_lon, "station_lat": u.station_lat,
            "utype": u.utype, "station": u.station,
        } for u in units
    }

def _is_unit_active_this_segment(name: str, duty_map: dict, seg_key: str, seg_start_abs: float) -> bool:
    if not duty_map: return True
    rules = duty_map.get(name.upper())
    if not rules: return True
    from datetime import datetime, timezone
    dow = datetime.fromtimestamp(seg_start_abs, tz=timezone.utc).weekday()
    sidx = int(seg_key.split("_S")[-1])
    seg_start_min, seg_end_min = config.SHIFT_WINDOWS[sidx]
    for days, w_start, w_end in rules:
        if dow in days and not (w_end <= seg_start_min or w_start >= seg_end_min):
            return True
    return False

def run_one_segment_with_policy(
    seg_key: str,
    calls_segment: List[dict],
    unit_state: Dict[str, dict],
    seg_start_abs_map: Dict[str, float],
    policy_select_fn,
    duty_map: dict | None = None,
) -> Tuple[dict, Dict[str, dict]]:
    traffic.reset_stats()
    sim = DES(select_unit_fn=policy_select_fn)
    new_units: List[Unit] = []
    seg_start_abs = float(seg_start_abs_map[seg_key])

    for name, s in unit_state.items():
        remaining_min = max(0.0, (s["abs_free_epoch"] - seg_start_abs) / 60.0)
        lon, lat = (s["lon"], s["lat"]) if remaining_min > 0 else (s["station_lon"], s["station_lat"])
        can_dispatch = _is_unit_active_this_segment(name, duty_map or {}, seg_key, seg_start_abs)
        u = Unit(
            name=name, utype=s["utype"], station=s["station"],
            lon=lon, lat=lat, station_lon=s["station_lon"], station_lat=s["station_lat"],
            busy_until=(remaining_min if remaining_min > 0 else 0.0),
            can_dispatch=can_dispatch
        )
        new_units.append(u)
        sim.add_unit(u)
        if u.busy_until > 0.0:
            sim.schedule(u.busy_until, "unit_free", unit=u, end_lon=u.lon, end_lat=u.lat)

    for c in calls_segment:
        c["_abs_epoch_start"] = seg_start_abs
        sim.schedule(c["tmin"], "call", **c)

    while sim.advance():
        pass

    updated_state = {}
    for u in new_units:
        abs_free = seg_start_abs + max(u.busy_until, 0.0) * 60.0
        updated_state[u.name] = {
            "abs_free_epoch": abs_free,
            "lon": u.lon, "lat": u.lat,
            "station_lon": u.station_lon, "station_lat": u.station_lat,
            "utype": u.utype, "station": u.station,
        }

    resp = np.array(sim.metrics["resp_times"]) if sim.metrics["resp_times"] else np.array([])
    row = {
        "n_calls": sim.metrics["n_calls"],
        "missed_calls": sim.metrics["missed_calls"],
        "p50_resp_min": np.percentile(resp, 50) if len(resp) else np.nan,
        "p90_resp_min": np.percentile(resp, 90) if len(resp) else np.nan,
        "avg_resp_min": float(resp.mean()) if len(resp) else np.nan,
        "units": len(new_units),
    }
    row.update(traffic.snapshot_stats())
    return row, updated_state

def run_all_policies():
    random.seed(config.RANDOM_SEED); np.random.seed(config.RANDOM_SEED)

    calls = load_calls()
    units = load_units()
    duty_map = load_unit_duty() if getattr(config, "DUTY_ENFORCEMENT", False) else {}
    groups, seg_start_abs = segment_calls_by_shift(calls)

    for pid, policy in POLICY_REGISTRY.items():
        unit_state = _build_initial_unit_state(units)
        rows = []
        for key, seg in sorted(groups.items()):
            row, unit_state = run_one_segment_with_policy(
                key, seg, unit_state, seg_start_abs, policy.select, duty_map=duty_map
            )
            row["segment"] = key
            row["policy_id"] = pid
            rows.append(row)

        per_shift_df = pd.DataFrame(rows).sort_values("segment")
        summary_df = weighted_aggregate(per_shift_df, units_count=len(units))

        run_id = new_run_id()
        meta = {
            "run_id": run_id,
            "policy": {"id": pid, "name": policy.display_name},
            "config": {
                "SEGMENT_BY_SHIFT": config.SEGMENT_BY_SHIFT,
                "SCENE_SPEED_MPH": config.SCENE_SPEED_MPH,
                "HOSPITAL_SPEED_MPH": config.HOSPITAL_SPEED_MPH,
                "DISPATCH_DELAY_MIN": config.DISPATCH_DELAY_MIN,
                "ROAD_FACTOR_BY_SHIFT": getattr(config, "ROAD_FACTOR_BY_SHIFT", {}),
                "HOUR_CONGESTION": getattr(config, "HOUR_CONGESTION", {}),
            },
            "counts": {"calls": len(calls), "units": len(units), "segments": len(groups)},
        }
        # write into a per-policy subfolder
        paths = write_artifacts(run_id=f"{pid}_{run_id}", per_shift_df=per_shift_df, summary_df=summary_df, meta=meta)
        print(f"[policy:{pid}] {summary_df.to_string(index=False)}")
        print(f"[policy:{pid}] → per-shift: {paths['per_shift_csv']}")
        print(f"[policy:{pid}] → summary:   {paths['summary_csv']}")

if __name__ == "__main__":
    run_all_policies()