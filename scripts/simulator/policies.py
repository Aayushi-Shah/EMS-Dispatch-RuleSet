# scripts/simulator/policies.py
from __future__ import annotations
from typing import Tuple, Optional, List
from .des import Unit
from . import config, traffic

def nearest_unit_policy(units: List[Unit], sim_t_min: float, call: dict) -> Tuple[Optional[Unit], float]:
    """
    Pick the unit with the minimum *heuristic travel time* to the call.
    Returns (unit, resp_minutes). Adds dispatch delay.
    """
    if not units:
        return None, float("inf")

    # absolute epoch "now" for this event
    abs_now = float(call["_abs_epoch_start"]) + float(sim_t_min) * 60.0

    best_u = None
    best_minutes = float("inf")

    for u in units:
        if u.busy_until > sim_t_min:
            continue
        tmin = traffic.travel_minutes(
            u.lon, u.lat, call["lon"], call["lat"],
            config.SCENE_SPEED_MPH, abs_now
        )
        tmin += config.DISPATCH_DELAY_MIN  # tone-out + call processing

        if tmin < best_minutes:
            best_minutes = tmin
            best_u = u

    return best_u, best_minutes