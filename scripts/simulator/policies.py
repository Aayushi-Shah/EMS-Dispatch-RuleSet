# scripts/simulator/policies.py (update)
from . import config, traffic
import math

def nearest_unit_policy(units, now_min, call):
    best = None
    best_resp = float("inf")
    abs_now = call["_abs_epoch_start"] + now_min*60.0

    for u in units:
        if getattr(u, "can_dispatch", True) is False:
            continue  # off-duty for dispatch purposes
        # travel to scene via heuristic
        t_to_scene = config.DISPATCH_DELAY_MIN + traffic.travel_minutes(
            u.lon, u.lat, call["lon"], call["lat"],
            config.SCENE_SPEED_MPH, abs_now
        )
        if t_to_scene < best_resp:
            best_resp = t_to_scene
            best = u
    return best, best_resp if best is not None else (None, None)