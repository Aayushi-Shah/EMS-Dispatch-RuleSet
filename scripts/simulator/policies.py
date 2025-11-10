# scripts/simulator/policies.py
from __future__ import annotations
import math
from typing import Callable, Tuple, Dict, Any
from . import config, traffic

def _hav_mi(lon1, lat1, lon2, lat2) -> float:
    R = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _available_by_station(units, now_min):
    by = {}
    for u in units:
        if getattr(u, "can_dispatch", True) and u.busy_until <= now_min:
            by.setdefault(u.station, 0)
            by[u.station] += 1
    return by

def _eta_to_call(u, call, now_abs_epoch) -> float:
    # DISPATCH delay + heuristic road minutes
    mi = _hav_mi(u.lon, u.lat, call["lon"], call["lat"])
    # convert using traffic heuristic multipliers
    base_travel = traffic.travel_minutes(u.lon, u.lat, call["lon"], call["lat"],
                                         config.SCENE_SPEED_MPH, now_abs_epoch)
    return config.DISPATCH_DELAY_MIN + base_travel

def nearest_eta_policy(units, now_min, call, **_) -> Tuple[Any, float, Dict[str, Any]]:
    """Return (best_unit, resp_minutes, debug)."""
    now_abs = call["_abs_epoch_start"] + now_min*60.0
    cand = []
    for u in units:
        if not getattr(u, "can_dispatch", True): 
            continue
        if u.busy_until > now_min:
            continue
        eta = _eta_to_call(u, call, now_abs)
        cand.append((eta, u, {"eta": eta, "penalty": 0.0, "why": "nearest_eta"}))
    if not cand:
        return None, float("inf"), {"candidates": 0}
    cand.sort(key=lambda x: x[0])
    eta, best, info = cand[0]
    return best, float(eta), {"candidates": len(cand), "chosen": best.name, "chosen_info": info}

def station_bias_eta_policy(units, now_min, call, penalty_min: float = 2.0):
    """Nearest ETA with penalty if selecting the last free unit at that station."""
    now_abs = call["_abs_epoch_start"] + now_min*60.0
    free_by_st = _available_by_station(units, now_min)
    cand = []
    for u in units:
        if not getattr(u, "can_dispatch", True): 
            continue
        if u.busy_until > now_min:
            continue
        eta = _eta_to_call(u, call, now_abs)
        pen = penalty_min if free_by_st.get(u.station, 0) == 1 else 0.0
        score = eta + pen
        cand.append((score, u, {"eta": eta, "penalty": pen, "why": "station_bias"}))
    if not cand:
        return None, float("inf"), {"candidates": 0}
    cand.sort(key=lambda x: x[0])
    score, best, info = cand[0]
    return best, float(info["eta"]), {"candidates": len(cand), "chosen": best.name, "chosen_info": info}

def max_radius_cap_policy(units, now_min, call, max_mi: float = 12.0):
    """Nearest ETA but drop candidates whose straight-line is beyond max_mi."""
    now_abs = call["_abs_epoch_start"] + now_min*60.0
    cand = []
    for u in units:
        if not getattr(u, "can_dispatch", True):
            continue
        if u.busy_until > now_min:
            continue
        if _hav_mi(u.lon, u.lat, call["lon"], call["lat"]) > max_mi:
            continue
        eta = _eta_to_call(u, call, now_abs)
        cand.append((eta, u, {"eta": eta, "penalty": 0.0, "why": "radius_cap"}))
    if not cand:
        return None, float("inf"), {"candidates": 0}
    cand.sort(key=lambda x: x[0])
    eta, best, info = cand[0]
    return best, float(eta), {"candidates": len(cand), "chosen": best.name, "chosen_info": info}

# simple registry
POLICIES: dict[str, Callable] = {
    "NearestETA": nearest_eta_policy,
    "StationBiasETA": station_bias_eta_policy,   # kwargs: penalty_min
    "MaxRadiusCap": max_radius_cap_policy,       # kwargs: max_mi
}

def select_policy(name: str):
    return POLICIES.get(name, nearest_eta_policy)