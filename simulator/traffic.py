# scripts/sim/traffic.py
from __future__ import annotations
import math, json, os, threading
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from urllib import request, error
import config

# --- lightweight diagnostics ---
_legs = 0
_sum_road_factor = 0.0
_sum_hour_mult = 0.0
_sum_zone_mult = 0.0
_ors_cache = {}
_ors_cache_lock = threading.Lock()
_ors_cache_loaded = False
_ors_cache_path: Path | None = None

def reset_stats():
    global _legs, _sum_road_factor, _sum_hour_mult, _sum_zone_mult
    _legs = 0
    _sum_road_factor = 0.0
    _sum_hour_mult = 0.0
    _sum_zone_mult = 0.0

def snapshot_stats():
    # Returns a dict with aggregate counts/means. Does not reset.
    if _legs == 0:
        return {"legs": 0, "avg_road_factor": 0.0, "avg_hour_mult": 0.0, "avg_zone_mult": 0.0}
    return {
        "legs": _legs,
        "avg_road_factor": _sum_road_factor / _legs,
        "avg_hour_mult": _sum_hour_mult / _legs,
        "avg_zone_mult": _sum_zone_mult / _legs,
    }


def _load_ors_cache():
    global _ors_cache_loaded, _ors_cache, _ors_cache_path
    if _ors_cache_loaded:
        return
    path = getattr(config, "ORS_CACHE_PATH", None)
    if path is None:
        _ors_cache_loaded = True
        return
    _ors_cache_path = Path(path)
    if _ors_cache_path.exists():
        try:
            _ors_cache = json.loads(_ors_cache_path.read_text())
        except Exception:
            _ors_cache = {}
    _ors_cache_loaded = True


def _save_ors_cache():
    if _ors_cache_path is None:
        return
    try:
        _ors_cache_path.parent.mkdir(parents=True, exist_ok=True)
        _ors_cache_path.write_text(json.dumps(_ors_cache))
    except Exception:
        pass


def _key_for_coords(lon1, lat1, lon2, lat2) -> str:
    return f"{lon1:.5f},{lat1:.5f}->{lon2:.5f},{lat2:.5f}"


def _ors_travel_minutes(lon1, lat1, lon2, lat2) -> float | None:
    """
    Attempt ORS if explicitly enabled; otherwise fall back to heuristic.
    You can force-disable ORS at runtime via env DISABLE_ORS=1/true.
    """
    if os.getenv("DISABLE_ORS", "").lower() in {"1", "true", "yes"}:
        return None
    if not getattr(config, "ORS_USE", False):
        return None
    api_key = getattr(config, "ORS_API_KEY", "") or os.getenv("ORS_API_KEY", "")
    if not api_key:
        return None
    _load_ors_cache()
    key = _key_for_coords(lon1, lat1, lon2, lat2)
    with _ors_cache_lock:
        if key in _ors_cache:
            return float(_ors_cache[key])

    url = f"{config.ORS_BASE_URL}/{config.ORS_PROFILE}"
    payload = {
        "locations": [
            [float(lon1), float(lat1)],
            [float(lon2), float(lat2)],
        ],
        "metrics": ["duration"],
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }
    try:
        req = request.Request(url, data=data, headers=headers, method="POST")
        with request.urlopen(req, timeout=2) as resp:
            body = resp.read().decode("utf-8")
        obj = json.loads(body)
        durations = obj.get("durations") or obj.get("matrix") or []
        # durations is a 2x2 matrix; [0][1] is origin->dest
        seconds = durations[0][1]
        minutes = float(seconds) / 60.0
        with _ors_cache_lock:
            _ors_cache[key] = minutes
            _save_ors_cache()
        return minutes
    except Exception:
        return None

def haversine_mi(lon1, lat1, lon2, lat2) -> float:
    R = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlmb = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def which_shift(min_in_day: int) -> int:
    for i,(a,b) in enumerate(config.SHIFT_WINDOWS):
        if a <= min_in_day < b: return i
    return len(config.SHIFT_WINDOWS)-1

def tod_parts_from_epoch(abs_epoch: float) -> tuple[int,int]:
    # returns (tod_min 0..1439, hour 0..23)
    tod_min = int(abs_epoch % (24*3600) // 60)
    hour = tod_min // 60
    return tod_min, hour

def classify_zone(lon: float, lat: float) -> str:
    for (xmin,ymin,xmax,ymax,name) in config.ZONES:
        if xmin <= lon <= xmax and ymin <= lat <= ymax:
            return name
    return "default"

def bottleneck_penalty(lon1,lat1,lon2,lat2) -> float:
    return 0.0

def travel_minutes(lon1,lat1,lon2,lat2, base_mph: float, abs_epoch: float) -> float:
    # Try ORS if enabled
    ors_minutes = _ors_travel_minutes(lon1, lat1, lon2, lat2)
    if ors_minutes is not None:
        return ors_minutes

    # 1) base crow-fly distance
    mi = haversine_mi(lon1,lat1,lon2,lat2)

    # 2) shift and hour multipliers
    tod_min, hour = tod_parts_from_epoch(abs_epoch)
    sidx = which_shift(tod_min)
    road_factor = config.ROAD_FACTOR_BY_SHIFT.get(sidx, 1.40)
    hour_mult   = config.HOUR_CONGESTION.get(hour, 1.0)

    # 3) zone multiplier using the origin zone (cheap; adequate)
    zone = classify_zone(lon1, lat1)
    zone_mult = config.ZONE_MULTIPLIER.get(zone, 1.0)

    # 4) inflate distance, then convert to minutes at base speed
    eff_mi = mi * road_factor * hour_mult * zone_mult

    # update diagnostics
    global _legs, _sum_road_factor, _sum_hour_mult, _sum_zone_mult
    _legs += 1
    _sum_road_factor += float(road_factor)
    _sum_hour_mult += float(hour_mult)
    _sum_zone_mult += float(zone_mult)

    minutes = 60.0 * (eff_mi / max(base_mph, 1e-6))

    # 5) small lognormal noise for realism and tie-breaking
    if config.TT_NOISE_SIGMA and config.TT_NOISE_SIGMA > 0:
        minutes *= float(np.random.lognormal(mean=0.0, sigma=config.TT_NOISE_SIGMA))

    return minutes
