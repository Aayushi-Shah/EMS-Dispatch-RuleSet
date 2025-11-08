# scripts/sim/traffic.py
from __future__ import annotations
import math, numpy as np
from datetime import datetime, timezone
from . import config

# --- lightweight diagnostics ---
_legs = 0
_sum_road_factor = 0.0
_sum_hour_mult = 0.0
_sum_zone_mult = 0.0

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
    # crude: if either endpoint is inside a bottleneck box, apply penalty once
    pen = 0.0
    for b in config.BOTTLENECKS:
        xmin,ymin,xmax,ymax = b["bbox"]
        if (xmin<=lon1<=xmax and ymin<=lat1<=ymax) or (xmin<=lon2<=xmax and ymin<=lat2<=ymax):
            pen += float(b.get("penalty_min", 0.0))
    return pen

def travel_minutes(lon1,lat1,lon2,lat2, base_mph: float, abs_epoch: float) -> float:
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

    # 5) bottleneck penalty minutes
    minutes += bottleneck_penalty(lon1,lat1,lon2,lat2)

    # 6) small lognormal noise for realism and tie-breaking
    if config.TT_NOISE_SIGMA and config.TT_NOISE_SIGMA > 0:
        minutes *= float(np.random.lognormal(mean=0.0, sigma=config.TT_NOISE_SIGMA))

    return minutes