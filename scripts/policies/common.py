# Shared helpers and base classes for simulator policies
from __future__ import annotations

import json
import math
from functools import lru_cache
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from scripts.simulator import config, traffic
from scripts.simulator.rules import apply_rules

UnitLike = Any
CallDict = Dict[str, Any]
PolicyResult = Tuple[Optional[UnitLike], float, Dict[str, Any]]

# Optional geospatial libraries
try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union
except ImportError:
    Point = None
    shape = None
    unary_union = None


# -----------------------------
# helper utilities shared by policies
# -----------------------------
_ALS_LABELS = {"ALS", "MEDIC", "MICU", "PARAMEDIC"}
_BLS_LABELS = {"BLS", "AMB", "AMBULANCE", "BASIC", "SUPPORT"}


def _norm_utype(u: UnitLike) -> str:
    """
    Normalize various unit_type strings into 'ALS' or 'BLS' when possible.
    Falls back to the raw uppercased type if unknown.
    """
    raw = str(getattr(u, "utype", "")).upper()
    if raw in _ALS_LABELS:
        return "ALS"
    if raw in _BLS_LABELS:
        return "BLS"
    return raw


def _norm_utype_cached(u: UnitLike) -> str:
    """Normalize once per unit instance, caching on the object if possible."""
    cached = getattr(u, "_norm_utype", None)
    if cached:
        return cached
    normed = _norm_utype(u)
    try:
        setattr(u, "_norm_utype", normed)
    except Exception:
        pass
    return normed


def _filter_dispatchable(units: List[UnitLike], now_min: float | None = None) -> List[UnitLike]:
    """Keep units that can dispatch (and optionally are not busy past now_min)."""
    out = [u for u in units if getattr(u, "can_dispatch", True)]
    if now_min is not None:
        out = [u for u in out if float(getattr(u, "busy_until", 0.0) or 0.0) <= now_min + 1e-9]
    return out


def _filter_bls_by_capability(units: List[UnitLike], bls_capable: bool) -> List[UnitLike]:
    """Drop BLS units when BLS is not geographically capable."""
    filtered: List[UnitLike] = []
    for u in units:
        utype = _norm_utype_cached(u)
        if utype == "BLS" and not bls_capable:
            continue
        filtered.append(u)
    return filtered or units


def _severity_multiplier(
    utype: str,
    risk: float,
    als_capable: bool,
    bls_capable: bool,
    high_thresh: float,
    low_thresh: float,
    penalty_bls_for_high: float,
    penalty_als_for_low: float,
) -> float:
    mult = 1.0
    if risk >= high_thresh and als_capable:
        if utype == "BLS":
            mult = penalty_bls_for_high
    elif risk <= low_thresh and bls_capable:
        if utype == "ALS":
            mult = penalty_als_for_low
    return mult


def _rule_signals_p2(call: CallDict) -> Dict[str, Any]:
    """Lightweight signals for P2 (no zone overlap / TOD)."""
    rr = apply_rules(["R2", "R3", "R5", "R8", "R9"], call)
    return {
        "als_capable": any(r.als_capable for r in rr if getattr(r, "als_capable", None) is not None),
        "bls_capable": any(r.bls_capable for r in rr if getattr(r, "bls_capable", None) is not None),
        "risk": float(next((r.risk_score for r in rr if getattr(r, "risk_score", None) is not None), 0.0)),
        "keep_free": any(getattr(r, "keep_free_flag", False) for r in rr),
    }


def _rule_signals_p3(call: CallDict) -> Dict[str, Any]:
    """Lightweight signals for P3 (keep flags)."""
    rr = apply_rules(["R5", "R7", "R8"], call)
    return {
        "keep_free": any(getattr(r, "keep_free_flag", False) for r in rr),
        "keep_close": any(getattr(r, "keep_close_flag", False) for r in rr),
    }


def _rule_signals_p4(call: CallDict) -> Dict[str, Any]:
    """Signals for P4 fairness: fairness weights + keep flags."""
    rr = apply_rules(["R5", "R7", "R8"], call)

    fairness_weights = [
        float(r.fairness_weight)
        for r in rr
        if getattr(r, "fairness_weight", None) is not None
    ]
    fairness_w = max(fairness_weights) if fairness_weights else 1.0

    return {
        "fairness_w": fairness_w,
        "keep_free": any(getattr(r, "keep_free_flag", False) for r in rr),
        "keep_close": any(getattr(r, "keep_close_flag", False) for r in rr),
    }


def _rule_signals_p5(call: CallDict) -> Dict[str, Any]:
    """
    Signals for P5 hybrid: capability, risk, fairness, demand/zone hints.
    """
    rr = apply_rules(["R2", "R3", "R5", "R6", "R7", "R8", "R9"], call)

    fairness_weights = [
        float(r.fairness_weight)
        for r in rr
        if getattr(r, "fairness_weight", None) is not None
    ]
    fairness_w = max(fairness_weights) if fairness_weights else 1.0

    high_demand_weights = [
        float(r.high_demand_weight)
        for r in rr
        if getattr(r, "high_demand_weight", None) is not None
    ]
    zone_demand_score = (max(high_demand_weights) - 1.0) if high_demand_weights else 0.0

    return {
        "als_capable": any(r.als_capable for r in rr if getattr(r, "als_capable", None) is not None),
        "bls_capable": any(r.bls_capable for r in rr if getattr(r, "bls_capable", None) is not None),
        "risk": float(next((r.risk_score for r in rr if getattr(r, "risk_score", None) is not None), 0.0)),
        "fairness_w": fairness_w,
        "keep_free": any(getattr(r, "keep_free_flag", False) for r in rr),
        "keep_close": any(getattr(r, "keep_close_flag", False) for r in rr),
        "zone_underprotected": any(getattr(r, "zone_underprotected", False) for r in rr) or bool(call.get("zone_underprotected", False)),
        "zone_demand_score": zone_demand_score,
        "require_als": bool(call.get("require_als", False)),
        "has_zone": call.get("zone") is not None,
    }


# -----------------------------
# small helpers for P1/P2
# -----------------------------
def _hav_miles(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance in statute miles (fallback if traffic.travel_minutes not used)."""
    R = 3958.7613  # Earth radius in miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _eta_with_traffic(unit: UnitLike, now_min: float, call: CallDict) -> float:
    """
    Compute ETA using traffic.travel_minutes (digital twin) + DISPATCH_DELAY_MIN.
    This is the shared "R1 ETA" for both P1 and P2.
    """
    u_lon, u_lat = float(unit.lon), float(unit.lat)
    c_lon, c_lat = float(call["lon"]), float(call["lat"])

    # Absolute time at decision = segment start + now_min
    abs_now = float(call.get("_abs_epoch_start", 0.0)) + 60.0 * float(now_min)

    travel_min = traffic.travel_minutes(
        u_lon, u_lat,
        c_lon, c_lat,
        config.SCENE_SPEED_MPH,
        abs_now,
    )
    dispatch_delay = getattr(config, "DISPATCH_DELAY_MIN", 0.0)
    return float(dispatch_delay + travel_min)


def r1_eta_minutes(unit: UnitLike, now_min: float, call: CallDict) -> float:
    """
    Approximate ETA from unit → incident in minutes.

    R1 definition for the project: dispatch delay + travel time with traffic.
    """
    return _eta_with_traffic(unit, now_min, call)


def r8_busy_load_penalty(unit: UnitLike, now_min: float) -> float:
    """
    Small bias against units that are still busy.

    If busy_until > 0 (minutes from segment start),
    add a tiny penalty proportional to workload.
    """
    busy_until = getattr(unit, "busy_until", 0.0) or 0.0
    if busy_until <= 0.0:
        return 0.0

    # Cap contribution so it never dominates ETA
    return min(float(busy_until) * 0.01, 1.0)


def r10_random_tiebreaker(rng: np.random.Generator) -> float:
    """
    Very small jitter so two identical ETAs don't always pick same unit.
    """
    return float(rng.uniform(0.0, 1e-3))


# -------------------------------------------------
# Urban / rural classification for Lancaster
# using pre-built county-clipped TIGER layers:
#   - reference/urban_areas_lancaster.geojson
#   - reference/rural_area_lancaster.geojson
# -------------------------------------------------
URBAN_UNION = None
RURAL_UNION = None
COUNTY_UNION = None


def _load_geojson_polygons(path: str) -> List[List[List[Tuple[float, float]]]]:
    """
    Load polygons from GeoJSON into list-of-polygons representation.
    Each polygon is a list of rings; each ring is a list of (lon, lat).
    """
    polys: List[List[List[Tuple[float, float]]]] = []
    fc = json.loads(open(path).read())
    for feat in fc.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "Polygon":
            polys.append([
                [(float(x), float(y)) for x, y in ring]
                for ring in coords
            ])
        elif gtype == "MultiPolygon":
            for poly in coords:
                polys.append([
                    [(float(x), float(y)) for x, y in ring]
                    for ring in poly
                ])
    return polys


def _point_in_ring(point: Tuple[float, float], ring: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon for a single ring."""
    x, y = point
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-18) + x1)
        if cond:
            inside = not inside
    return inside


def _point_in_polygon(point: Tuple[float, float], polygon: List[List[Tuple[float, float]]]) -> bool:
    """
    Point-in-polygon for polygon with holes.
    polygon[0] = outer ring, polygon[1:] = holes.
    """
    if not polygon:
        return False
    if not _point_in_ring(point, polygon[0]):
        return False
    for hole in polygon[1:]:
        if _point_in_ring(point, hole):
            return False
    return True


def _union_contains(union_obj: Any, lon: float, lat: float) -> bool:
    """Generic containment check for shapely unions or list-of-polygons fallback."""
    if union_obj is None:
        return False
    if Point is not None and hasattr(union_obj, "contains"):
        try:
            return bool(union_obj.contains(Point(float(lon), float(lat))))
        except Exception:
            return False
    if isinstance(union_obj, list):
        pt = (float(lon), float(lat))
        return any(_point_in_polygon(pt, poly) for poly in union_obj)
    return False


def _load_urban_rural_unions() -> None:
    """
    Lazy-load union of urban and rural polygons for Lancaster.

    Defaults:
      URBAN_GEOJSON_PATH = 'reference/urban_areas_lancaster.geojson'
      RURAL_GEOJSON_PATH = 'reference/rural_area_lancaster.geojson'
    """
    global URBAN_UNION, RURAL_UNION, COUNTY_UNION

    if URBAN_UNION is not None and RURAL_UNION is not None and COUNTY_UNION is not None:
        return

    urban_path = getattr(
        config,
        "URBAN_GEOJSON_PATH",
        "reference/urban_areas_lancaster.geojson",
    )
    rural_path = getattr(
        config,
        "RURAL_GEOJSON_PATH",
        "reference/rural_area_lancaster.geojson",
    )
    county_path = getattr(
        config,
        "LANCASTER_BOUNDARY_PATH",
        "reference/lancaster_county_boundary.geojson",
    )

    if gpd is not None:
        # Preferred: geopandas load + CRS normalization
        try:
            ugdf = gpd.read_file(urban_path)
            if ugdf.crs is not None and ugdf.crs.to_epsg() != 4326:
                ugdf = ugdf.to_crs("EPSG:4326")
            URBAN_UNION = ugdf.unary_union
        except Exception:
            URBAN_UNION = None

        try:
            rgdf = gpd.read_file(rural_path)
            if rgdf.crs is not None and rgdf.crs.to_epsg() != 4326:
                rgdf = rgdf.to_crs("EPSG:4326")
            RURAL_UNION = rgdf.unary_union
        except Exception:
            RURAL_UNION = None

    elif Point is not None and shape is not None and unary_union is not None:
        # Fallback: shapely-only load if geopandas is unavailable
        try:
            urban_fc = json.loads(open(urban_path).read())
            u_geoms = [shape(feat["geometry"]) for feat in urban_fc.get("features", []) if "geometry" in feat]
            URBAN_UNION = unary_union(u_geoms) if u_geoms else None
        except Exception:
            URBAN_UNION = None

        try:
            rural_fc = json.loads(open(rural_path).read())
            r_geoms = [shape(feat["geometry"]) for feat in rural_fc.get("features", []) if "geometry" in feat]
            RURAL_UNION = unary_union(r_geoms) if r_geoms else None
        except Exception:
            RURAL_UNION = None

        try:
            county_fc = json.loads(open(county_path).read())
            c_geoms = [shape(feat["geometry"]) for feat in county_fc.get("features", []) if "geometry" in feat]
            COUNTY_UNION = unary_union(c_geoms) if c_geoms else None
        except Exception:
            COUNTY_UNION = None
    else:
        # Last-resort: pure Python PIP fallback (no geopandas/shapely)
        try:
            URBAN_UNION = _load_geojson_polygons(urban_path)
        except Exception:
            URBAN_UNION = None
        try:
            RURAL_UNION = _load_geojson_polygons(rural_path)
        except Exception:
            RURAL_UNION = None
        try:
            COUNTY_UNION = _load_geojson_polygons(county_path)
        except Exception:
            COUNTY_UNION = None

    if COUNTY_UNION is None and gpd is not None:
        # Try to populate COUNTY_UNION via geopandas if still missing
        try:
            cgdf = gpd.read_file(county_path)
            if cgdf.crs is not None and cgdf.crs.to_epsg() != 4326:
                cgdf = cgdf.to_crs("EPSG:4326")
            COUNTY_UNION = cgdf.unary_union
        except Exception:
            COUNTY_UNION = None


def classify_urban_rural(lon: float, lat: float) -> str:
    _load_urban_rural_unions()

    if _union_contains(URBAN_UNION, lon, lat):
        return "urban"
    if _union_contains(RURAL_UNION, lon, lat):
        return "rural"
    # fallback: inside county but not tagged → treat as rural
    if _union_contains(COUNTY_UNION, lon, lat):
        return "rural"
    return "unknown"


@lru_cache(maxsize=1024)
def classify_urban_rural_cached(lon: float, lat: float) -> str:
    """Cached wrapper for classify_urban_rural to avoid repeated geometry work."""
    return classify_urban_rural(float(lon), float(lat))


def call_urban_rural(call: CallDict) -> str:
    """
    Prefer pre-tagged call['urban_rural']; fall back to geometry-based classify_urban_rural.
    """
    if call is None:
        return "unknown"
    val = call.get("urban_rural")
    if isinstance(val, str) and val:
        return val
    lon = call.get("lon")
    lat = call.get("lat")
    if lon is None or lat is None:
        return "unknown"
    return classify_urban_rural_cached(float(lon), float(lat))


def unit_urban_rural(u: UnitLike) -> str:
    """
    Prefer cached/pre-tagged unit.urban_rural; otherwise classify once and cache on the unit.
    """
    val = getattr(u, "urban_rural", None)
    if isinstance(val, str) and val:
        return val
    lon = getattr(u, "lon", None) or getattr(u, "station_lon", None)
    lat = getattr(u, "lat", None) or getattr(u, "station_lat", None)
    if lon is None or lat is None:
        return "unknown"
    area = classify_urban_rural_cached(float(lon), float(lat))
    try:
        setattr(u, "urban_rural", area)
    except Exception:
        pass
    return area


# -----------------------------
# Base policy
# -----------------------------
class BasePolicy:
    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs
        seed = kwargs.get("seed", config.RANDOM_SEED)
        self._rng = np.random.default_rng(seed)

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        raise NotImplementedError
