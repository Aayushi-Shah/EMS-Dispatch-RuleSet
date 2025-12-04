# Shared helpers and base classes for simulator policies
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np

from scripts.simulator import config, traffic
from scripts.simulator.rules import apply_rules


UnitLike = Any
CallDict = Dict[str, Any]
PolicyResult = Tuple[Optional[UnitLike], float, Dict[str, Any]]


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
    """
    Keep units that can dispatch (and optionally are not busy past now_min).

    now_min is in minutes from segment start.
    """
    out = [u for u in units if getattr(u, "can_dispatch", True)]
    if now_min is not None:
        out = [u for u in out if float(getattr(u, "busy_until", 0.0) or 0.0) <= now_min + 1e-9]
    return out


def _filter_bls_by_capability(units: List[UnitLike], bls_capable: bool) -> List[UnitLike]:
    """
    Drop BLS units when BLS is not geographically/clinically capable.
    If that leaves us with an empty list, fall back to the original units.
    """
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
    """
    Adjust score based on severity and ALS/BLS capability.

    - For high-risk calls (risk >= high_thresh), penalize BLS if ALS is capable.
    - For low-risk calls (risk <= low_thresh), penalize ALS if BLS is capable.
    """
    mult = 1.0
    if risk >= high_thresh and als_capable:
        if utype == "BLS":
            mult = penalty_bls_for_high
    elif risk <= low_thresh and bls_capable:
        if utype == "ALS":
            mult = penalty_als_for_low
    return mult


# -----------------------------
# Rule signal helpers (P2–P5)
# -----------------------------
def _iter_rule_results(rr: Any) -> List[Any]:
    """
    Normalize apply_rules(...) output into a list of rule results.

    Handles:
      - None -> []
      - single RuleResult -> [RuleResult]
      - iterable of RuleResult -> list(rr)
    """
    if rr is None:
        return []
    if isinstance(rr, (list, tuple)):
        return list(rr)
    # Assume a single RuleResult-like object
    return [rr]

def _rule_signals_p2(call: CallDict, unit: UnitLike | None, now_min: float) -> Dict[str, Any]:
    """
    Signals for P2: ALS/BLS capability + risk.

    Uses R2/R3/R9. If those rules cannot determine capability at all
    (both als_capable and bls_capable are False/unknown), we fall back
    to assuming BOTH ALS and BLS are capable, so we don't globally
    disable BLS just because the boundary tagging is incomplete.
    """
    rr_raw = apply_rules(["R2", "R3", "R9"], call, unit, now_min)
    rr = _iter_rule_results(rr_raw)

    als_flags: List[bool] = []
    bls_flags: List[bool] = []
    risks: List[float] = []

    for r in rr:
        if getattr(r, "als_capable", None) is not None:
            als_flags.append(bool(r.als_capable))
        if getattr(r, "bls_capable", None) is not None:
            bls_flags.append(bool(r.bls_capable))
        if getattr(r, "risk_score", None) is not None:
            try:
                risks.append(float(r.risk_score))
            except (TypeError, ValueError):
                pass

    als_capable = any(als_flags) if als_flags else False
    bls_capable = any(bls_flags) if bls_flags else False
    risk = risks[0] if risks else 0.0

    # Fallback when rules cannot say anything: assume both are capable.
    # This prevents the "BLS never capable anywhere" degeneracy.
    if not als_capable and not bls_capable:
        als_capable = True
        bls_capable = True

    return {
        "als_capable": als_capable,
        "bls_capable": bls_capable,
        "risk": float(risk),
    }

def _rule_signals_p3(call: CallDict, unit: UnitLike | None, now_min: float) -> Dict[str, Any]:
    """
    Signals for P3: keep_free / keep_close.

    Note: apply_rules(...) may return a single RuleResult or a list.
    """
    rr = apply_rules(["R5", "R7", "R8"], call, unit, now_min)

    if isinstance(rr, (list, tuple)):
        results = rr
    else:
        results = [rr]

    return {
        "keep_free": any(getattr(r, "keep_free_flag", False) for r in results),
        "keep_close": any(getattr(r, "keep_close_flag", False) for r in results),
    }


def _rule_signals_p4(call: CallDict, unit: UnitLike | None, now_min: float) -> Dict[str, Any]:
    """
    Signals for P4 fairness: fairness weights + keep flags.

    Note: apply_rules(...) may return a single RuleResult or a list.
    """
    rr = apply_rules(["R5", "R7", "R8"], call, unit, now_min)

    if isinstance(rr, (list, tuple)):
        results = rr
    else:
        results = [rr]

    fairness_weights = [
        float(r.fairness_weight)
        for r in results
        if getattr(r, "fairness_weight", None) is not None
    ]
    fairness_w = max(fairness_weights) if fairness_weights else 1.0

    return {
        "fairness_w": fairness_w,
        "keep_free": any(getattr(r, "keep_free_flag", False) for r in results),
        "keep_close": any(getattr(r, "keep_close_flag", False) for r in results),
    }


def _rule_signals_p5(call: CallDict, unit: UnitLike | None, now_min: float) -> Dict[str, Any]:
    """
    Signals for P5 hybrid: capability, risk, fairness, and demand hints.

    Note: apply_rules(...) may return a single RuleResult or a list.
    """
    rr = apply_rules(["R2", "R3", "R5", "R6", "R7", "R8", "R9"], call, unit, now_min)

    if isinstance(rr, (list, tuple)):
        results = rr
    else:
        results = [rr]

    fairness_weights = [
        float(r.fairness_weight)
        for r in results
        if getattr(r, "fairness_weight", None) is not None
    ]
    fairness_w = max(fairness_weights) if fairness_weights else 1.0

    high_demand_weights = [
        float(r.high_demand_weight)
        for r in results
        if getattr(r, "high_demand_weight", None) is not None
    ]
    zone_demand_score = (max(high_demand_weights) - 1.0) if high_demand_weights else 0.0

    return {
        "als_capable": any(
            getattr(r, "als_capable", None)
            for r in results
            if getattr(r, "als_capable", None) is not None
        ),
        "bls_capable": any(
            getattr(r, "bls_capable", None)
            for r in results
            if getattr(r, "bls_capable", None) is not None
        ),
        "risk": float(
            next(
                (r.risk_score for r in results if getattr(r, "risk_score", None) is not None),
                0.0,
            )
        ),
        "fairness_w": fairness_w,
        "keep_free": any(getattr(r, "keep_free_flag", False) for r in results),
        "keep_close": any(getattr(r, "keep_close_flag", False) for r in results),
        "zone_underprotected": any(getattr(r, "zone_underprotected", False) for r in results)
        or bool(call.get("zone_underprotected", False)),
        "zone_demand_score": zone_demand_score,
        "require_als": bool(call.get("require_als", False)),
        "has_zone": call.get("zone") is not None,
    }

# -----------------------------
# ETA / distance helpers (P1/P2)
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

    now_min: minutes from segment start.
    call["_abs_epoch_start"]: absolute start time (epoch seconds) of the segment.
    """
    u_lon, u_lat = float(unit.lon), float(unit.lat)
    c_lon, c_lat = float(call["lon"]), float(call["lat"])

    # Absolute time at decision = segment start + now_min
    abs_now = float(call.get("_abs_epoch_start", 0.0)) + 60.0 * float(now_min)

    travel_min = traffic.travel_minutes(
        u_lon,
        u_lat,
        c_lon,
        c_lat,
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
    busy_until = float(getattr(unit, "busy_until", 0.0) or 0.0)
    if busy_until <= 0.0:
        return 0.0
    # Cap contribution so it never dominates ETA
    return min(busy_until * 0.01, 1.0)


def r10_random_tiebreaker(rng: np.random.Generator) -> float:
    """
    Very small jitter so two identical ETAs don't always pick same unit.
    """
    return float(rng.uniform(0.0, 1e-3))


# -----------------------------
# Urban / rural accessors (tag-based)
# -----------------------------
def call_urban_rural(call: CallDict) -> str:
    """
    Prefer pre-tagged call['urban_rural'] or call['call_area'].
    No geometry here; tagging is done upstream in build_lemsa_tagged.
    """
    if not call:
        return "unknown"
    if isinstance(call.get("urban_rural"), str) and call["urban_rural"]:
        return call["urban_rural"]
    if isinstance(call.get("call_area"), str) and call["call_area"]:
        return call["call_area"]
    return "unknown"


def unit_urban_rural(u: UnitLike) -> str:
    """
    Prefer pre-tagged unit.unit_area or unit.urban_rural.

    unit_area comes from lemsa_units_from_calls.csv;
    we cache it on the unit if we have to infer anything.
    """
    val = getattr(u, "unit_area", None)
    if isinstance(val, str) and val:
        return val

    val2 = getattr(u, "urban_rural", None)
    if isinstance(val2, str) and val2:
        return val2

    # As a last resort, tag as unknown; we don't want geometry in the hot path.
    area = "unknown"
    try:
        setattr(u, "urban_rural", area)
    except Exception:
        pass
    return area


def classify_urban_rural(lon: float, lat: float) -> str:
    """
    Legacy hook kept for compatibility. In the current architecture,
    urban/rural classification is done upstream; this returns 'unknown'.
    """
    return "unknown"


@lru_cache(maxsize=1024)
def classify_urban_rural_cached(lon: float, lat: float) -> str:
    """
    Cached legacy hook. Kept for compatibility with any code that still calls it.
    """
    return classify_urban_rural(float(lon), float(lat))


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