# scripts/simulator/rules.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from scripts.simulator import config
from scripts.simulator.geo import load_boundary
from shapely.geometry import Point
# -----------------------------
# Rule return object
# -----------------------------
@dataclass
class RuleResult:
    # capability flags
    als_capable: bool | None = None
    bls_capable: bool | None = None

    # boundaries
    in_als_boundary: bool | None = None
    in_bls_boundary: bool | None = None
    in_overlap_boundary: bool | None = None

    # zone / demand
    zone: str | None = None
    call_zone: str | None = None
    zone_unit_count: int | None = None
    zone_underprotected: bool | None = None
    high_demand_flag: bool | None = None
    high_demand_weight: float | None = None

    # fairness / geography
    risk_score: float | None = None
    fairness_weight: float | None = None
    keep_free_flag: bool | None = None
    keep_close_flag: bool | None = None
    rural_flag: bool | None = None
    urban_flag: bool | None = None

    # tags
    priority_tag: str | None = None
    time_of_day_tag: str | None = None
# -----------------------------
# R1 — Nearest ETA baseline
# -----------------------------
def r1_nearest_eta(call: Dict[str, Any]) -> RuleResult:
    """
    R1 — Nearest ETA baseline.
    Just tags that the policy should use ETA-based selection.
    """
    return RuleResult(priority_tag="nearest_eta")


# -----------------------------
# R2 — ALS Capability
# -----------------------------

def r2_als_capability(call: Dict[str, Any]) -> RuleResult:
    """
    R2 — ALS capability:
      • Does ALS *geographically* cover this incident?
      • Uses ALS + OVERLAP polygons tagged in io.py.
      • Severity is handled by R9, not here.
    """
    in_als = bool(call.get("in_als_boundary", False))
    in_overlap = bool(call.get("in_overlap_boundary", False))

    als_capable = in_als or in_overlap

    return RuleResult(
        als_capable=als_capable,
        in_als_boundary=in_als,
        in_overlap_boundary=in_overlap,
    )

# -----------------------------
# R3 — BLS Capability
# -----------------------------
def r3_bls_capability(call: Dict[str, Any]) -> RuleResult:
    """
    R3 — BLS capability:
      • Does BLS *geographically* cover this incident?
      • Uses BLS + OVERLAP polygons.
      • No clinical decision here.
    """
    in_bls = bool(call.get("in_bls_boundary", False))
    in_overlap = bool(call.get("in_overlap_boundary", False))

    bls_capable = in_bls or in_overlap

    return RuleResult(
        bls_capable=bls_capable,
        in_bls_boundary=in_bls,
        in_overlap_boundary=in_overlap,
    )

# -----------------------------
# R4 — ZONE COVERAGE PROTECTION
# -----------------------------
def r4_zone_coverage(
    call: Dict[str, Any],
    units: List[Any],
    zone_lookup: Any,         # function: (lon,lat) → zone_id or None
    min_units_per_zone: int = 1,
) -> RuleResult:
    """
    R4: Prevent draining a zone below minimum coverage unless unavoidable.

    Outputs:
        call_zone: zone of incident
        zone_unit_count: number of free units in that zone
        zone_underprotected: True if units < min_units_per_zone

    This rule needs access to current units → NOT used via apply_rules().
    Call it directly from the policy when scoring candidate units.
    """
    lon = call.get("lon")
    lat = call.get("lat")
    zone = zone_lookup(lon, lat) if (lon is not None and lat is not None) else None

    count = 0
    if zone is not None:
        for u in units:
            # requires Unit to have .zone and .busy_until
            if getattr(u, "zone", None) == zone and getattr(u, "busy_until", 0.0) <= 0.01:
                count += 1

    underprotected = (zone is not None and count < min_units_per_zone)

    return RuleResult(
        call_zone=zone,
        zone_unit_count=count,
        zone_underprotected=underprotected,
    )

try:
    URBAN_POLY = load_boundary(str(config.URBAN_BOUNDARY))
except Exception:
    URBAN_POLY = None

def r5_fairness(call: Dict[str, Any]) -> RuleResult:
    lon, lat = call.get("lon"), call.get("lat")
    if lon is None or lat is None or URBAN_POLY is None:
        return RuleResult()

    p = Point(lon, lat)
    is_urban = URBAN_POLY.contains(p)

    return RuleResult(
        rural_flag=not is_urban,
        urban_flag=is_urban,
        fairness_weight=1.3 if not is_urban else 1.0  # rural gets a small boost
    )

HIGH_DEMAND_ZONES = {"ALS", "OVERLAP"}


def r6_zone_penalty(call: Dict[str, Any]) -> RuleResult:
    """
    R6: High-demand zone penalty / tag.

    Input:
        call["zone"] is tagged by io.load_calls() via ZONE_LOOKUP:
          e.g., "ALS", "BLS", "OVERLAP", or None.

    Output:
        high_demand_flag    – True if call is in a high-demand zone.
        high_demand_weight  – Multiplier policies can use to boost
                              or penalize pulling units *out of* that zone.
    """
    zone = call.get("zone")
    is_high = zone in HIGH_DEMAND_ZONES

    # Simple default: slight boost for calls in high-demand zones
    weight = 1.2 if is_high else 1.0

    return RuleResult(
        zone=zone,
        high_demand_flag=is_high,
        high_demand_weight=weight,
    )

# -----------------------------
# R7 — Low-priority calls keep nearest unit available
# -----------------------------
def r7_keep_close_for_high_priority(call: Dict[str, Any]) -> RuleResult:
    """
    R7: For low-priority calls, try to *keep the very closest units free*.

    Assumptions:
      - CAD priority: 1 = highest, larger numbers = lower priority.
      - We'll treat priority >= 3 as "low" for now (tunable).

    Output:
      keep_close_flag = True  -> policy should *avoid* consuming the very nearest unit
                                 if there is a slightly farther alternative.
      keep_close_flag = False -> no special protection; OK to use nearest.
    """
    prio_raw = call.get("priority", None)

    try:
        priority = int(prio_raw)
    except (TypeError, ValueError):
        # If we can't parse it, treat as mid priority (no special handling)
        priority = 2

    is_low_priority = priority >= 3
    return RuleResult(
        keep_close_flag=is_low_priority
    )

# -----------------------------
# R8 — Hospital load rule (keep some units clear)
# -----------------------------

# Approximate busy ED hours (tunable)
BUSY_HOURS = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}  # 10:00–21:59

def r8_hospital_load(call: Dict[str, Any]) -> RuleResult:
    """
    R8: Adjust behavior based on approximate hospital load.

    Heuristic:
      - Use call["tod_min"] (minutes since midnight) to derive hour-of-day.
      - If hour is in BUSY_HOURS, assume ED is busier.
    
    Outputs:
      fairness_weight: >1.0 during busy hours, 1.0 otherwise.
      keep_free_flag:  True during busy hours to signal policies to avoid
                       unnecessarily committing extra units to long transports.
    """
    tod_min = call.get("tod_min", None)
    try:
        tod_min = int(tod_min)
    except (TypeError, ValueError):
        # If no valid time-of-day, do nothing special
        return RuleResult()

    hour = (tod_min // 60) % 24
    busy = hour in BUSY_HOURS

    fairness_weight = 1.2 if busy else 1.0
    keep_free = busy  # “be conservative” when ED is busy

    return RuleResult(
        fairness_weight=fairness_weight,
        keep_free_flag=keep_free
    )

# -----------------------------
# R9 — Risk-score rule (severity → weight)
# -----------------------------

# Simple keyword buckets; tune once you inspect real CAD text
R9_SEVERE = {
    "cardiac", "chest pain", "not breathing", "no pulse",
    "respiratory arrest", "gunshot", "stabbing", "overdose",
    "unconscious", "seizure", "stroke"
}

R9_MODERATE = {
    "difficulty breathing", "shortness of breath", "altered mental status",
    "fall", "trauma", "bleeding", "vehicle accident", "mvc",
    "assault", "head injury"
}

R9_LOW = {
    "sick person", "flu", "fever", "weakness", "abdominal pain",
    "general illness", "public assist", "lift assist", "minor injury",
    "ankle", "wrist", "laceration", "non-emergent"
}

def r9_risk_score(call: Dict[str, Any]) -> RuleResult:
    """
    R9: Assign a continuous-ish risk score in [0,1] based on:
      1) CAD priority code if present (1–4, or 'alpha/bravo/...').
      2) Fallback to description keyword buckets.

    Higher = more severe = policy should favor lower ETA, ALS, etc.
    """

    desc = str(call.get("description", "")).lower()
    incident_type = str(call.get("incidentType", "")).lower()
    pri_raw = call.get("priority") or call.get("cad_priority") or call.get("determinant")

    # --- 1) Try to use explicit CAD priority if we have it ---
    score_from_priority = None
    if pri_raw is not None:
        p = str(pri_raw).strip().lower()

        # Numeric style: 1 (highest) .. 4 (lowest)
        if p.isdigit():
            n = int(p)
            # compress into [0.2, 0.95]
            if n <= 1:
                score_from_priority = 0.95
            elif n == 2:
                score_from_priority = 0.75
            elif n == 3:
                score_from_priority = 0.5
            else:
                score_from_priority = 0.25

        else:
            # Determinant style: alpha/bravo/charlie/delta/echo, etc.
            if p.startswith("e"):   # Echo — life threatening
                score_from_priority = 0.98
            elif p.startswith("d"):
                score_from_priority = 0.9
            elif p.startswith("c"):
                score_from_priority = 0.75
            elif p.startswith("b"):
                score_from_priority = 0.5
            elif p.startswith("a"):
                score_from_priority = 0.3

    # --- 2) Fallback: text buckets from description / incident_type ---
    if score_from_priority is None:
        text = f"{desc} {incident_type}"
        t = text.lower()

        if any(k in t for k in R9_SEVERE):
            score = 0.9
        elif any(k in t for k in R9_MODERATE):
            score = 0.6
        elif any(k in t for k in R9_LOW):
            score = 0.3
        else:
            score = 0.15  # unknown → low-but-nonzero
    else:
        score = score_from_priority

    return RuleResult(risk_score=score)

def r10_time_of_day(call: Dict[str, Any]) -> RuleResult:
    """
    R10: Time-of-day modifier.

    Example convention:
      - Night:   00:00–06:00  -> tag 'night'
      - Peak:    06:00–10:00, 16:00–20:00 -> tag 'peak'
      - Offpeak: everything else.

    Policies can use time_of_day_tag to bias choices.
    """

    tod = int(call.get("tod_min", 0))  # minutes since midnight
    hour = tod // 60

    if 0 <= hour < 6:
        tag = "night"
    elif 6 <= hour < 10 or 16 <= hour < 20:
        tag = "peak"
    else:
        tag = "offpeak"

    return RuleResult(time_of_day_tag=tag)


# -----------------------------
# RULE REGISTRY FOR PRUNING (per-call rules only)
# -----------------------------
RULES = {
    "R1": r1_nearest_eta,
    "R2": r2_als_capability,
    "R3": r3_bls_capability,
    # R4 is intentionally NOT in this registry because it needs units+zone_lookup.
    "R5": r5_fairness,
    "R6": r6_zone_penalty,
    "R7": r7_keep_close_for_high_priority,
    "R8": r8_hospital_load,
    "R9": r9_risk_score,
    "R10": r10_time_of_day,
}

def apply_rules(rule_names: List[str], call: Dict[str, Any]) -> List[RuleResult]:
    """Run selected per-call rules (R1–R3, etc.) and return RuleResults."""
    results: List[RuleResult] = []
    for r in rule_names:
        fn = RULES.get(r)
        if fn is None:
            continue
        results.append(fn(call))
    return results
