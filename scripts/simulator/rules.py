# scripts/simulator/rules.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple

from scripts.simulator import config


@dataclass
class RuleResult:
    """
    Aggregated outcome of applying a set of rule primitives to a (call, unit).
    All fields are optional; policies can pick what they care about.

    Conventions:
      - eta_score: lower is better (e.g., ETA in minutes or normalized).
      - coverage_loss: higher means worse coverage after taking this unit.
      - fairness_weight: multiplicative weight to up/down-weight calls that
        are currently under-served (e.g., rural/urban gap).
      - risk_weight: multiplicative weight based on medical risk.
      - als_pref_score: penalty if we violate ALS/BLS preference.
      - debug: per-rule trace.
    """
    eta_score: float | None = None
    coverage_loss: float | None = None
    fairness_weight: float = 1.0
    risk_weight: float = 1.0
    als_pref_score: float | None = None
    debug: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _get_call_area(call: Dict[str, Any]) -> str:
    # Prefer explicit call_area; fall back to urban_rural.
    val = (call.get("call_area") or call.get("urban_rural") or "unknown").lower()
    if val not in {"urban", "rural", "unknown"}:
        return "unknown"
    return val


def _get_unit_area(unit: Any) -> str:
    val = getattr(unit, "unit_area", None)
    if not val:
        return "unknown"
    val = str(val).lower()
    if val not in {"urban", "rural", "unknown"}:
        return "unknown"
    return val


def _get_call_zone(call: Dict[str, Any]) -> str | None:
    z = call.get("zone")
    return str(z) if z is not None else None


def _get_unit_zone(unit: Any) -> str | None:
    z = getattr(unit, "zone", None)
    return str(z) if z is not None else None


def _get_risk_score(call: Dict[str, Any]) -> float:
    rs = call.get("risk_score")
    try:
        return float(rs) if rs is not None else 0.0
    except Exception:
        return 0.0


def _get_severity_bucket(call: Dict[str, Any]) -> str:
    val = (call.get("severity_bucket") or "unknown").lower()
    if val not in {"low", "medium", "high", "unknown"}:
        return "unknown"
    return val


def _get_tod_hour(call: Dict[str, Any]) -> int | None:
    tod_min = call.get("tod_min")
    try:
        m = int(tod_min)
    except Exception:
        return None
    return (m // 60) % 24


# ----------------------------------------------------------------------
# Individual rules R1–R10
# These are intentionally simple and only use tagged fields, no geometry.
# ----------------------------------------------------------------------
def R1_nearest_eta(call: Dict[str, Any], unit: Any, now_min: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    R1: Nearest-ETA style score.
    The actual ETA calculation should be done in the policy using traffic.travel_minutes;
    here we just pass through an 'eta_min' value if the policy provided it in call['eta_hint'].
    """
    eta = call.get("eta_hint")  # optional precomputed ETA in minutes
    try:
        eta_val = float(eta) if eta is not None else None
    except Exception:
        eta_val = None

    contrib: Dict[str, Any] = {}
    if eta_val is not None:
        contrib["eta_score"] = eta_val

    dbg = {
        "rule": "R1_nearest_eta",
        "eta_hint": eta_val,
    }
    return contrib, dbg


def R2_als_bls_capability(call: Dict[str, Any], unit: Any, now_min: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    R2: ALS/BLS capability vs call preference.
    Penalize mismatch: high-risk/ALS-preferred calls being covered by BLS units.
    This does NOT enforce hard constraints; policies can use als_pref_score.
    """
    risk = _get_risk_score(call)
    sev = _get_severity_bucket(call)
    pref = (call.get("preferred_unit_type") or "").upper()
    utype = (getattr(unit, "utype", "") or "").upper()

    # Default: no penalty
    penalty = 0.0

    # If call prefers ALS (or high-risk), penalize BLS
    if risk >= getattr(config, "HIGH_RISK_THRESHOLD", 0.75) or sev == "high" or pref == "ALS":
        if utype == "BLS":
            penalty = getattr(config, "ALS_MISMATCH_PENALTY", 1.0)

    contrib = {"als_pref_score": penalty}
    dbg = {
        "rule": "R2_als_bls_capability",
        "risk_score": risk,
        "severity_bucket": sev,
        "preferred_unit_type": pref,
        "unit_type": utype,
        "als_pref_score": penalty,
    }
    return contrib, dbg


def R3_boundary_guardrails(call: Dict[str, Any], unit: Any, now_min: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    R3: Simple ALS/BLS boundary awareness.
    Uses pre-tagged boolean flags from the call; does not load any polygons.
    """
    in_als = bool(call.get("in_als_boundary", False))
    in_bls = bool(call.get("in_bls_boundary", False))
    in_overlap = bool(call.get("in_overlap_boundary", False))
    utype = (getattr(unit, "utype", "") or "").upper()

    penalty = 0.0
    if in_als and utype == "BLS":
        penalty = getattr(config, "ALS_BOUNDARY_BLS_PENALTY", 1.0)
    if in_bls and utype == "ALS":
        penalty = getattr(config, "BLS_BOUNDARY_ALS_PENALTY", 0.5)
    if in_overlap:
        penalty *= getattr(config, "OVERLAP_BOUNDARY_MULT", 0.5)

    contrib = {"boundary_penalty": penalty}
    dbg = {
        "rule": "R3_boundary_guardrails",
        "in_als_boundary": in_als,
        "in_bls_boundary": in_bls,
        "in_overlap_boundary": in_overlap,
        "unit_type": utype,
        "boundary_penalty": penalty,
    }
    return contrib, dbg


def R4_zone_coverage(call: Dict[str, Any], unit: Any, now_min: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    R4: Zone-aware coverage penalty.
    If unit leaves its 'home' zone to cover another zone, add a coverage_loss.
    """
    call_zone = _get_call_zone(call)
    unit_zone = _get_unit_zone(unit)

    coverage_loss = 0.0
    if unit_zone and call_zone and unit_zone != call_zone:
        coverage_loss = getattr(config, "CROSS_ZONE_COVERAGE_LOSS", 1.0)

    contrib = {"coverage_loss": coverage_loss}
    dbg = {
        "rule": "R4_zone_coverage",
        "call_zone": call_zone,
        "unit_zone": unit_zone,
        "coverage_loss": coverage_loss,
    }
    return contrib, dbg


def R5_fairness(call: Dict[str, Any], unit: Any, now_min: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    R5: Fairness – use tagged call_area / urban_rural, no geometry.
    Idea: boost calls in under-served areas (e.g., rural) by a multiplicative weight.

    We DO NOT measure fairness gap inside the rule. Instead, we:
      - Tag each decision with call_area + unit_area (DES debug).
      - Use build_full_kpis to measure rural vs urban response gaps.

    Here we only define a static prior weight, e.g.:
      - rural: fairness_weight > 1
      - urban: fairness_weight = 1
    """
    call_area = _get_call_area(call)         # "urban" / "rural" / "unknown"
    unit_area = _get_unit_area(unit)         # from unit.unit_area
    base = 1.0

    rural_boost = getattr(config, "FAIRNESS_RURAL_WEIGHT", 1.2)
    urban_boost = getattr(config, "FAIRNESS_URBAN_WEIGHT", 1.0)

    if call_area == "rural":
        fairness_weight = rural_boost
    elif call_area == "urban":
        fairness_weight = urban_boost
    else:
        fairness_weight = base

    contrib = {"fairness_weight": fairness_weight}
    dbg = {
        "rule": "R5_fairness",
        "call_area": call_area,
        "unit_area": unit_area,
        "fairness_weight": fairness_weight,
    }
    return contrib, dbg


def R6_high_demand_zone(call: Dict[str, Any], unit: Any, now_min: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    R6: High-demand zone boost.
    Uses a simple config-driven list of high-demand zones (e.g., downtown).
    """
    call_zone = _get_call_zone(call)
    high_demand_zones = getattr(config, "HIGH_DEMAND_ZONES", [])
    boost = 1.0
    if call_zone in high_demand_zones:
        boost = getattr(config, "HIGH_DEMAND_WEIGHT", 1.1)

    contrib = {"high_demand_weight": boost}
    dbg = {
        "rule": "R6_high_demand_zone",
        "call_zone": call_zone,
        "high_demand_weight": boost,
    }
    return contrib, dbg


def R7_overcoverage_penalty(call: Dict[str, Any], unit: Any, now_min: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    R7: Over-coverage penalty.
    Placeholder: penalize sending units from zones marked as 'scarce'.
    """
    unit_zone = _get_unit_zone(unit)
    scarce_zones = getattr(config, "SCARCE_ZONES", [])
    penalty = 0.0
    if unit_zone in scarce_zones:
        penalty = getattr(config, "SCARCE_ZONE_PENALTY", 1.0)

    contrib = {"overcoverage_penalty": penalty}
    dbg = {
        "rule": "R7_overcoverage_penalty",
        "unit_zone": unit_zone,
        "overcoverage_penalty": penalty,
    }
    return contrib, dbg


def R8_time_of_day(call: Dict[str, Any], unit: Any, now_min: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    R8: Time-of-day adjustment.
    Example: additional weight during peak hours.
    """
    hour = _get_tod_hour(call)
    peak_hours = getattr(config, "PEAK_HOURS", [7, 8, 9, 16, 17, 18])
    peak_weight = getattr(config, "PEAK_HOUR_WEIGHT", 1.1)

    weight = 1.0
    if hour is not None and hour in peak_hours:
        weight = peak_weight

    contrib = {"tod_weight": weight}
    dbg = {
        "rule": "R8_time_of_day",
        "hour": hour,
        "tod_weight": weight,
    }
    return contrib, dbg


def R9_risk_weight(call: Dict[str, Any], unit: Any, now_min: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    R9: Risk-based weighting using risk_score / severity_bucket.
    High severity / high risk -> higher weight.
    """
    risk = _get_risk_score(call)
    sev = _get_severity_bucket(call)

    base = 1.0
    if sev == "high" or risk >= getattr(config, "HIGH_RISK_THRESHOLD", 0.75):
        weight = getattr(config, "HIGH_RISK_WEIGHT", 1.5)
    elif sev == "medium":
        weight = getattr(config, "MEDIUM_RISK_WEIGHT", 1.2)
    else:
        weight = base

    contrib = {"risk_weight": weight}
    dbg = {
        "rule": "R9_risk_weight",
        "risk_score": risk,
        "severity_bucket": sev,
        "risk_weight": weight,
    }
    return contrib, dbg


def R10_debug_tag(call: Dict[str, Any], unit: Any, now_min: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    R10: No-op rule used to just capture call/unit context into debug.
    Policies can include this to make sure decisions.csv has enough features.
    """
    contrib: Dict[str, Any] = {}
    dbg = {
        "rule": "R10_debug_tag",
        "call_id": call.get("id"),
        "unit_name": getattr(unit, "name", None),
        "unit_type": getattr(unit, "utype", None),
        "call_area": _get_call_area(call),
        "unit_area": _get_unit_area(unit),
        "zone_call": _get_call_zone(call),
        "zone_unit": _get_unit_zone(unit),
    }
    return contrib, dbg


# Registry
RULES = {
    "R1": R1_nearest_eta,
    "R2": R2_als_bls_capability,
    "R3": R3_boundary_guardrails,
    "R4": R4_zone_coverage,
    "R5": R5_fairness,
    "R6": R6_high_demand_zone,
    "R7": R7_overcoverage_penalty,
    "R8": R8_time_of_day,
    "R9": R9_risk_weight,
    "R10": R10_debug_tag,
}


def apply_rules(
    rule_names: Iterable[str],
    call: Dict[str, Any],
    unit: Any,
    now_min: float,
) -> RuleResult:
    """
    Apply a set of rules to (call, unit) and merge outputs into a RuleResult.

    Policies can call this, then combine:
      - result.eta_score
      - result.coverage_loss
      - result.fairness_weight
      - result.risk_weight
      - result.als_pref_score
    into a final scalar score or priority.
    """
    out = RuleResult()
    all_debug: Dict[str, Any] = {}

    for name in rule_names:
        fn = RULES.get(name)
        if fn is None:
            continue
        contrib, dbg = fn(call, unit, now_min)
        all_debug[name] = dbg

        if "eta_score" in contrib:
            out.eta_score = contrib["eta_score"]
        if "coverage_loss" in contrib:
            out.coverage_loss = (
                (out.coverage_loss or 0.0) + float(contrib["coverage_loss"])
            )
        if "fairness_weight" in contrib:
            out.fairness_weight *= float(contrib["fairness_weight"])
        if "risk_weight" in contrib:
            out.risk_weight *= float(contrib["risk_weight"])
        if "als_pref_score" in contrib:
            out.als_pref_score = (out.als_pref_score or 0.0) + float(
                contrib["als_pref_score"]
            )

    out.debug = all_debug
    return out