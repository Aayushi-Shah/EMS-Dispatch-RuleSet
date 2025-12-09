# experiments/rulelist_policies/rules/rule_templates.py
from __future__ import annotations

from typing import Any, Sequence, Optional
import config
from policies.base import unit_can_cover_call


# -----------------------
# Generic field accessor
# -----------------------

def _get(obj: Any, field: str, default=None):
    """
    Safely get a field from either a dict-like or attribute-like object.
    """
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


# -----------------------
# Basic predicates
# -----------------------

def _call_low_priority(call: Any) -> bool:
    """
    'Low priority' for R1 = anything that's clearly NOT high.
    """
    sev = _get(call, "severity_bucket", None)
    if not isinstance(sev, str):
        return False
    sev = sev.lower()
    return sev in {"low", "medium"}


def _is_als_unit(u: Any) -> bool:
    """
    ALS iff utype == 'ALS' (case-insensitive).
    """
    utype = _get(u, "utype", None)
    return isinstance(utype, str) and utype.upper() == "ALS"


def _is_bls_unit(u: Any) -> bool:
    """
    BLS iff utype == 'BLS' (case-insensitive).
    """
    utype = _get(u, "utype", None)
    return isinstance(utype, str) and utype.upper() == "BLS"


# -----------------------
# ETA helper
# -----------------------

def _best_unit_within_k(
    call: Any,
    sim_state: Any,
    candidates: Sequence[Any],
    k_minutes: float,
    eta_near: float,
    predicate,
) -> Optional[Any]:
    """
    Among candidates satisfying predicate(u) and ETA(u) <= eta_near + k_minutes,
    return the one with minimum ETA. If none exist, return None.
    """
    best_u: Optional[Any] = None
    best_eta: Optional[float] = None

    for u in candidates:
        if not predicate(u):
            continue

        eta = float(sim_state.eta_to_scene(u, call))
        if eta <= eta_near + k_minutes:
            if best_eta is None or eta < best_eta:
                best_eta = eta
                best_u = u

    return best_u


# -----------------------
# Rule R1 – BLS for low priority
# -----------------------

def prefer_bls_for_low_priority(
    call: Any,
    candidates: Sequence[Any],
    sim_state: Any,
    u_near: Any,
    eta_near: float,
    k_minutes: float,
) -> Any:
    """
    Rule 1: If call is LOW priority and nearest is ALS, try to send a BLS
    (or non-ALS) unit within +K minutes instead.
    """
    if not _call_low_priority(call):
        return u_near

    if not _is_als_unit(u_near):
        return u_near

    # Find a BLS (or non-ALS) candidate within +K minutes
    def _is_bls_or_nonals(u: Any) -> bool:
        return _is_bls_unit(u) or (not _is_als_unit(u))

    alt = _best_unit_within_k(
        call=call,
        sim_state=sim_state,
        candidates=candidates,
        k_minutes=k_minutes,
        eta_near=eta_near,
        predicate=_is_bls_or_nonals,
    )

    return alt if alt is not None else u_near

def protect_last_unit_in_muni_for_low_priority(
    call: Any,
    candidates: Sequence[Any],
    sim_state: Any,
    u_near: Any,
    eta_near: float,
    k_minutes: float,
) -> Any:
    """
    Rule 2: For low/medium-priority calls, avoid stripping a municipality of
    its last idle unit if there is an alternative from another municipality
    within +K minutes.

    Logic:
      - Only low/medium severity.
      - Let M = call.municipality_std.
      - If nearest unit is in M AND it's the only idle unit in M,
        then look for a unit from outside M whose ETA <= eta_near + K.
      - If such an alternative exists, use it; otherwise keep nearest.
    """
    if not _call_low_priority(call):
        return u_near

    call_muni = _get(call, "municipality_std", None)
    if call_muni is None:
        return u_near

    # Nearest must be from same municipality as the call
    u_near_muni = _get(u_near, "municipality_std", None)
    if u_near_muni != call_muni:
        return u_near

    now = getattr(sim_state, "now_min", None)
    if now is None:
        return u_near

    # Count idle units in this municipality (ALS or BLS, doesn't matter)
    idle_in_muni = [
        u
        for u in getattr(sim_state, "units", [])
        if _get(u, "municipality_std", None) == call_muni
        and _get(u, "can_dispatch", False)
        and _get(u, "busy_until", 0.0) <= now
    ]

    # Only act if this is the last idle unit in that muni
    if len(idle_in_muni) != 1:
        return u_near

    # Sanity: make sure the single idle unit is actually u_near
    if _get(idle_in_muni[0], "name", None) != _get(u_near, "name", None):
        return u_near

    # Look for an alternative candidate from a different municipality
    def _outside_call_muni(u: Any) -> bool:
        return _get(u, "municipality_std", None) != call_muni

    alt = _best_unit_within_k(
        call=call,
        sim_state=sim_state,
        candidates=candidates,
        k_minutes=k_minutes,
        eta_near=eta_near,
        predicate=_outside_call_muni,
    )

    return alt if alt is not None else u_near
