# experiments/rulelist_policies/policies/base.py

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


CallLike = Any          # call row / object
CandidateLike = Any     # unit row / object or unit_id
SimStateLike = Any      # sim state / context handle


@runtime_checkable
class DispatchPolicy(Protocol):
    name: str

    def choose_unit(
        self,
        call: CallLike,
        candidates: Sequence[CandidateLike],
        sim_state: SimStateLike,
    ) -> CandidateLike:
        ...


def call_zone_type(call: CallLike) -> str:
    z = getattr(call, "zone", None)
    if isinstance(z, str):
        z_up = z.upper()
        if z_up in ("ALS", "BLS", "OVERLAP"):
            return z_up
    return "UNKNOWN"


def call_needs_als(call: CallLike) -> bool:
    """
    Needs ALS if preferred_unit_type == 'ALS'.
    """
    p = getattr(call, "preferred_unit_type", None)
    if isinstance(p, str):
        return p.upper() == "ALS"
    return False


def unit_capability(unit: CandidateLike) -> str:
    """
    ALS/BLS from units.utype.
    """
    u = getattr(unit, "utype", None)
    if isinstance(u, str):
        u_up = u.upper()
        if u_up in ("ALS", "BLS"):
            return u_up
    return "UNKNOWN"


def unit_can_cover_call(call: CallLike, unit: CandidateLike) -> bool:
    """
    Hard feasibility:

      1. A BLS unit cannot go in ALS boundary.
      2. If ALS is needed, BLS cannot cover it.
    """
    zone = call_zone_type(call)          # ALS / BLS / OVERLAP / UNKNOWN
    needs_als = call_needs_als(call)     # from preferred_unit_type
    cap = unit_capability(unit)          # ALS / BLS / UNKNOWN

    if cap not in ("ALS", "BLS"):
        return False

    # Rule 2: If ALS is needed, BLS cannot cover it.
    if needs_als and cap != "ALS":
        return False

    # Rule 1: A BLS unit cannot go in ALS boundary.
    if zone == "ALS" and cap != "ALS":
        return False

    # BLS or OVERLAP or UNKNOWN zone:
    # - If ALS not needed: ALS or BLS both allowed.
    # - If ALS needed: already enforced above.
    return True