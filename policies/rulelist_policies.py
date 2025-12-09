from __future__ import annotations

from typing import Sequence, Dict, Any, Tuple

from tools.delta_eta.delta_eta_logging import log_delta_eta_choice

from .base import (
    DispatchPolicy,
    CallLike,
    CandidateLike,
    SimStateLike,
    unit_can_cover_call,
)

# Import rules from rule_templates (R1 + new R2)
from policies.rule_templates import (
    prefer_bls_for_low_priority,                  # R1
    protect_last_unit_in_muni_for_low_priority,   # R2 (new)
)

# -------------------------------------------------------------------
# Shared helper: compute feasible units, ETA map, and nearest unit
# -------------------------------------------------------------------

K_MINUTES_DEFAULT: float = 7.72  # shared K unless overridden per-policy


def _compute_feasible_and_etas(
    call: CallLike,
    candidates: Sequence[CandidateLike],
    sim_state: SimStateLike,
) -> Tuple[Sequence[CandidateLike], CandidateLike, float, Dict[str, float]]:
    """
    Enforce ALS/BLS feasibility (unit_can_cover_call), compute ETAs for all
    feasible units, and identify the nearest one.

    Returns:
        feasible_units, u_near, eta_near, eta_by_unit
    """
    # 1) Feasibility filter (ALS/BLS rules)
    feasible: list[CandidateLike] = [
        u for u in candidates if unit_can_cover_call(call, u)
    ]

    # If everything is infeasible by our filter, fall back to all candidates
    if not feasible:
        feasible = list(candidates)

    # 2) Compute ETA to scene for each feasible unit
    eta_by_unit: Dict[str, float] = {}
    u_near: CandidateLike | None = None
    eta_near: float | None = None

    for u in feasible:
        uid = getattr(u, "name", None)
        if uid is None:
            continue

        eta = float(sim_state.eta_to_scene(u, call))
        eta_by_unit[uid] = eta

        if eta_near is None or eta < eta_near:
            eta_near = eta
            u_near = u

    if u_near is None or eta_near is None:
        # Fallback: pick first feasible with infinite ETA
        u_near = feasible[0]
        eta_near = float("inf")

    return feasible, u_near, eta_near, eta_by_unit


# -------------------------------------------------------------------
# Rule sequencing
# -------------------------------------------------------------------

def _apply_rule_sequence(
    call: CallLike,
    sim_state: SimStateLike,
    feasible: Sequence[CandidateLike],
    u_near: CandidateLike,
    eta_near: float,
    k_minutes: float,
    rules: Sequence[str],
) -> CandidateLike:
    """
    Apply a sequence of rules (by name: "r1", "r2") on top of nearest ETA.

    Each rule sees:
        - call, candidates, sim_state, u_near, eta_near, k_minutes
    and returns either:
        - the same u_near (no change), or
        - a new CandidateLike (alternative unit).

    We apply them in order; first rule that changes the unit "wins".
    """
    chosen = u_near

    for r in rules:
        r_lower = r.lower()

        if r_lower == "r1":
            new_u = prefer_bls_for_low_priority(
                call=call,
                candidates=feasible,
                sim_state=sim_state,
                u_near=chosen,
                eta_near=eta_near,
                k_minutes=k_minutes,
            )

        elif r_lower == "r2":
            new_u = protect_last_unit_in_muni_for_low_priority(
                call=call,
                candidates=feasible,
                sim_state=sim_state,
                u_near=chosen,
                eta_near=eta_near,
                k_minutes=k_minutes,
            )

        else:
            # Unknown rule label -> skip
            continue

        if new_u is not None and new_u is not chosen:
            chosen = new_u
            break

    return chosen


# -------------------------------------------------------------------
# ΔETA logging (for analysis only)
# -------------------------------------------------------------------

def _log_delta_eta(
    self_policy: DispatchPolicy,
    call: CallLike,
    feasible: Sequence[CandidateLike],
    u_near: CandidateLike,
    eta_by_unit: Dict[str, float],
) -> None:
    """
    Centralized ΔETA logging: only if we have at least 2 feasible units.
    """
    if len(feasible) < 2 or not eta_by_unit:
        return

    scenario_name = getattr(self_policy, "scenario_name", "unknown")

    log_delta_eta_choice(
        scenario=scenario_name,
        call=call,
        nearest_unit=u_near,
        candidate_units=feasible,
        eta_by_unit={
            getattr(u, "name", None): eta_by_unit.get(getattr(u, "name", None))
            for u in feasible
            if getattr(u, "name", None) in eta_by_unit
        },
        unit_id_field="name",
    )


# -------------------------------------------------------------------
# Policy classes (R1, R2, R1+R2)
# -------------------------------------------------------------------

class NearestETAR1Policy(DispatchPolicy):
    """
    nearest_eta_r1:
      - baseline nearest ETA
      - plus R1 (BLS preference for low-priority calls within +K minutes)
    """
    name = "nearest_eta_r1"

    def __init__(self, *args, k_minutes: float = K_MINUTES_DEFAULT, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_minutes = float(k_minutes)

    def choose_unit(
        self,
        call: CallLike,
        candidates: Sequence[CandidateLike],
        sim_state: SimStateLike,
    ) -> CandidateLike:
        feasible, u_near, eta_near, eta_by_unit = _compute_feasible_and_etas(
            call, candidates, sim_state
        )

        chosen = _apply_rule_sequence(
            call=call,
            sim_state=sim_state,
            feasible=feasible,
            u_near=u_near,
            eta_near=eta_near,
            k_minutes=self.k_minutes,
            rules=["r1"],
        )

        _log_delta_eta(self, call, feasible, u_near, eta_by_unit)
        return chosen


class NearestETAR2Policy(DispatchPolicy):
    """
    nearest_eta_r2:
      - baseline nearest ETA
      - plus R2 (protect last idle unit in call municipality for low/medium calls)
    """
    name = "nearest_eta_r2"

    def __init__(self, *args, k_minutes: float = K_MINUTES_DEFAULT, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_minutes = float(k_minutes)

    def choose_unit(
        self,
        call: CallLike,
        candidates: Sequence[CandidateLike],
        sim_state: SimStateLike,
    ) -> CandidateLike:
        feasible, u_near, eta_near, eta_by_unit = _compute_feasible_and_etas(
            call, candidates, sim_state
        )

        chosen = _apply_rule_sequence(
            call=call,
            sim_state=sim_state,
            feasible=feasible,
            u_near=u_near,
            eta_near=eta_near,
            k_minutes=self.k_minutes,
            rules=["r2"],
        )

        _log_delta_eta(self, call, feasible, u_near, eta_by_unit)
        return chosen


class NearestETAR1R2Policy(DispatchPolicy):
    """
    nearest_eta_r1_r2:
      - R1 first (BLS for low-priority),
      - if R1 does NOT change the unit, then R2 (protect last idle unit in muni).
    """
    name = "nearest_eta_r1_r2"

    def __init__(self, *args, k_minutes: float = K_MINUTES_DEFAULT, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_minutes = float(k_minutes)

    def choose_unit(
        self,
        call: CallLike,
        candidates: Sequence[CandidateLike],
        sim_state: SimStateLike,
    ) -> CandidateLike:
        feasible, u_near, eta_near, eta_by_unit = _compute_feasible_and_etas(
            call, candidates, sim_state
        )

        chosen = _apply_rule_sequence(
            call=call,
            sim_state=sim_state,
            feasible=feasible,
            u_near=u_near,
            eta_near=eta_near,
            k_minutes=self.k_minutes,
            rules=["r1", "r2"],
        )

        _log_delta_eta(self, call, feasible, u_near, eta_by_unit)
        return chosen