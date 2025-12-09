from __future__ import annotations

from typing import Sequence, Any, Optional
from tools.delta_eta.delta_eta_logging import log_delta_eta_choice

from .base import (
    DispatchPolicy,
    CallLike,
    CandidateLike,
    SimStateLike,
    unit_can_cover_call,
)


class NearestETAPolicy(DispatchPolicy):
    """
    P0: Pure nearest ETA, but with hard ALS/BLS feasibility enforced.

    - If call.preferred_unit_type == 'ALS' -> only ALS units are allowed.
    - If call.zone == 'ALS'               -> only ALS units are allowed.
    - Otherwise, ALS or BLS may be chosen purely by ETA.

    We also log per-call ETA structure for ΔETA analysis.
    """

    name = "nearest_eta"

    def choose_unit(
        self,
        call: CallLike,
        candidates: Sequence[CandidateLike],
        sim_state: SimStateLike,
    ) -> CandidateLike:
        # 1) Enforce ALS/BLS feasibility.
        feasible: list[CandidateLike] = [
            u for u in candidates if unit_can_cover_call(call, u)
        ]

        # If nothing passes feasibility, fall back to all candidates.
        if not feasible:
            feasible = list(candidates)

        # 2) Compute ETAs and find nearest.
        eta_by_unit: dict[Any, float] = {}
        best_unit: Optional[CandidateLike] = None
        best_eta: float | None = None

        for u in feasible:
            eta: float = sim_state.eta_to_scene(u, call)

            # Identify the unit by its designator or fallback to .name on DES units.
            if hasattr(u, "unit_designator"):
                uid = u.unit_designator
            elif isinstance(u, dict):
                uid = u.get("unit_designator")
            elif hasattr(u, "name"):
                uid = u.name
            else:
                uid = None

            if uid is not None:
                eta_by_unit[uid] = eta

            if best_eta is None or eta < best_eta:
                best_eta = eta
                best_unit = u

        # Safety net: there should always be at least one feasible candidate.
        if best_unit is None:
            best_unit = feasible[0]

        # 3) Log ETA structure for ΔETA analysis (only when we have a choice).
        try:
            scenario_name = getattr(self, "scenario_name", "unknown")
            if len(feasible) >= 2 and eta_by_unit:
                log_delta_eta_choice(
                    scenario=scenario_name,
                    call=call,
                    nearest_unit=best_unit,
                    candidate_units=feasible,
                    eta_by_unit=eta_by_unit,
                    unit_id_field="name",
                )
        except Exception:
            # Never kill the sim due to logging
            pass

        return best_unit
