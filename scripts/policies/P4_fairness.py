# P4: Urban/rural fairness baseline
from __future__ import annotations

from typing import Any, Dict, List

from scripts.policies.common import (
    BasePolicy,
    CallDict,
    PolicyResult,
    UnitLike,
    _filter_dispatchable,
    call_urban_rural,
    unit_urban_rural,
    r1_eta_minutes,
    r10_random_tiebreaker,
)


class UrbanRuralFairnessPolicy(BasePolicy):
    """
    P4: Fairness baseline focused on urban/rural.

    Behavior:
      - Start from P1: pick the unit with smallest ETA.
      - Add ONE fairness rule:
          * For URBAN calls, avoid stripping RURAL coverage if possible.
            - Penalize sending a rural unit to an urban call, especially
              if it would leave rural with no free units.
      - Always dispatch a unit if any is available; never drop a call
        just to preserve fairness.

    No ALS/BLS, no severity, no demand, no keep-free/keep-close.
    """

    name = "p4_fairness"

    # We want to keep at least this many rural units free if possible
    MIN_RURAL_FREE = 1

    # Penalties (in ETA minutes) when using a rural unit for an urban call
    # SMALL: rural still has some spare units after dispatch
    # BIG: sending this unit would leave rural at or below MIN_RURAL_FREE
    SMALL_RURAL_TO_URBAN_PENALTY = 3.0
    BIG_RURAL_TO_URBAN_PENALTY = 10.0

    # Scale to convert penalty into a fairness_weight for logging
    FAIRNESS_ALPHA = 10.0  # minutes per unit of fairness_weight above 1.0

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        # 1) Dispatchable candidates (same as P1/P2/P3)
        candidates = _filter_dispatchable(units, now_min=now_min)

        if not candidates:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_dispatchable_units",
                "n_candidates": 0,
            }

        # 2) Compute current free units by area, focusing on urban/rural
        free_by_area: Dict[str, int] = {}
        for u in candidates:
            area = unit_urban_rural(u)  # "urban", "rural", or "unknown"
            free_by_area[area] = free_by_area.get(area, 0) + 1

        rural_free = free_by_area.get("rural", 0)

        call_area = call_urban_rural(call)  # "urban", "rural", or "unknown"

        best_unit: UnitLike | None = None
        best_eta: float = float("inf")
        best_score: float = float("inf")
        best_debug: Dict[str, Any] = {}

        # 3) Score each candidate: ETA + fairness penalty + jitter
        for u in candidates:
            unit_area = unit_urban_rural(u)
            eta = r1_eta_minutes(u, now_min, call)

            # Default: no fairness penalty
            fairness_penalty = 0.0

            # Urban call pulling a rural unit: penalize
            if call_area == "urban" and unit_area == "rural":
                # If sending this unit would drop rural below the desired reserve,
                # apply a big penalty. Otherwise, a smaller penalty.
                remaining_rural = rural_free - 1
                if remaining_rural < self.MIN_RURAL_FREE:
                    fairness_penalty = self.BIG_RURAL_TO_URBAN_PENALTY
                else:
                    fairness_penalty = self.SMALL_RURAL_TO_URBAN_PENALTY

            # Rural call pulling an urban unit: no fairness penalty.
            # Rural call pulling a rural unit: also no fairness penalty.
            # Unknown areas: no fairness penalty.

            # Derive fairness_weight for logging (>= 1.0)
            fairness_weight = 1.0 + (fairness_penalty / self.FAIRNESS_ALPHA)

            jitter = r10_random_tiebreaker(self._rng)
            score = eta + fairness_penalty + jitter

            if score < best_score:
                best_score = score
                best_eta = eta
                best_unit = u
                best_debug = {
                    "policy": self.name,
                    "best_eta_min": float(best_eta),
                    "best_score": float(best_score),
                    "n_candidates": len(candidates),
                    "call_area": call_area,
                    "unit_area": unit_area,
                    "rural_free_before": rural_free,
                    "min_rural_free": self.MIN_RURAL_FREE,
                    "fairness_penalty": float(fairness_penalty),
                    "fairness_weight": float(fairness_weight),
                    "fairness_alpha": self.FAIRNESS_ALPHA,
                    # Optional simple KPI hook: 1 if we used rural for urban call
                    "rural_to_urban_dispatch": bool(
                        call_area == "urban" and unit_area == "rural"
                    ),
                }

        if best_unit is None:
            # Should not happen in normal operation
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_candidate_after_scoring",
                "n_candidates": len(candidates),
            }

        return best_unit, float(best_eta), best_debug