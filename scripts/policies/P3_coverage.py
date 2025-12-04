# P3: Coverage-preserving baseline
from __future__ import annotations

from typing import Dict, List, Any

from scripts.policies.common import (
    BasePolicy,
    CallDict,
    PolicyResult,
    UnitLike,
    _filter_dispatchable,
    unit_urban_rural,
    r1_eta_minutes,
    r10_random_tiebreaker,
)


class CoveragePreservingPolicy(BasePolicy):
    """
    P3: Coverage-preserving baseline.

    Behavior:
      - Start from P1: pick the unit with the smallest ETA (R1).
      - Add ONE extra idea: don't strip an area bare if other areas
        can spare a unit.
      - Always dispatch a unit if any is available; never drop a call
        just to preserve coverage.

    Implementation:
      - Define an "area" via unit_urban_rural(u) / unit_area.
      - For all dispatchable units at now_min, count how many free units
        exist per area.
      - When scoring a candidate unit u:
          * Compute ETA to the call.
          * Compute remaining_free = free_by_area[area] - 1 if we send u.
          * If this area can still meet the coverage target after sending u,
            no coverage penalty.
          * If this area cannot meet the target, but some other area has
            spare capacity (can send a unit and still meet target),
            add a big coverage penalty to discourage stripping this area.
          * If every area is at or below the target, accept stripping
            coverage (small or zero penalty).
    """

    name = "p3_coverage"

    # Minimum free units per area we would like to preserve
    MIN_FREE_PER_AREA = 1

    # Penalties in "ETA minutes" units
    BIG_COVERAGE_PENALTY = 10.0  # when we could have used another area instead
    SMALL_COVERAGE_PENALTY = 0.0  # when everyone is fragile, we just accept it

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        # 1) Dispatchable candidates (same filter as P1/P2)
        candidates = _filter_dispatchable(units, now_min=now_min)

        if not candidates:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_dispatchable_units",
                "n_candidates": 0,
            }

        # 2) Coverage pool and free units per area
        #    (here coverage_pool == candidates, both are dispatchable at now_min)
        coverage_pool = candidates

        free_by_area: Dict[str, int] = {}
        for u in coverage_pool:
            area = unit_urban_rural(u)
            free_by_area[area] = free_by_area.get(area, 0) + 1

        # For each area, precompute whether it can spare a unit and still
        # meet the "MIN_FREE_PER_AREA" target after dispatching one unit.
        can_spare_by_area: Dict[str, bool] = {}
        for area, count in free_by_area.items():
            can_spare_by_area[area] = (count - 1) >= self.MIN_FREE_PER_AREA

        # Is there at least one candidate from an area that can spare a unit?
        has_spare_area_candidate = any(
            can_spare_by_area.get(unit_urban_rural(u), False) for u in candidates
        )

        best_unit: UnitLike | None = None
        best_eta: float = float("inf")
        best_score: float = float("inf")

        best_debug: Dict[str, Any] = {}

        # 3) Score each candidate: ETA + coverage penalty + tiny jitter
        for u in candidates:
            area = unit_urban_rural(u)
            current_free = free_by_area.get(area, 0)
            remaining_free = current_free - 1

            # Base ETA (R1: dispatch delay + travel with traffic)
            eta = r1_eta_minutes(u, now_min, call)

            # Coverage penalty:
            #  - If this area can spare a unit and remain at/above target, no penalty.
            #  - If this area cannot spare a unit, but some other area CAN spare a unit,
            #    apply a big penalty to discourage stripping this area.
            #  - If no area can spare a unit (everyone is fragile), accept stripping
            #    coverage (small or zero penalty).
            if can_spare_by_area.get(area, False):
                coverage_pen = 0.0
            else:
                if has_spare_area_candidate:
                    coverage_pen = self.BIG_COVERAGE_PENALTY
                else:
                    coverage_pen = self.SMALL_COVERAGE_PENALTY

            jitter = r10_random_tiebreaker(self._rng)
            score = eta + coverage_pen + jitter

            if score < best_score:
                best_score = score
                best_eta = eta
                best_unit = u
                best_debug = {
                    "policy": self.name,
                    "best_eta_min": float(best_eta),
                    "best_score": float(best_score),
                    "n_candidates": len(candidates),
                    "min_free_per_area": self.MIN_FREE_PER_AREA,
                    "unit_area": area,
                    "area_current_free": current_free,
                    "area_remaining_free": remaining_free,
                    "area_can_spare": bool(can_spare_by_area.get(area, False)),
                    "has_spare_area_candidate": has_spare_area_candidate,
                    # A simple scalar "coverage_loss" signal for downstream KPIs:
                    #   0.0 if we did NOT strip below target,
                    #   1.0 if we dispatched from an area that could not spare a unit.
                    "coverage_loss": 0.0 if can_spare_by_area.get(area, False) else 1.0,
                    "coverage_penalty_applied": float(coverage_pen),
                }

        if best_unit is None:
            # Should not happen in normal operation
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_candidate_after_scoring",
                "n_candidates": len(candidates),
            }

        return best_unit, float(best_eta), best_debug