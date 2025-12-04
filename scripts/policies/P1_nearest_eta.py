# scripts/policies/nearest_eta.py
# P1: pure nearest-unit (ETA-only) baseline

from __future__ import annotations

from typing import List

from scripts.policies.common import (
    BasePolicy,
    CallDict,
    PolicyResult,
    UnitLike,
    _filter_dispatchable,
    r1_eta_minutes,
    r10_random_tiebreaker,
)


class NearestETAPolicy(BasePolicy):
    """
    P1: pure nearest-unit baseline.

    Behavior:
      - Filter to dispatchable units (respect can_dispatch and busy_until <= now_min).
      - Compute ETA (R1: dispatch delay + travel with traffic) for each candidate.
      - Add a tiny random jitter to avoid deterministic ties.
      - Select the unit with the smallest score.
      - No ALS/BLS logic, no coverage, no fairness, no severity, no keep-free/keep-close.
    """

    # Keep this name stable to align with existing outputs/metrics
    name = "nearest_eta"

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        # 1) Filter to dispatchable units (not busy past now_min)
        candidates = _filter_dispatchable(units, now_min=now_min)

        if not candidates:
            # No one available to send
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_dispatchable_units",
                "n_candidates": 0,
            }

        best_unit: UnitLike | None = None
        best_eta: float = float("inf")
        best_score: float = float("inf")

        # 2) Pure ETA selection (+ tiny jitter)
        for u in candidates:
            eta = r1_eta_minutes(u, now_min, call)  # R1: dispatch delay + travel
            jitter = r10_random_tiebreaker(self._rng)

            score = eta + jitter  # no other terms

            if score < best_score:
                best_score = score
                best_eta = eta
                best_unit = u

        if best_unit is None:
            # Should be rare: all candidates somehow filtered out mid-loop
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_candidate_after_scoring",
                "n_candidates": len(candidates),
            }

        debug = {
            "policy": self.name,
            "best_eta_min": float(best_eta),
            "best_score": float(best_score),
            "n_candidates": len(candidates),
        }
        return best_unit, float(best_eta), debug