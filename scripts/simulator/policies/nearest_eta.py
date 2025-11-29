# P1: Nearest ETA
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from scripts.simulator.policies.common import (
    BasePolicy,
    CallDict,
    PolicyResult,
    UnitLike,
    _filter_dispatchable,
    r1_eta_minutes,
    r8_busy_load_penalty,
    r10_random_tiebreaker,
)


class NearestETA(BasePolicy):
    """
    P1: Baseline policy – choose the dispatchable unit with smallest ETA.
    """

    name = "nearest_eta"

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        best_unit = None
        best_score = float("inf")
        best_eta = float("inf")

        # Only consider units that can be dispatched in this segment
        candidates = _filter_dispatchable(units, now_min=now_min)
        if not candidates:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_dispatchable_units",
            }

        for u in candidates:
            eta = r1_eta_minutes(u, now_min, call)
            busy_pen = r8_busy_load_penalty(u, now_min)
            jitter = r10_random_tiebreaker(self._rng)

            score = eta + busy_pen + jitter

            if score < best_score:
                best_score = score
                best_eta = eta
                best_unit = u

        debug = {
            "policy": self.name,
            "best_eta_min": best_eta,
            "best_score": best_score,
            "n_candidates": len(candidates),
        }
        return best_unit, float(best_eta), debug
