# P2: ALS/BLS + severity aware
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from scripts.simulator.policies.common import (
    BasePolicy,
    CallDict,
    PolicyResult,
    UnitLike,
    _filter_bls_by_capability,
    _filter_dispatchable,
    _norm_utype,
    _rule_signals_p2,
    _severity_multiplier,
    r1_eta_minutes,
    r8_busy_load_penalty,
    r10_random_tiebreaker,
)


class ALSBLSSeverityPolicy(BasePolicy):
    """
    P2: ALS/BLS + severity-aware policy.

    Intent:
      - Use R2/R3 to know if ALS/BLS are *geographically capable* for this call.
      - Use R9 to get a [0,1] risk_score.
      - Prefer ALS for high-risk calls (when ALS-capable).
      - Prefer BLS for low-risk calls (when BLS-capable), to protect ALS capacity.
      - Never send BLS where BLS is not capable (boundary constraint).
    """

    name = "p2_als_bls"

    # Tunable thresholds
    HIGH_RISK_THRESH = 0.75
    LOW_RISK_THRESH = 0.35

    # Penalties (multipliers on ETA)
    PENALTY_BLS_FOR_HIGH = 1.6   # severe call but BLS unit
    PENALTY_ALS_FOR_LOW = 1.3    # low-acuity call but ALS unit

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        signals = _rule_signals_p2(call)
        als_capable = signals["als_capable"]
        bls_capable = signals["bls_capable"]
        risk = signals["risk"]
        keep_free = signals["keep_free"]

        best_unit = None
        best_score = float("inf")
        best_eta = float("inf")

        # Only dispatchable and not busy past now_min
        candidates = _filter_dispatchable(units, now_min=now_min)
        candidates = _filter_bls_by_capability(candidates, bls_capable)

        if not candidates:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_dispatchable_units",
            }

        for u in candidates:
            utype = _norm_utype(u)

            # 2) Never send BLS outside its geographic capability
            if utype == "BLS" and not bls_capable:
                continue

            # 2a) For high-risk calls where ALS is capable, skip BLS
            if risk >= self.HIGH_RISK_THRESH and als_capable and utype == "BLS":
                continue

            # 3) Compute base ETA from R1 (with traffic)
            eta = r1_eta_minutes(u, now_min, call)

            # 4) Severity-based weighting
            mult = _severity_multiplier(
                utype,
                risk,
                als_capable,
                bls_capable,
                self.HIGH_RISK_THRESH,
                self.LOW_RISK_THRESH,
                self.PENALTY_BLS_FOR_HIGH,
                self.PENALTY_ALS_FOR_LOW,
            )

            busy_pen = r8_busy_load_penalty(u, now_min)
            jitter = r10_random_tiebreaker(self._rng)

            score = eta * mult + busy_pen + jitter

            # Hospital load hint: keep a small reserve when flagged busy
            if keep_free:
                score += 0.25

            if score < best_score:
                best_score = score
                best_eta = eta
                best_unit = u

        if best_unit is None:
            # Fallback: no candidate survived boundary filters
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_candidate_after_filters",
                "als_capable": als_capable,
                "bls_capable": bls_capable,
                "risk": risk,
                "keep_free_flag": keep_free,
            }

        debug = {
            "policy": self.name,
            "best_eta_min": best_eta,
            "best_score": best_score,
            "n_candidates": len(candidates),
            "als_capable": als_capable,
            "bls_capable": bls_capable,
            "risk": risk,
            "keep_free_flag": keep_free,
        }
        return best_unit, float(best_eta), debug
