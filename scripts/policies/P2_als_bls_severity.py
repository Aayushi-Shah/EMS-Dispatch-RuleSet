# P2: ALS/BLS + severity-aware baseline
from __future__ import annotations

from typing import Any, Dict, List

from scripts.policies.common import (
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
    # ... same docstring, name, thresholds, penalties as you already have ...

    name = "p2_als_bls"

    HIGH_RISK_THRESH = 0.75
    LOW_RISK_THRESH = 0.35

    PENALTY_BLS_FOR_HIGH = 1.6
    PENALTY_ALS_FOR_LOW = 1.3

    REQUIRE_ALS_BLS_MULT = 10.0

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        # 1) Dispatchable units (not busy past now_min)
        base_candidates = _filter_dispatchable(units, now_min=now_min)

        if not base_candidates:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_dispatchable_units",
                "n_candidates": 0,
            }

        # 2) Call-level rule signals: capability + risk
        call_signals: Dict[str, Any] = _rule_signals_p2(call, None, now_min)
        als_capable: bool = bool(call_signals["als_capable"])
        bls_capable: bool = bool(call_signals["bls_capable"])
        risk: float = float(call_signals["risk"])

        # ALS requirement: explicit tag OR high-risk
        require_als_flag: bool = bool(call.get("require_als", False))
        require_als: bool = require_als_flag or (risk >= self.HIGH_RISK_THRESH)

        # 3) For baseline P2, do NOT hard-drop BLS by capability.
        #    We keep all dispatchable candidates and let scoring decide.
        candidates = base_candidates

        best_unit: UnitLike | None = None
        best_eta: float = float("inf")
        best_score: float = float("inf")

        # 4) Score each candidate using ETA + severity-aware multipliers
        for u in candidates:
            utype = _norm_utype(u)  # "ALS" / "BLS" / other

            # Base ETA (R1: dispatch delay + travel with traffic)
            eta = r1_eta_minutes(u, now_min, call)

            # Severity-based weighting based on risk and capability hints
            mult = _severity_multiplier(
                utype=utype,
                risk=risk,
                als_capable=als_capable,
                bls_capable=bls_capable,
                high_thresh=self.HIGH_RISK_THRESH,
                low_thresh=self.LOW_RISK_THRESH,
                penalty_bls_for_high=self.PENALTY_BLS_FOR_HIGH,
                penalty_als_for_low=self.PENALTY_ALS_FOR_LOW,
            )

            # If ALS is effectively required, strongly penalize BLS in scoring
            if require_als and utype == "BLS":
                mult *= self.REQUIRE_ALS_BLS_MULT

            busy_pen = r8_busy_load_penalty(u, now_min)
            jitter = r10_random_tiebreaker(self._rng)

            score = eta * mult + busy_pen + jitter

            if score < best_score:
                best_score = score
                best_eta = eta
                best_unit = u

        if best_unit is None:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_candidate_after_scoring",
                "n_candidates": len(candidates),
                "als_capable": als_capable,
                "bls_capable": bls_capable,
                "risk": risk,
                "require_als": require_als,
            }

        debug = {
            "policy": self.name,
            "best_eta_min": float(best_eta),
            "best_score": float(best_score),
            "n_candidates": len(candidates),
            "n_candidates_base": len(base_candidates),
            "als_capable": als_capable,
            "bls_capable": bls_capable,
            "risk": risk,
            "require_als_flag": require_als_flag,
            "require_als": require_als,
            "high_risk_thresh": self.HIGH_RISK_THRESH,
            "low_risk_thresh": self.LOW_RISK_THRESH,
            "penalty_bls_for_high": self.PENALTY_BLS_FOR_HIGH,
            "penalty_als_for_low": self.PENALTY_ALS_FOR_LOW,
            "require_als_bls_mult": self.REQUIRE_ALS_BLS_MULT,
        }
        return best_unit, float(best_eta), debug