# P5: Hybrid baseline (ETA + ALS/BLS + severity + coverage + urban/rural fairness)
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
    call_urban_rural,
    unit_urban_rural,
    r1_eta_minutes,
    r8_busy_load_penalty,
    r10_random_tiebreaker,
)


class HybridPolicy(BasePolicy):
    """
    P5: Hybrid baseline.

    Combines:
      - P1: ETA-based selection.
      - P2: ALS/BLS + severity-aware (R2/R3/R9-based).
      - P3: Coverage preservation (don't strip an area bare if others can spare).
      - P4: Urban/rural fairness (don't strand rural to cover urban if avoidable).

    Always dispatch a unit if any is available; never drop a call just to
    preserve coverage or fairness.

    No demand / underprotection, no keep-free/keep-close in this baseline.
    """

    name = "p5_hybrid"

    # ---- Severity / ALS-BLS thresholds (P2 behavior) ----
    HIGH_RISK_THRESH = 0.75
    LOW_RISK_THRESH = 0.35

    PENALTY_BLS_FOR_HIGH = 1.6   # high-risk call, BLS unit (when ALS capable)
    PENALTY_ALS_FOR_LOW = 1.3    # low-risk call, ALS unit (when BLS capable)

    REQUIRE_ALS_BLS_MULT = 10.0  # extra multiplier when ALS is effectively required

    # ---- Coverage parameters (P3 behavior) ----
    MIN_FREE_PER_AREA = 1            # want at least this many free units per area
    BIG_COVERAGE_PENALTY = 10.0      # discourage stripping area that can't spare
    SMALL_COVERAGE_PENALTY = 0.0     # when everyone is fragile, accept stripping

    # ---- Urban/rural fairness (P4 behavior) ----
    MIN_RURAL_FREE = 1
    SMALL_RURAL_TO_URBAN_PENALTY = 3.0
    BIG_RURAL_TO_URBAN_PENALTY = 10.0
    FAIRNESS_ALPHA = 10.0  # converts penalty → fairness_weight

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        # 1) Dispatchable candidates (same as P1/P2/P3/P4)
        base_candidates = _filter_dispatchable(units, now_min=now_min)

        if not base_candidates:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_dispatchable_units",
                "n_candidates": 0,
            }

        # 2) Call-level ALS/BLS + severity signals (reuse P2 rule helper)
        call_signals: Dict[str, Any] = _rule_signals_p2(call, None, now_min)
        als_capable: bool = bool(call_signals["als_capable"])
        bls_capable: bool = bool(call_signals["bls_capable"])
        risk: float = float(call_signals["risk"])

        require_als_flag: bool = bool(call.get("require_als", False))
        require_als: bool = require_als_flag or (risk >= self.HIGH_RISK_THRESH and als_capable)

        # 3) ALS/BLS boundary constraint: drop BLS when not capable, with fallback
        candidates = _filter_bls_by_capability(base_candidates, bls_capable)

        if not candidates:
            # Should be rare because helper falls back to original list
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_candidates_after_bls_filter",
                "n_candidates": len(base_candidates),
                "als_capable": als_capable,
                "bls_capable": bls_capable,
                "risk": risk,
                "require_als": require_als,
            }

        # 4) Coverage pool and free units per area (P3 style)
        free_by_area: Dict[str, int] = {}
        for u in candidates:
            area = unit_urban_rural(u)  # "urban", "rural", or "unknown"
            free_by_area[area] = free_by_area.get(area, 0) + 1

        # For coverage: can each area spare a unit and still meet target?
        can_spare_by_area: Dict[str, bool] = {}
        for area, count in free_by_area.items():
            can_spare_by_area[area] = (count - 1) >= self.MIN_FREE_PER_AREA

        has_spare_area_candidate = any(
            can_spare_by_area.get(unit_urban_rural(u), False) for u in candidates
        )

        # For fairness: rural pool size and call/area tags
        rural_free = free_by_area.get("rural", 0)
        call_area = call_urban_rural(call)  # "urban", "rural", or "unknown"

        best_unit: UnitLike | None = None
        best_eta: float = float("inf")
        best_score: float = float("inf")
        best_debug: Dict[str, Any] = {}

        # 5) Score each candidate: ETA * severity + coverage_pen + fairness_pen + busy + jitter
        for u in candidates:
            utype = _norm_utype(u)            # "ALS"/"BLS"/other
            unit_area = unit_urban_rural(u)   # "urban"/"rural"/"unknown"

            eta = r1_eta_minutes(u, now_min, call)

            # Severity-based multiplier (P2 behavior)
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

            # Extra push when ALS is effectively required
            if require_als and utype == "BLS":
                mult *= self.REQUIRE_ALS_BLS_MULT

            # Coverage penalty (P3 behavior)
            current_free = free_by_area.get(unit_area, 0)
            remaining_free = current_free - 1

            if can_spare_by_area.get(unit_area, False):
                coverage_pen = 0.0
            else:
                if has_spare_area_candidate:
                    coverage_pen = self.BIG_COVERAGE_PENALTY
                else:
                    coverage_pen = self.SMALL_COVERAGE_PENALTY

            # Urban/rural fairness penalty (P4 behavior)
            fairness_pen = 0.0
            if call_area == "urban" and unit_area == "rural":
                remaining_rural = rural_free - 1
                if remaining_rural < self.MIN_RURAL_FREE:
                    fairness_pen = self.BIG_RURAL_TO_URBAN_PENALTY
                else:
                    fairness_pen = self.SMALL_RURAL_TO_URBAN_PENALTY

            fairness_weight = 1.0 + (fairness_pen / self.FAIRNESS_ALPHA)

            busy_pen = r8_busy_load_penalty(u, now_min)
            jitter = r10_random_tiebreaker(self._rng)

            total_penalty = coverage_pen + fairness_pen
            score = eta * mult + total_penalty + busy_pen + jitter

            if score < best_score:
                best_score = score
                best_eta = eta
                best_unit = u
                best_debug = {
                    "policy": self.name,
                    "best_eta_min": float(best_eta),
                    "best_score": float(best_score),
                    "n_candidates": len(candidates),
                    "n_candidates_base": len(base_candidates),

                    # ALS/BLS + severity context
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
                    "severity_mult": mult,

                    # Coverage context
                    "unit_area": unit_area,
                    "call_area": call_area,
                    "area_current_free": current_free,
                    "area_remaining_free": remaining_free,
                    "min_free_per_area": self.MIN_FREE_PER_AREA,
                    "area_can_spare": bool(can_spare_by_area.get(unit_area, False)),
                    "has_spare_area_candidate": has_spare_area_candidate,
                    # Simple coverage_loss: 0 if not stripping below target, 1 otherwise
                    "coverage_loss": 0.0 if can_spare_by_area.get(unit_area, False) else 1.0,
                    "coverage_penalty_applied": float(coverage_pen),

                    # Fairness context
                    "rural_free_before": rural_free,
                    "min_rural_free": self.MIN_RURAL_FREE,
                    "fairness_penalty": float(fairness_pen),
                    "fairness_weight": float(fairness_weight),
                    "fairness_alpha": self.FAIRNESS_ALPHA,
                    "rural_to_urban_dispatch": bool(
                        call_area == "urban" and unit_area == "rural"
                    ),

                    # Busy load
                    "busy_penalty": float(busy_pen),
                }

        if best_unit is None:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_candidate_after_scoring",
                "n_candidates": len(candidates),
            }

        return best_unit, float(best_eta), best_debug