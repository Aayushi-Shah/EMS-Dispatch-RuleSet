# Configurable feature-flag policy for bottom-up construction
from __future__ import annotations

from typing import Any, Dict, List, Optional

from scripts.simulator.policies.common import (
    BasePolicy,
    CallDict,
    PolicyResult,
    UnitLike,
    _filter_bls_by_capability,
    _filter_dispatchable,
    _hav_miles,
    _norm_utype_cached,
    _rule_signals_p2,
    _rule_signals_p4,
    call_urban_rural,
    unit_urban_rural,
    r1_eta_minutes,
    r8_busy_load_penalty,
    r10_random_tiebreaker,
)


class FeatureFlagPolicy(BasePolicy):
    """
    Bottom-up configurable policy built from a set of feature flags:
      - Severity/ALS-BLS handling
      - Coverage (unit area / call area) with optional guardrail and boosts
      - Fairness (area mismatch + same-region)
      - Zone weighting (underprotected/high demand)
    """

    name = "feature_flag"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Feature flags (defaults give a middle-ground policy)
        self.use_severity = kwargs.get("use_severity", False)
        self.use_als_guard = kwargs.get("use_als_guard", False)
        self.use_coverage_unit = kwargs.get("use_coverage_unit", False)
        self.use_coverage_call = kwargs.get("use_coverage_call", False)
        self.use_guardrail = kwargs.get("use_guardrail", False)
        self.use_urban_boost = kwargs.get("use_urban_boost", False)
        self.use_fairness = kwargs.get("use_fairness", False)
        self.use_fairness_weight = kwargs.get("use_fairness_weight", False)
        self.use_zone_weights = kwargs.get("use_zone_weights", False)

        # Tunables
        self.high_risk_thresh = kwargs.get("high_risk_thresh", 0.75)
        self.low_risk_thresh = kwargs.get("low_risk_thresh", 0.35)
        self.penalty_bls_high = kwargs.get("penalty_bls_high", 1.6)
        self.penalty_als_low = kwargs.get("penalty_als_low", 1.3)
        self.last_unit_bump = kwargs.get("last_unit_bump", 1.0)
        self.rural_same_region_radius_mi = kwargs.get("rural_same_region_radius_mi", 10.0)
        self.urban_same_region_radius_mi = kwargs.get("urban_same_region_radius_mi", 5.0)

        self.urban_radius = kwargs.get("urban_radius", 1.0)
        self.urban_min_free = kwargs.get("urban_min_free", 3)
        self.rural_radius = kwargs.get("rural_radius", 3.0)
        self.rural_min_free = kwargs.get("rural_min_free", 2)
        self.beta_coverage = kwargs.get("beta_coverage", 2.0)
        self.urban_call_boost = kwargs.get("urban_call_boost", 1.5)
        self.area_mismatch_penalty = kwargs.get("area_mismatch_penalty", 0.5)
        self.gamma_fairness = kwargs.get("gamma_fairness", 1.5)

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        signals = _rule_signals_p2(call)  # capability, risk, keep_free
        fairness_signals = _rule_signals_p4(call)  # fairness_w, keep flags
        als_capable = signals["als_capable"]
        bls_capable = signals["bls_capable"]
        risk = signals["risk"]
        keep_free = signals["keep_free"] or fairness_signals["keep_free"]
        fairness_w = fairness_signals["fairness_w"]

        # Candidate units: can_dispatch and free by now_min
        candidates = _filter_dispatchable(units, now_min=now_min)
        candidates = _filter_bls_by_capability(candidates, bls_capable)
        if not candidates:
            return None, float("inf"), {"policy": self.name, "reason": "no_dispatchable_units"}

        # Coverage pool (free units at this moment)
        coverage_pool: List[UnitLike] = [
            u for u in units
            if getattr(u, "can_dispatch", True)
            and float(getattr(u, "busy_until", 0.0) or 0.0) <= now_min + 1e-9
        ]
        total_free = len(coverage_pool)

        c_lon, c_lat = float(call["lon"]), float(call["lat"])
        call_area = call_urban_rural(call)
        call_is_urban = call_area == "urban"

        best_unit: Optional[UnitLike] = None
        best_eta = float("inf")
        best_score = float("inf")
        best_debug: Dict[str, Any] = {}

        for u in candidates:
            utype = _norm_utype_cached(u)

            # ALS guard for high risk
            if self.use_als_guard and risk >= self.high_risk_thresh and als_capable and utype == "BLS":
                continue

            eta = r1_eta_minutes(u, now_min, call)
            base_score = eta
            base_score += r8_busy_load_penalty(u, now_min)
            if keep_free:
                base_score += 0.2

            # Severity multiplier
            if self.use_severity:
                mult = 1.0
                if risk >= self.high_risk_thresh and als_capable and utype == "BLS":
                    mult = self.penalty_bls_high
                elif risk <= self.low_risk_thresh and utype == "ALS":
                    mult = self.penalty_als_low
                base_score *= mult

            # Coverage
            coverage_mult = 1.0
            if self.use_coverage_unit or self.use_coverage_call:
                u_lon, u_lat = float(u.lon), float(u.lat)
                u_area = unit_urban_rural(u)
                if u_area == "urban":
                    unit_radius = self.urban_radius
                    unit_min_free = self.urban_min_free
                elif u_area == "rural":
                    unit_radius = self.rural_radius
                    unit_min_free = self.rural_min_free
                else:
                    unit_radius = self.rural_radius
                    unit_min_free = self.rural_min_free

                if call_area == "urban":
                    call_radius = self.urban_radius
                    call_min_free = self.urban_min_free
                elif call_area == "rural":
                    call_radius = self.rural_radius
                    call_min_free = self.rural_min_free
                else:
                    call_radius = self.rural_radius
                    call_min_free = self.rural_min_free

                remaining_unit = remaining_call = 0
                for other in coverage_pool:
                    if other is u:
                        continue
                    o_lon, o_lat = float(other.lon), float(other.lat)
                    if self.use_coverage_unit and _hav_miles(u_lon, u_lat, o_lon, o_lat) <= unit_radius:
                        remaining_unit += 1
                    if self.use_coverage_call and _hav_miles(c_lon, c_lat, o_lon, o_lat) <= call_radius:
                        remaining_call += 1

                unit_loss = 0.0
                if self.use_coverage_unit and unit_min_free > 0:
                    unit_loss = max(0.0, (unit_min_free - remaining_unit) / unit_min_free)
                call_loss = 0.0
                if self.use_coverage_call and call_min_free > 0:
                    call_loss = max(0.0, (call_min_free - remaining_call) / call_min_free)

                coverage_loss = 0.4 * unit_loss + 0.6 * call_loss
                if self.use_urban_boost and call_is_urban:
                    coverage_loss *= self.urban_call_boost
                if self.use_zone_weights:
                    # Zone weights would come from call tags; use risk proxy if none
                    coverage_loss *= (1.0 + 0.2 * min(max(risk, 0.0), 1.0))
                if self.use_guardrail and total_free > 1 and remaining_unit == 0:
                    coverage_loss += self.last_unit_bump

                coverage_mult = 1.0 + self.beta_coverage * max(0.0, coverage_loss)

            # Fairness
            fairness_mult = 1.0
            fairness_penalty = 0.0
            if self.use_fairness:
                u_lon, u_lat = float(u.lon), float(u.lat)
                u_area = unit_urban_rural(u)
                d_call = _hav_miles(c_lon, c_lat, u_lon, u_lat)
                if u_area != "unknown" and call_area != "unknown" and u_area != call_area:
                    fairness_penalty += self.area_mismatch_penalty
                if call_area == "rural" and u_area == "rural" and d_call > self.rural_same_region_radius_mi:
                    fairness_penalty += self.area_mismatch_penalty
                if call_area == "urban" and u_area == "urban" and d_call > self.urban_same_region_radius_mi:
                    fairness_penalty += self.area_mismatch_penalty
                fairness_mult = 1.0 + self.gamma_fairness * fairness_penalty
                if self.use_fairness_weight:
                    fairness_mult /= max(1.0, fairness_w)

            score = base_score * coverage_mult * fairness_mult
            score += r10_random_tiebreaker(self._rng)

            if score < best_score:
                best_score = score
                best_eta = eta
                best_unit = u
                best_debug = {
                    "utype": utype,
                    "call_area": call_area,
                    "score": float(score),
                    "base_score": float(base_score),
                    "coverage_mult": float(coverage_mult),
                    "fairness_mult": float(fairness_mult),
                    "fairness_penalty": float(fairness_penalty),
                    "risk": risk,
                    "keep_free": keep_free,
                }

        if best_unit is None:
            return None, float("inf"), {"policy": self.name, "reason": "no_candidate_after_scoring"}

        debug = {
            "policy": self.name,
            "best_eta_min": float(best_eta),
            "best_score": float(best_score),
            **best_debug,
        }
        return best_unit, float(best_eta), debug
