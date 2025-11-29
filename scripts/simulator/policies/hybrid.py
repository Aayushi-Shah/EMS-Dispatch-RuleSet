# P5: All-in-One – Severity + Coverage + Fairness
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from scripts.simulator.policies.common import (
    BasePolicy,
    CallDict,
    PolicyResult,
    UnitLike,
    _filter_bls_by_capability,
    _filter_dispatchable,
    _hav_miles,
    _norm_utype_cached,
    _rule_signals_p5,
    _severity_multiplier,
    classify_urban_rural,
    r1_eta_minutes,
    r8_busy_load_penalty,
    r10_random_tiebreaker,
)


class HybridSeverityCoverageFairness(BasePolicy):
    """
    P5: Hybrid policy that combines:
      - P2-style ALS/BLS + severity handling.
      - P3-style coverage preservation (unit area + call area).
      - P4-style fairness bias (urban vs rural, underserved zones).
    """

    name = "p5_hybrid"

    # --- Severity (ALS/BLS) ---
    HIGH_RISK_THRESH = 0.75
    LOW_RISK_THRESH = 0.35
    PENALTY_BLS_FOR_HIGH = 1.6   # high-risk call but BLS
    PENALTY_ALS_FOR_LOW = 1.3    # low-risk call but ALS
    HIGH_RISK_OVERRIDE = 0.85    # allow draining last unit for very high risk

    # --- Coverage (unit area + call area) ---
    URBAN_RADIUS_MI = 1.0
    URBAN_MIN_FREE = 3

    RURAL_RADIUS_MI = 3.0
    RURAL_MIN_FREE = 2

    BETA_COVERAGE = 2.0          # importance of coverage in score
    URBAN_CALL_BOOST = 1.5       # extra protection when call is urban

    # --- Fairness (area + rule-based weight) ---
    GAMMA_FAIRNESS = 1.5         # strength of fairness penalty
    AREA_MISMATCH_PENALTY = 0.5  # base penalty when unit area != call area
    RURAL_SAME_REGION_RADIUS_MI = 10.0
    URBAN_SAME_REGION_RADIUS_MI = 5.0

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        # 1) Pull rich rule signals (capability, risk, fairness, demand, etc.)
        signals = _rule_signals_p5(call)
        als_capable = signals["als_capable"]
        bls_capable = signals["bls_capable"]
        risk = signals["risk"]
        fairness_w = signals["fairness_w"]
        keep_free = signals["keep_free"]
        keep_close = signals["keep_close"]
        zone_underprotected = signals["zone_underprotected"]
        zone_demand_score = signals["zone_demand_score"]
        require_als = signals["require_als"] or (risk >= self.HIGH_RISK_THRESH and als_capable)
        has_zone = signals["has_zone"]

        # 2) Candidate units: dispatchable AND not busy past now_min
        candidates = _filter_dispatchable(units, now_min=now_min)
        if not candidates:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_dispatchable_units",
            }

        # Respect BLS geographic capability at the call level
        candidates = _filter_bls_by_capability(candidates, bls_capable)
        if not candidates:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_dispatchable_after_bls_filter",
            }

        # 3) Coverage pool = units that remain available if we don't dispatch them
        coverage_pool: List[UnitLike] = [
            u for u in units
            if getattr(u, "can_dispatch", True)
            and float(getattr(u, "busy_until", 0.0) or 0.0) <= now_min + 1e-9
        ]

        # 4) Call geometry + area classification
        c_lon, c_lat = float(call["lon"]), float(call["lat"])
        call_area = classify_urban_rural(c_lon, c_lat)
        call_is_urban = (call_area == "urban")

        best_unit: Optional[UnitLike] = None
        best_eta = float("inf")
        best_score = float("inf")
        best_debug: Dict[str, Any] = {}

        for u in candidates:
            utype = _norm_utype_cached(u)

            # If rules/severity require ALS and ALS is capable, skip BLS
            if require_als and als_capable and utype == "BLS":
                continue

            # Never send BLS where BLS is not geographically capable
            if utype == "BLS" and not bls_capable:
                continue

            u_lon, u_lat = float(u.lon), float(u.lat)
            u_area = classify_urban_rural(u_lon, u_lat)
            d_call_unit = _hav_miles(c_lon, c_lat, u_lon, u_lat)

            # --- 5) Choose coverage radius/min_free based on unit area ---
            if u_area == "urban":
                unit_radius = self.URBAN_RADIUS_MI
                unit_min_free = self.URBAN_MIN_FREE
            elif u_area == "rural":
                unit_radius = self.RURAL_RADIUS_MI
                unit_min_free = self.RURAL_MIN_FREE
            else:
                # Unknown → conservative rural-ish assumption
                unit_radius = self.RURAL_RADIUS_MI
                unit_min_free = self.RURAL_MIN_FREE

            # Call-area coverage settings
            if call_area == "urban":
                call_radius = self.URBAN_RADIUS_MI
                call_min_free = self.URBAN_MIN_FREE
            elif call_area == "rural":
                call_radius = self.RURAL_RADIUS_MI
                call_min_free = self.RURAL_MIN_FREE
            else:
                call_radius = self.RURAL_RADIUS_MI
                call_min_free = self.RURAL_MIN_FREE

            # --- 6) Base ETA (R1) + severity multiplier (P2-style) ---
            eta = r1_eta_minutes(u, now_min, call)

            sev_mult = _severity_multiplier(
                utype,
                risk,
                als_capable,
                bls_capable,
                self.HIGH_RISK_THRESH,
                self.LOW_RISK_THRESH,
                self.PENALTY_BLS_FOR_HIGH,
                self.PENALTY_ALS_FOR_LOW,
            )

            # --- 7) Coverage around unit area and call area (P3-style) ---
            remaining_around_unit = 0
            remaining_around_call = 0
            for other in coverage_pool:
                if other is u:
                    continue
                o_lon, o_lat = float(other.lon), float(other.lat)
                d_unit = _hav_miles(u_lon, u_lat, o_lon, o_lat)
                d_call = _hav_miles(c_lon, c_lat, o_lon, o_lat)
                if d_unit <= unit_radius:
                    remaining_around_unit += 1
                if d_call <= call_radius:
                    remaining_around_call += 1

            # fractional coverage loss (0 = fine, 1 = totally empty)
            unit_loss = 0.0
            if unit_min_free > 0:
                unit_loss = max(0.0, (unit_min_free - remaining_around_unit) / unit_min_free)

            call_loss = 0.0
            if call_min_free > 0:
                call_loss = max(0.0, (call_min_free - remaining_around_call) / call_min_free)

            # extra weight for underprotected/high-demand zones
            coverage_loss = 0.4 * unit_loss + 0.6 * call_loss
            if call_is_urban:
                coverage_loss *= self.URBAN_CALL_BOOST
            if zone_underprotected:
                coverage_loss *= 1.5
            if zone_demand_score > 0.0:
                coverage_loss *= (1.0 + min(zone_demand_score, 2.0) * 0.25)

            # Soften last-unit guardrail
            if remaining_around_unit == 0 and risk < self.HIGH_RISK_OVERRIDE:
                coverage_loss += 1.0

            coverage_mult = 1.0 + self.BETA_COVERAGE * max(0.0, coverage_loss)

            # --- 8) Fairness penalty (P4-style bias) ---
            fairness_penalty = 0.0

            # 8a) Area mismatch between unit and call
            if u_area != "unknown" and call_area != "unknown" and u_area != call_area:
                fairness_penalty += self.AREA_MISMATCH_PENALTY

            # 8a-2) Same-area but distant counts as mismatch
            if call_area == "rural" and u_area == "rural" and d_call_unit > self.RURAL_SAME_REGION_RADIUS_MI:
                fairness_penalty += self.AREA_MISMATCH_PENALTY
            if call_area == "urban" and u_area == "urban" and d_call_unit > self.URBAN_SAME_REGION_RADIUS_MI:
                fairness_penalty += self.AREA_MISMATCH_PENALTY

            # 8b) Optional: add small penalty when call lacks zone info to avoid bias
            if not has_zone:
                fairness_penalty *= 0.8

            fairness_mult = 1.0 + self.GAMMA_FAIRNESS * fairness_penalty

            # --- 9) Base score: ETA + light hints ---
            base_score = eta * sev_mult
            base_score += r8_busy_load_penalty(u, now_min)

            if keep_free:
                base_score += 0.2
            if keep_close:
                base_score += 0.2

            # Combine: coverage + fairness
            score = base_score * coverage_mult * fairness_mult

            # Policy-level fairness weight (from call-side R5) – favor underserved calls
            score /= max(fairness_w, 1.0)

            # jitter for stability
            score += r10_random_tiebreaker(self._rng)

            debug_for_u = {
                "unit_id": getattr(u, "unit_id", None),
                "utype": utype,
                "u_area": u_area,
                "call_area": call_area,
                "eta_min": float(eta),
                "sev_mult": float(sev_mult),
                "unit_radius": unit_radius,
                "unit_min_free": unit_min_free,
                "remaining_around_unit": remaining_around_unit,
                "call_radius": call_radius,
                "call_min_free": call_min_free,
                "remaining_around_call": remaining_around_call,
                "coverage_loss": float(coverage_loss),
                "coverage_mult": float(coverage_mult),
                "fairness_penalty": float(fairness_penalty),
                "fairness_mult": float(fairness_mult),
                "fairness_weight": float(fairness_w),
                "keep_free_flag": keep_free,
                "keep_close_flag": keep_close,
                "zone_underprotected": zone_underprotected,
                "zone_demand_score": zone_demand_score,
                "score": float(score),
            }

            if score < best_score:
                best_score = score
                best_eta = eta
                best_unit = u
                best_debug = debug_for_u

        if best_unit is None:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_candidate_after_scoring",
            }

        debug = {
            "policy": self.name,
            "best_eta_min": float(best_eta),
            "best_score": float(best_score),
            **best_debug,
        }
        return best_unit, float(best_eta), debug
