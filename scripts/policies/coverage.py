# P3: Coverage-Preserving ETA
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from scripts.policies.common import (
    BasePolicy,
    CallDict,
    PolicyResult,
    UnitLike,
    _filter_dispatchable,
    _hav_miles,
    _rule_signals_p3,
    call_urban_rural,
    unit_urban_rural,
    r1_eta_minutes,
    r8_busy_load_penalty,
    r10_random_tiebreaker,
)


class CoveragePreservingETA(BasePolicy):
    """
    P3: Coverage-preserving policy (coverage-focused).

    Intent:
      - Start from R1 ETA (traffic + dispatch delay), same as P1.
      - Penalize draining local coverage with different rules for
        urban vs rural, using TIGER-based urban/rural polygons.
      - Use current unit positions (not stations) for coverage.
      - Ignore busy units when counting coverage.
      - Do NOT re-implement ALS/severity logic (that’s P2’s job).
    """

    name = "p3_coverage"

    # Tunables – can move to config if needed
    URBAN_RADIUS_MI = 1.0       # coverage radius for urban units/calls
    URBAN_MIN_FREE = 3        # min free units required in urban radius

    RURAL_RADIUS_MI = 3.0       # coverage radius for rural units/calls
    RURAL_MIN_FREE = 2         # min free units required in rural radius

    BETA = 2.0                 # strength of coverage penalty
    URBAN_CALL_BOOST = 2.0      # extra protection when the call itself is urban

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        # We only use signals for  keep_free / keep_close hints
        signals = _rule_signals_p3(call)
        keep_free = signals["keep_free"]
        keep_close = signals["keep_close"]

        # 1) Candidate units: can_dispatch and not busy past now_min
        candidates = _filter_dispatchable(units, now_min=now_min)
        if not candidates:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_dispatchable_units",
            }

        # 2) Coverage pool = units that remain available if we don't dispatch them
        #    (free + dispatchable at this moment)
        coverage_pool: List[UnitLike] = [
            u for u in units
            if getattr(u, "can_dispatch", True)
            and float(getattr(u, "busy_until", 0.0) or 0.0) <= now_min + 1e-9
        ]

        # 3) Call geometry + classification
        c_lon, c_lat = float(call["lon"]), float(call["lat"])
        call_area = call_urban_rural(call)
        call_is_urban = (call_area == "urban")

        best_unit: Optional[UnitLike] = None
        best_eta = float("inf")
        best_score = float("inf")
        best_debug_fields: Dict[str, Any] = {}

        total_free = len(coverage_pool)

        for u in candidates:
            u_lon, u_lat = float(u.lon), float(u.lat)

            # 4) Classify unit's current location → pick radius & min_free
            u_area = unit_urban_rural(u)
            if u_area == "urban":
                radius = self.URBAN_RADIUS_MI
                min_free = self.URBAN_MIN_FREE
            elif u_area == "rural":
                radius = self.RURAL_RADIUS_MI
                min_free = self.RURAL_MIN_FREE
            else:
                # Unknown or outside county → conservative rural-ish assumption
                radius = self.RURAL_RADIUS_MI
                min_free = self.RURAL_MIN_FREE

            # 5) Base ETA from R1 (with traffic)
            eta = r1_eta_minutes(u, now_min, call)

            # 6) Coverage remaining if we dispatch this unit
            remaining_free = 0
            for other in coverage_pool:
                if other is u:
                    continue
                o_lon, o_lat = float(other.lon), float(other.lat)
                d = _hav_miles(u_lon, u_lat, o_lon, o_lat)
                if d <= radius:
                    remaining_free += 1

            # Coverage loss = how far below target we drop
            coverage_loss = max(0, (min_free - remaining_free) / min_free)

            # Calls in urban area get extra protection
            if call_is_urban:
                coverage_loss *= self.URBAN_CALL_BOOST

            # Avoid draining the last unit if there is at least one other free unit elsewhere
            if total_free > 1 and remaining_free == 0:
                coverage_loss += 0.5  # nudge away from emptying the area

            # Turn coverage loss into multiplicative penalty
            coverage_mult = 1.0 + self.BETA * coverage_loss

            # 7) Score = ETA + light hints, modulated by coverage
            base_score = eta
            base_score += r8_busy_load_penalty(u, now_min)

            if keep_free:
                base_score += 0.2
            if keep_close:
                base_score += 0.2

            score = base_score * coverage_mult

            # Tiny jitter so ties don't always pick same unit
            score += r10_random_tiebreaker(self._rng)

            debug_fields_for_this_unit = {
                "unit_id": getattr(u, "unit_id", None),
                "u_area": u_area,
                "call_area": call_area,
                "radius": radius,
                "min_free": min_free,
                "remaining_free": remaining_free,
                "coverage_loss": coverage_loss,
                "coverage_mult": coverage_mult,
            }

            if score < best_score:
                best_score = score
                best_eta = eta
                best_unit = u
                best_debug_fields = debug_fields_for_this_unit

        if best_unit is None:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_candidate_after_scoring",
            }

        debug = {
            "policy": self.name,
            "best_eta_min": float(best_eta),
            "best_score": float(best_score),
            "urban_radius_mi": self.URBAN_RADIUS_MI,
            "urban_min_free": self.URBAN_MIN_FREE,
            "rural_radius_mi": self.RURAL_RADIUS_MI,
            "rural_min_free": self.RURAL_MIN_FREE,
            "beta": self.BETA,
            "urban_call_boost": self.URBAN_CALL_BOOST,
            "keep_free_flag": keep_free,
            "keep_close_flag": keep_close,
            **best_debug_fields,
        }
        return best_unit, float(best_eta), debug
