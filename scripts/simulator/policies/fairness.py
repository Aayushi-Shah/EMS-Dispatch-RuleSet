# P4: Fairness-First Dispatch
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from scripts.simulator.policies.common import (
    BasePolicy,
    CallDict,
    PolicyResult,
    UnitLike,
    _filter_dispatchable,
    _hav_miles,
    _rule_signals_p4,
    classify_urban_rural_cached,
    r1_eta_minutes,
    r8_busy_load_penalty,
    r10_random_tiebreaker,
)


class FairnessFirstPolicy(BasePolicy):
    """
    P4: Fairness-first policy.

    Intent:
      - Prioritize fairness / protection of rural capacity over raw ETA.
      - Do NOT re-implement ALS/severity (that's P2's job).
      - Use urban/rural classification + fairness_weight (R5) to:
          * Avoid draining rural units for urban calls when avoidable.
          * Prefer rural units for rural calls when they exist.
      - ETA is ONLY a tie-breaker among equally fair options.
    """

    name = "p4_fairness"
    RURAL_SAME_REGION_RADIUS_MI = 10.0
    URBAN_SAME_REGION_RADIUS_MI = 5.0

    def __call__(self, units: List[UnitLike], now_min: float, call: CallDict) -> PolicyResult:
        # Call-level fairness signals (R5/R7/R8 subset)
        signals = _rule_signals_p4(call)
        fairness_w = signals["fairness_w"]
        keep_free = signals["keep_free"]
        keep_close = signals["keep_close"]

        # Candidate units: can_dispatch and free by now_min
        candidates = _filter_dispatchable(units, now_min=now_min)
        if not candidates:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_dispatchable_units",
            }

        # Call location + area classification
        c_lon, c_lat = float(call["lon"]), float(call["lat"])
        call_area = classify_urban_rural_cached(c_lon, c_lat)

        def fairness_penalty_for_unit(u: UnitLike) -> float:
            """
            Lower = better (more fair).
            We treat draining rural capacity for urban calls as highly unfair.
            For rural calls, we prefer rural units when possible.
            Penalty is scaled by fairness_w so rural/underserved calls matter more.
            """
            u_lon, u_lat = float(u.lon), float(u.lat)
            u_area = classify_urban_rural_cached(u_lon, u_lat)
            d_mi = _hav_miles(c_lon, c_lat, u_lon, u_lat)

            # Base penalties by (call_area, unit_area)
            if call_area == "urban":
                # Urban call: avoid draining rural units if urban units exist
                if u_area == "urban":
                    # Same-region urban preferred; distant urban treated as mismatch
                    if d_mi <= self.URBAN_SAME_REGION_RADIUS_MI:
                        base = 0.15
                    else:
                        base = 2.5
                elif u_area == "unknown":
                    base = 0.7
                else:  # u_area == "rural"
                    base = 2.5
            elif call_area == "rural":
                # Rural call: prefer rural units, but urban support is OK if needed
                if u_area == "rural":
                    # Same-region rural preferred; distant rural treated as mismatch
                    if d_mi <= self.RURAL_SAME_REGION_RADIUS_MI:
                        base = 0.1
                    else:
                        base = 2.5
                elif u_area == "unknown":
                    base = 0.6
                else:  # u_area == "urban"
                    base = 1.4
            else:
                # Unknown call area → mild, symmetric penalties
                base = 0.4

            # Scale by fairness weight (R5) so underserved calls get stronger fairness
            return base * max(fairness_w, 1.0)

        best_unit: Optional[UnitLike] = None
        best_fair = float("inf")
        best_eta = float("inf")
        best_debug_fields: Dict[str, Any] = {}

        for u in candidates:
            # 1) Fairness penalty (PRIMARY objective)
            f_pen = fairness_penalty_for_unit(u)

            # 2) ETA + light hints (SECONDARY tie-breaker)
            eta = r1_eta_minutes(u, now_min, call)
            eta_component = eta + r8_busy_load_penalty(u, now_min)

            if keep_free:
                eta_component += 0.2
            if keep_close:
                eta_component += 0.2

            # Small jitter only on ETA component (tie-breaking among equals)
            eta_component += r10_random_tiebreaker(self._rng)

            # Lexicographic comparison: fairness first, then ETA
            if f_pen < best_fair - 1e-6:
                best_fair = f_pen
                best_eta = eta
                best_unit = u
                best_debug_fields = {
                    "unit_id": getattr(u, "unit_id", None),
                    "fairness_penalty": f_pen,
                    "eta_component": float(eta_component),
                    "eta_min": float(eta),
                    "u_area": classify_urban_rural_cached(float(u.lon), float(u.lat)),
                    "call_area": call_area,
                }
            elif abs(f_pen - best_fair) <= 1e-6 and eta < best_eta:
                # Same fairness bucket → prefer faster ETA
                best_fair = f_pen
                best_eta = eta
                best_unit = u
                best_debug_fields = {
                    "unit_id": getattr(u, "unit_id", None),
                    "fairness_penalty": f_pen,
                    "eta_component": float(eta_component),
                    "eta_min": float(eta),
                    "u_area": classify_urban_rural_cached(float(u.lon), float(u.lat)),
                    "call_area": call_area,
                }

        if best_unit is None:
            return None, float("inf"), {
                "policy": self.name,
                "reason": "no_candidate_after_scoring",
            }

        debug = {
            "policy": self.name,
            "best_eta_min": float(best_eta),
            "best_fairness_penalty": float(best_fair),
            "fairness_weight": fairness_w,
            "keep_free_flag": keep_free,
            "keep_close_flag": keep_close,
            **best_debug_fields,
        }
        return best_unit, float(best_eta), debug
