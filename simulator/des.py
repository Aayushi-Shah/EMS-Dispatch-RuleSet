# scripts/simulator/des.py
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Any

import numpy as np

import config
from simulator import traffic


# -----------------
# Core DES entities
# -----------------


@dataclass(order=True)
class Event:
    t: float
    etype: str
    payload: dict = field(compare=False, default_factory=dict)


@dataclass
class Unit:
    name: str
    utype: str
    station: str
    lon: float
    lat: float
    station_lon: float = 0.0
    station_lat: float = 0.0
    busy_until: float = 0.0
    on_call_id: str | None = None
    can_dispatch: bool = True
    zone: str | None = None       # "ALS" / "BLS" / "OVERLAP"
    unit_area: str | None = None  # "urban" / "rural" / "unknown"
    # NOTE: we deliberately don't declare municipality fields here;
    # they can be attached dynamically in load_units() and accessed via getattr.


class DES:
    """
    Minimal DES kernel for EMS dispatch.

    Responsibilities:
      - Maintain event queue and unit state.
      - For each call event, use `select_unit_fn` to choose units.
      - Model on-scene, transport, turnaround, return-to-base times.
      - Log a compact per-decision record for downstream KPIs.
    """

    def __init__(self, select_unit_fn):
        # Policy hook: (units: list[Unit], now_min: float, call: dict)
        # -> (unit: Unit | None, resp_minutes: float, debug: dict | Any)
        self.select_unit_fn = select_unit_fn

        self.t: float = 0.0
        self.Q: list[Event] = []
        self.units: list[Unit] = []

        self.metrics: dict[str, Any] = {
            "n_calls": 0,
            "missed_calls": 0,
            "unit_busy_min": {},   # unit_name -> total busy minutes
            "decisions": [],       # list[dict] per dispatched unit
        }

        # Configurable list of critical municipalities (standardized, upper-case).
        # kpi_analysis.py uses these to compute critical-zone coverage KPIs.
        default_critical = [
            "EAST LAMPETER TOWNSHIP",
            "LANCASTER",
            "LANCASTER TOWNSHIP",
            "MANHEIM TOWNSHIP",
            "WEST LAMPETER TOWNSHIP",
        ]
        raw_critical = getattr(config, "CRITICAL_MUNICIPALITIES_STD", default_critical)
        self.critical_munis_std: set[str] = {
            m.strip().upper() for m in raw_critical if isinstance(m, str)
        }

    # -----------
    # Event queue
    # -----------

    def schedule(self, t: float, etype: str, **payload):
        heapq.heappush(self.Q, Event(t, etype, payload))

    def add_unit(self, u: Unit):
        self.units.append(u)
        self.metrics["unit_busy_min"][u.name] = 0.0

    def advance(self) -> bool:
        """
        Pop and handle the next event; return False when no events remain.
        """
        if not self.Q:
            return False
        ev = heapq.heappop(self.Q)
        self.t = ev.t
        handler = getattr(self, f"on_{ev.etype}")
        handler(ev)
        return True

    # -----------
    # Event types
    # -----------

    def on_unit_free(self, ev: Event):
        u: Unit = ev.payload["unit"]
        u.lon = ev.payload.get("end_lon", u.lon)
        u.lat = ev.payload.get("end_lat", u.lat)
        u.on_call_id = None
        u.can_dispatch = True

    # ------------
    # Policy hook
    # ------------

    def _pick_unit(self, call: dict):
        """
        Policy hook: returns (unit, resp_minutes, debug_dict_or_any).
        Policy is expected to handle:
          - Availability (busy_until, can_dispatch)
          - ALS/BLS + zone feasibility
        """
        return self.select_unit_fn(self.units, self.t, call)

    # -----------------------
    # Travel / timing helpers
    # -----------------------

    def _get_call_area(self, call: dict) -> str:
        """
        Normalize call area to 'urban', 'rural', or 'unknown'.
        Uses either call_area or urban_rural if present.
        """
        area = (call.get("call_area")
                or call.get("urban_rural")
                or "").lower()
        if area in ("urban", "rural"):
            return area
        return "unknown"

    def _scene_speed_mph(self, call: dict) -> float:
        area = self._get_call_area(call)
        if area == "urban":
            return getattr(config, "SCENE_SPEED_MPH_URBAN", config.SCENE_SPEED_MPH)
        if area == "rural":
            return getattr(config, "SCENE_SPEED_MPH_RURAL", config.SCENE_SPEED_MPH)
        return getattr(config, "SCENE_SPEED_MPH_DEFAULT", config.SCENE_SPEED_MPH)

    def _hospital_speed_mph(self, call: dict) -> float:
        area = self._get_call_area(call)
        if area == "urban":
            return getattr(
                config, "HOSPITAL_SPEED_MPH_URBAN", config.HOSPITAL_SPEED_MPH
            )
        if area == "rural":
            return getattr(
                config, "HOSPITAL_SPEED_MPH_RURAL", config.HOSPITAL_SPEED_MPH
            )
        return getattr(
            config, "HOSPITAL_SPEED_MPH_DEFAULT", config.HOSPITAL_SPEED_MPH
        )

    def _travel_minutes(
        self,
        start_lon: float,
        start_lat: float,
        end_lon: float,
        end_lat: float,
        speed_mph: float,
        abs_epoch_now: float,
    ) -> float:
        return traffic.travel_minutes(
            start_lon,
            start_lat,
            end_lon,
            end_lat,
            speed_mph,
            abs_epoch_now,
        )

    def _sample_onscene_minutes(self, call: dict) -> float:
        """
        Sample on-scene time. You can refine by severity later if you want.
        """
        base = float(getattr(config, "ONSCENE_MIN", 10.0))
        scale = float(getattr(config, "ONSCENE_SCALE", 10.0))
        return base + np.random.gamma(2.0, scale / 2.0)

    def _sample_turnaround_minutes(self, call: dict, to_hosp_minutes: float) -> float:
        """
        Sample hospital offload + turnaround time with a heavy tail for rural
        and/or long transports (LG overload / Southern county).
        """
        base_min = float(getattr(config, "TURNAROUND_MIN", 10.0))
        scale = float(getattr(config, "TURNAROUND_SCALE", 10.0))
        t = base_min + np.random.gamma(2.0, scale / 2.0)

        area = self._get_call_area(call)
        is_rural = area == "rural"
        long_thresh = float(getattr(config, "LONG_TRANSPORT_MIN_MINUTES", 20.0))
        long_prob = float(getattr(config, "LONG_TURNAROUND_PROB", 0.3))
        extra_min = float(getattr(config, "LONG_TURNAROUND_EXTRA_MIN", 30.0))
        extra_max = float(getattr(config, "LONG_TURNAROUND_EXTRA_MAX", 90.0))

        if (is_rural or to_hosp_minutes >= long_thresh) and np.random.random() < long_prob:
            t += np.random.uniform(extra_min, extra_max)

        return t

    # --------------
    # Coverage snapshot
    # --------------

    def _snapshot_idle_counts(self, call: dict) -> dict:
        """
        Snapshot how many ALS/BLS units are idle right now, globally and
        specifically in the call's zone, plus critical-municipality coverage.

        Returns keys:
          - idle_als_total_before
          - idle_bls_total_before
          - idle_als_in_call_zone_before
          - idle_bls_in_call_zone_before
          - call_municipality_std
          - call_in_critical_muni
          - idle_units_in_critical_muni_before
        """
        call_zone = (call.get("zone") or "").upper()

        idle_als_total = 0
        idle_bls_total = 0
        idle_als_in_call_zone = 0
        idle_bls_in_call_zone = 0

        # --- critical municipality context for this call ---
        raw_call_muni = (
            call.get("call_municipality_std")
            or call.get("municipality_std")
            or call.get("muni_std")
        )
        if isinstance(raw_call_muni, str):
            call_muni_std = raw_call_muni.strip().upper()
        else:
            call_muni_std = None

        call_in_critical = (
            call_muni_std is not None and call_muni_std in self.critical_munis_std
        )
        idle_units_in_critical_muni = 0

        for u in self.units:
            if not (u.can_dispatch and u.busy_until <= self.t):
                continue

            cap = (u.utype or "").upper()
            uz_raw = getattr(u, "zone", None)
            uz = (uz_raw or "").upper()
            if not uz or uz == "UNKNOWN":
                uz = "OVERLAP"

            # Global tallies
            if cap == "ALS":
                idle_als_total += 1
            elif cap == "BLS":
                idle_bls_total += 1

            # Zone-aware coverage
            if call_zone == "ALS":
                if cap == "ALS":
                    idle_als_in_call_zone += 1

            elif call_zone == "BLS":
                if cap == "BLS" and uz in ("BLS", "OVERLAP"):
                    idle_bls_in_call_zone += 1

            elif call_zone == "OVERLAP" or call_zone == "":
                if cap == "ALS":
                    idle_als_in_call_zone += 1
                elif cap == "BLS":
                    idle_bls_in_call_zone += 1

            # Critical-municipality idle units:
            # require unit to have a standardized station municipality field.
            if call_in_critical:
                u_muni_raw = (
                    getattr(u, "station_municipality_std", None)
                    or getattr(u, "unit_municipality_std", None)
                )
                if isinstance(u_muni_raw, str):
                    u_muni_std = u_muni_raw.strip().upper()
                    if u_muni_std == call_muni_std:
                        idle_units_in_critical_muni += 1

        return {
            "idle_als_total_before": idle_als_total,
            "idle_bls_total_before": idle_bls_total,
            "idle_als_in_call_zone_before": idle_als_in_call_zone,
            "idle_bls_in_call_zone_before": idle_bls_in_call_zone,
            "call_municipality_std": call_muni_std,
            "call_in_critical_muni": bool(call_in_critical),
            "idle_units_in_critical_muni_before": (
                idle_units_in_critical_muni if call_in_critical else 0
            ),
        }

    def _count_idle_units_in_municipality(self, muni_std: str | None) -> int:
        """
        Count how many dispatch-ready units currently have the given
        standardized municipality tag.
        """
        if not muni_std:
            return 0
        target = str(muni_std).strip().upper()
        if not target:
            return 0

        count = 0
        for u in self.units:
            if not (u.can_dispatch and u.busy_until <= self.t):
                continue
            u_muni_raw = (
                getattr(u, "municipality_std", None)
                or getattr(u, "station_municipality_std", None)
                or getattr(u, "unit_municipality_std", None)
            )
            if isinstance(u_muni_raw, str) and u_muni_raw.strip().upper() == target:
                count += 1
        return count

    # --------------
    # Transport decision
    # --------------

    def _decide_transport(self, call: dict) -> bool:
        """
        Decide if the call results in a transport to hospital.

        True  -> there is at least one transporting unit for this call.
        False -> non-transport.
        """
        # Global switch: disable non-transport logic if requested
        if not getattr(config, "USE_NON_TRANSPORT", True):
            return True

        # Severity from tagged parquet
        sev = (call.get("severity_bucket")
               or call.get("call_severity_bucket")
               or "unknown").lower()

        # Severity-specific baseline
        by_sev = getattr(config, "NON_TRANSPORT_BY_SEVERITY", None)
        if isinstance(by_sev, dict):
            default_base = float(getattr(config, "NON_TRANSPORT_BASE", 0.1))
            prob_non_tx = float(by_sev.get(sev, by_sev.get("unknown", default_base)))
        else:
            prob_non_tx = float(getattr(config, "NON_TRANSPORT_BASE", 0.1))

        # Keyword overrides based on description/incidentType
        text = f"{call.get('description', '')} {call.get('incidentType', '')}".lower()

        high_prob = float(getattr(config, "NON_TRANSPORT_HIGH_PROB", 0.8))
        low_prob = float(getattr(config, "NON_TRANSPORT_LOW_PROB", 0.02))

        nt_keywords = [k.lower() for k in getattr(config, "NON_TRANSPORT_KEYWORDS", [])]
        if nt_keywords and any(k in text for k in nt_keywords):
            prob_non_tx = max(prob_non_tx, high_prob)

        t_keywords = [k.lower() for k in getattr(config, "TRANSPORT_KEYWORDS", [])]
        if t_keywords and any(k in text for k in t_keywords):
            prob_non_tx = min(prob_non_tx, low_prob)

        prob_non_tx = max(0.0, min(1.0, prob_non_tx))
        r = np.random.random()
        non_transport = r < prob_non_tx

        return not non_transport

    # --------------
    # Call handling
    # --------------

    def on_call(self, ev: Event):
        call = ev.payload

        # Track when this CAD call first hit the queue (for queue delay)
        if "_first_t" not in call:
            call["_first_t"] = self.t

        # Count distinct CAD calls once (even if retried)
        if not call.get("_counted", False):
            self.metrics["n_calls"] += 1
            call["_counted"] = True

        units_needed = int(call.get("units_needed", 1) or 1)
        assigned = call.get("_assigned", 0)
        remaining = max(units_needed - assigned, 0)

        # Decide once per CAD call whether there will be transport at all
        transport_flag = call.get("_transport_flag")
        if transport_flag is None:
            transport_flag = self._decide_transport(call)
            call["_transport_flag"] = transport_flag

        transport_unit_name = call.get("_transport_unit")

        # Simple risk flags based on severity bucket; configurable via HIGH_SEVERITY_BUCKETS
        sev = (call.get("severity_bucket") or "unknown").lower()
        high_buckets = [
            s.lower()
            for s in getattr(
                config,
                "HIGH_SEVERITY_BUCKETS",
                ["high", "critical", "p1", "priority1"],
            )
        ]
        call_is_high = sev in high_buckets
        call_is_low = not call_is_high

        # Try to assign as many units as needed at this event time
        local_assigned = 0
        while remaining > 0:
            # Coverage snapshot BEFORE this assignment
            cov = self._snapshot_idle_counts(call)
            idle_units_in_call_muni_before = self._count_idle_units_in_municipality(
                cov.get("call_municipality_std")
            )

            # Queue delay from first arrival of this CAD call to this assignment attempt
            first_t = float(call.get("_first_t", self.t))
            queue_delay = max(self.t - first_t, 0.0)

            # Effective current epoch for travel-time model
            call_abs = float(call.get("_abs_epoch", 0.0))
            call_tmin = float(call.get("tmin", 0.0))
            abs_epoch_now = call_abs + (self.t - call_tmin) * 60.0

            # Ask policy for a unit
            u, resp_minutes, debug = self._pick_unit(call)

            # If no unit is available/feasible, log a failure decision and stop trying
            if u is None:
                reason = None
                if isinstance(debug, dict):
                    reason = debug.get("reason")
                decision_row = {
                    "tmin": float(self.t),
                    "call_id": call["id"],
                    "call_zone": call.get("zone"),
                    "call_area": call.get("call_area"),
                    "call_urban_rural": call.get("urban_rural"),
                    "call_risk_score": call.get("risk_score"),
                    "call_severity_bucket": call.get("severity_bucket"),
                    "call_preferred_unit_type": call.get("preferred_unit_type"),
                    "call_units_needed": units_needed,
                    # No unit assigned
                    "unit": None,
                    "unit_type": None,
                    "unit_zone": None,
                    "unit_area": None,
                    "unit_station": None,
                    "resp_min": float("inf"),
                    "queue_delay_min": float(queue_delay),
                    "unit_will_transport": False,
                    "call_transport_flag": bool(transport_flag),
                    "busy_total_min": 0.0,
                    # Coverage snapshot
                    **cov,
                    "idle_units_in_call_muni_before": idle_units_in_call_muni_before,
                    # Risk flags
                    "call_is_high": bool(call_is_high),
                    "call_is_low": bool(call_is_low),
                    # Failure info
                    "failure_reason": reason or "no_unit_available_or_feasible",
                }
                if isinstance(debug, dict):
                    decision_row.update(
                        {f"policy_{k}": v for k, v in debug.items()}
                    )
                self.metrics["decisions"].append(decision_row)
                break

            # Successful assignment
            resp_minutes = float(resp_minutes)

            # Decide if THIS unit transports (only one when SINGLE_TRANSPORT_PER_CALL is True)
            unit_will_transport = False
            if transport_flag:
                if transport_unit_name is None:
                    unit_will_transport = True
                    transport_unit_name = u.name
                    call["_transport_unit"] = transport_unit_name
                elif not getattr(config, "SINGLE_TRANSPORT_PER_CALL", True):
                    unit_will_transport = True

            # On-scene time
            onscene = self._sample_onscene_minutes(call)

            # Scene -> hospital
            to_hosp = 0.0
            turn = 0.0
            if unit_will_transport and call.get("h_lon") is not None:
                hosp_speed = self._hospital_speed_mph(call)
                to_hosp = self._travel_minutes(
                    call["lon"],
                    call["lat"],
                    call["h_lon"],
                    call["h_lat"],
                    hosp_speed,
                    abs_epoch_now,
                )
                if to_hosp > 0:
                    turn = self._sample_turnaround_minutes(call, to_hosp)

            # Return to base (from scene or hospital)
            end_lon, end_lat = u.station_lon, u.station_lat
            start_lon = (
                call["h_lon"] if (unit_will_transport and to_hosp > 0) else call["lon"]
            )
            start_lat = (
                call["h_lat"] if (unit_will_transport and to_hosp > 0) else call["lat"]
            )
            if unit_will_transport and to_hosp > 0:
                ret_speed = self._hospital_speed_mph(call)
            else:
                ret_speed = self._scene_speed_mph(call)

            return_time = self._travel_minutes(
                start_lon,
                start_lat,
                end_lon,
                end_lat,
                ret_speed,
                abs_epoch_now,
            )

            total_busy = resp_minutes + onscene + to_hosp + turn + return_time

            # Update unit state
            u.busy_until = self.t + total_busy
            u.on_call_id = call["id"]
            u.can_dispatch = False

            # Aggregate unit busy time
            self.metrics["unit_busy_min"][u.name] += total_busy

            # Schedule when this unit becomes free again
            self.schedule(
                u.busy_until,
                "unit_free",
                unit=u,
                end_lon=end_lon,
                end_lat=end_lat,
            )

            # Compact per-decision log row (for KPIs)
            decision_row = {
                "tmin": float(self.t),
                "call_id": call["id"],
                "call_zone": call.get("zone"),
                "call_area": call.get("call_area"),
                "call_urban_rural": call.get("urban_rural"),
                "call_risk_score": call.get("risk_score"),
                "call_severity_bucket": call.get("severity_bucket"),
                "call_preferred_unit_type": call.get("preferred_unit_type"),
                "call_units_needed": units_needed,
                "unit": u.name,
                "unit_type": u.utype,
                "unit_zone": getattr(u, "zone", None),
                "unit_area": getattr(u, "unit_area", None),
                "unit_station": u.station,
                "resp_min": resp_minutes,
                "queue_delay_min": float(queue_delay),
                "unit_will_transport": bool(unit_will_transport),
                "call_transport_flag": bool(transport_flag),
                "busy_total_min": float(total_busy),
                # Coverage snapshot BEFORE dispatch
                **cov,
                "idle_units_in_call_muni_before": idle_units_in_call_muni_before,
                # Risk flags
                "call_is_high": bool(call_is_high),
                "call_is_low": bool(call_is_low),
                # No failure here
                "failure_reason": None,
            }

            # Allow policy to add extra debug fields if needed
            if isinstance(debug, dict):
                decision_row.update(
                    {f"policy_{k}": v for k, v in debug.items()}
                )

            self.metrics["decisions"].append(decision_row)

            local_assigned += 1
            remaining -= 1

        total_assigned = assigned + local_assigned
        if total_assigned < units_needed:
            retry = call.get("_retries", 0)
            if retry < config.MAX_QUEUE_RETRIES:
                call["_retries"] = retry + 1
                call["_assigned"] = total_assigned
                # Simple retry after 1 minute; configurable if needed
                self.schedule(self.t + 1.0, "call", **call)
            else:
                self.metrics["missed_calls"] += (units_needed - total_assigned)
