# scripts/simulator/des.py
from __future__ import annotations
from dataclasses import dataclass, field
import heapq
import numpy as np

from scripts.simulator import config, traffic


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
    zone: str | None = None  # ALS/BLS/OVERLAP; unit_area is attached dynamically


class DES:
    def __init__(self, select_unit_fn):
        self.select_unit_fn = select_unit_fn
        self.t = 0.0
        self.Q: list[Event] = []
        self.units: list[Unit] = []
        self.metrics = {
            "n_calls": 0,
            "resp_times": [],
            "on_scene": [],
            "transport": [],
            "turnaround": [],
            "unit_util": {},
            "wait_minutes": [],
            "missed_calls": 0,
            "decisions": [],        # per-assigned-unit debug log
            "turnaround_hour": {},  # hour -> list[mins]
        }

    def schedule(self, t, etype, **payload):
        heapq.heappush(self.Q, Event(t, etype, payload))

    def add_unit(self, u: Unit):
        self.units.append(u)
        self.metrics["unit_util"][u.name] = 0.0

    def advance(self):
        if not self.Q:
            return False
        ev = heapq.heappop(self.Q)
        self.t = ev.t
        getattr(self, f"on_{ev.etype}")(ev)
        return True

    def on_unit_free(self, ev: Event):
        u: Unit = ev.payload["unit"]
        u.lon = ev.payload.get("end_lon", u.lon)
        u.lat = ev.payload.get("end_lat", u.lat)
        u.on_call_id = None
        u.can_dispatch = True

    def _pick_unit(self, call: dict):
        """
        Policy hook: returns (unit, resp_minutes, debug_dict_or_any)
        """
        return self.select_unit_fn(self.units, self.t, call)

    def _decide_transport(self, call: dict) -> bool:
        """
        Decide if the call results in a transport to hospital.

        Logic:
        1) If USE_NON_TRANSPORT is False -> always transport.
        2) Start from severity-bucket-based non-transport probability
           (NON_TRANSPORT_BY_SEVERITY), falling back to NON_TRANSPORT_BASE.
        3) Use NON_TRANSPORT_KEYWORDS / TRANSPORT_KEYWORDS as overrides.
        4) Sample a Bernoulli; return True if this call is transported.
        """
        # Global switch: disable non-transport logic if requested
        if not getattr(config, "USE_NON_TRANSPORT", True):
            call["_non_transport_prob"] = 0.0
            call["_non_transport_draw"] = 1.0
            call["_severity_bucket"] = (call.get("severity_bucket") or "unknown").lower()
            return True  # always transport

        # Severity from tagged parquet
        sev = (call.get("severity_bucket")
               or call.get("call_severity_bucket")
               or "unknown").lower()

        # 2) Severity-specific baseline
        by_sev = getattr(config, "NON_TRANSPORT_BY_SEVERITY", None)
        if isinstance(by_sev, dict):
            default_base = float(getattr(config, "NON_TRANSPORT_BASE", 0.1))
            prob_non_tx = float(by_sev.get(sev, by_sev.get("unknown", default_base)))
        else:
            prob_non_tx = float(getattr(config, "NON_TRANSPORT_BASE", 0.1))

        # 3) Keyword overrides based on description/incidentType
        text = f"{call.get('description', '')} {call.get('incidentType', '')}".lower()

        high_prob = float(getattr(config, "NON_TRANSPORT_HIGH_PROB", 0.8))
        low_prob = float(getattr(config, "NON_TRANSPORT_LOW_PROB", 0.02))

        nt_keywords = [k.lower() for k in getattr(config, "NON_TRANSPORT_KEYWORDS", [])]
        if nt_keywords and any(k in text for k in nt_keywords):
            prob_non_tx = max(prob_non_tx, high_prob)

        t_keywords = [k.lower() for k in getattr(config, "TRANSPORT_KEYWORDS", [])]
        if t_keywords and any(k in text for k in t_keywords):
            prob_non_tx = min(prob_non_tx, low_prob)

        # Clamp and sample
        prob_non_tx = max(0.0, min(1.0, prob_non_tx))
        r = np.random.random()
        non_transport = r < prob_non_tx

        # For debugging / KPIs
        call["_non_transport_prob"] = prob_non_tx
        call["_non_transport_draw"] = r
        call["_severity_bucket"] = sev

        # True = transport, False = non-transport
        return not non_transport

    def on_call(self, ev: Event):
        call = ev.payload

        # Track when this CAD call first hit the queue (for queue delay KPIs)
        if "_first_t" not in call:
            call["_first_t"] = self.t

        # Count distinct CAD calls once (even if retried in queue)
        if not call.get("_counted", False):
            self.metrics["n_calls"] += 1
            call["_counted"] = True

        units_needed = int(call.get("units_needed", 1) or 1)
        assigned = call.get("_assigned", 0)
        remaining = max(units_needed - assigned, 0)

        # Decide once per CAD call whether it should be transported at all
        transport_flag = call.get("_transport_flag")
        if transport_flag is None:
            transport_flag = self._decide_transport(call)
            call["_transport_flag"] = transport_flag

        transport_unit_name = call.get("_transport_unit")

        # Try to assign as many units as needed at this event time
        local_assigned = 0
        while remaining > 0:
            # Policy sees actual units; busy_until / can_dispatch already encode availability
            u, resp_minutes, debug = self._pick_unit(call)
            if u is None:
                break

            resp_minutes = float(resp_minutes)
            self.metrics["resp_times"].append(resp_minutes)
            self.metrics["wait_minutes"].append(float(call.get("_retries", 0)))

            # ------------------------------------------------------------------
            # Enriched debug payload for KPI v2
            # ------------------------------------------------------------------
            if isinstance(debug, dict):
                dbg = dict(debug)  # shallow copy to avoid mutating policy internals
            else:
                dbg = {"policy_debug": debug}

            # Call context (from tagged parquet)
            dbg.update(
                {
                    "call_lon": call.get("lon"),
                    "call_lat": call.get("lat"),
                    "call_area": call.get("call_area"),
                    "call_urban_rural": call.get("urban_rural"),
                    "call_zone": call.get("zone"),
                    "call_in_als_boundary": call.get("in_als_boundary"),
                    "call_in_bls_boundary": call.get("in_bls_boundary"),
                    "call_in_overlap_boundary": call.get("in_overlap_boundary"),
                    "call_risk_score": call.get("risk_score"),
                    "call_severity_bucket": call.get("severity_bucket"),
                    "call_preferred_unit_type": call.get("preferred_unit_type"),
                    "call_units_needed": units_needed,
                }
            )

            # Unit context
            dbg.update(
                {
                    "unit_type": u.utype,
                    "unit_zone": getattr(u, "zone", None),
                    "unit_area": getattr(u, "unit_area", None),
                    "unit_station": u.station,
                    "unit_station_lon": getattr(u, "station_lon", None),
                    "unit_station_lat": getattr(u, "station_lat", None),
                    "unit_lon": u.lon,
                    "unit_lat": u.lat,
                }
            )

            # Queue delay from first arrival of this CAD call to assignment of this unit
            first_t = float(call.get("_first_t", self.t))
            queue_delay = max(self.t - first_t, 0.0)
            dbg["queue_delay_min"] = float(queue_delay)

            # Core decision log row
            decision_row = {
                "tmin": float(self.t),
                "call_id": call["id"],
                "unit": u.name,
                "station": u.station,
                "resp_min": resp_minutes,
                "debug": dbg,
            }
            self.metrics["decisions"].append(decision_row)

            # ------------------------------------------------------------------
            # Downstream time components: on-scene, transport, turnaround, return
            # ------------------------------------------------------------------
            onscene = config.ONSCENE_MIN + np.random.gamma(
                2.0, config.ONSCENE_SCALE / 2.0
            )
            abs_epoch_now = call["_abs_epoch_start"] + self.t * 60.0

            # Decide if THIS unit transports (only one when SINGLE_TRANSPORT_PER_CALL is True)
            unit_will_transport = False
            if transport_flag:
                if transport_unit_name is None:
                    unit_will_transport = True
                    transport_unit_name = u.name
                    call["_transport_unit"] = transport_unit_name
                elif not getattr(config, "SINGLE_TRANSPORT_PER_CALL", True):
                    unit_will_transport = True

            to_hosp = 0.0
            turn = 0.0
            if unit_will_transport:
                to_hosp = (
                    traffic.travel_minutes(
                        call["lon"],
                        call["lat"],
                        call["h_lon"],
                        call["h_lat"],
                        config.HOSPITAL_SPEED_MPH,
                        abs_epoch_now,
                    )
                    if call.get("h_lon")
                    else 0.0
                )

                if to_hosp > 0:
                    base_turn = config.TURNAROUND_MIN + np.random.gamma(
                        2.0, config.TURNAROUND_SCALE / 2.0
                    )
                    turn = base_turn
                    hr = int((abs_epoch_now % (24 * 3600)) // 3600)
                    self.metrics["turnaround_hour"].setdefault(hr, []).append(turn)

            # Return to base
            end_lon, end_lat = u.station_lon, u.station_lat
            start_lon = (
                call["h_lon"] if (unit_will_transport and to_hosp > 0) else call["lon"]
            )
            start_lat = (
                call["h_lat"] if (unit_will_transport and to_hosp > 0) else call["lat"]
            )
            ret_speed = (
                config.HOSPITAL_SPEED_MPH
                if (unit_will_transport and to_hosp > 0)
                else config.SCENE_SPEED_MPH
            )
            return_time = traffic.travel_minutes(
                start_lon, start_lat, end_lon, end_lat, ret_speed, abs_epoch_now
            )

            total_busy = resp_minutes + onscene + to_hosp + turn + return_time
            u.busy_until = self.t + total_busy
            u.on_call_id = call["id"]
            u.can_dispatch = False

            # Aggregate metrics
            self.metrics["unit_util"][u.name] += total_busy
            self.metrics["on_scene"].append(onscene)
            if unit_will_transport and to_hosp > 0:
                self.metrics["transport"].append(to_hosp)
            if unit_will_transport and turn > 0:
                self.metrics["turnaround"].append(turn)

            # Enrich debug with time components and transport decision
            dbg["unit_will_transport"] = bool(unit_will_transport)
            dbg["call_transport_flag"] = bool(transport_flag)
            dbg["transport_unit_name"] = transport_unit_name
            dbg["on_scene_min"] = float(onscene)
            dbg["to_hosp_min"] = float(to_hosp)
            dbg["turnaround_min"] = float(turn)
            dbg["return_to_base_min"] = float(return_time)
            dbg["busy_total_min"] = float(total_busy)

            # Schedule when this unit becomes free again
            self.schedule(
                u.busy_until,
                "unit_free",
                unit=u,
                end_lon=end_lon,
                end_lat=end_lat,
            )

            local_assigned += 1
            remaining -= 1

        total_assigned = assigned + local_assigned
        if total_assigned < units_needed:
            retry = call.get("_retries", 0)
            if retry < config.MAX_QUEUE_RETRIES:
                call["_retries"] = retry + 1
                call["_assigned"] = total_assigned
                self.schedule(self.t + 1.0, "call", **call)
            else:
                self.metrics["missed_calls"] += (units_needed - total_assigned)