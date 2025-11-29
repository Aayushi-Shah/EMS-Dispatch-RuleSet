# scripts/simulator/des.py
from __future__ import annotations
from dataclasses import dataclass, field
import heapq, numpy as np
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
    zone: str | None = None

class DES:
    def __init__(self, select_unit_fn):
        self.select_unit_fn = select_unit_fn
        self.t = 0.0
        self.Q = []
        self.units = []
        self.metrics = {
            "n_calls": 0,
            "resp_times": [],
            "on_scene": [],
            "transport": [],
            "turnaround": [],
            "unit_util": {},
            "wait_minutes": [],
            "missed_calls": 0,
            "decisions": [],             # per-call debug log
            "turnaround_hour": {},       # hour->list[mins]
        }

    def schedule(self, t, etype, **payload):
        heapq.heappush(self.Q, Event(t, etype, payload))

    def add_unit(self, u: Unit):
        self.units.append(u)
        self.metrics["unit_util"][u.name] = 0.0

    def advance(self):
        if not self.Q: return False
        ev = heapq.heappop(self.Q)
        self.t = ev.t
        getattr(self, f"on_{ev.etype}")(ev)
        return True

    def on_unit_free(self, ev: Event):
        u: Unit = ev.payload["unit"]
        u.lon = ev.payload.get("end_lon", u.lon)
        u.lat = ev.payload.get("end_lat", u.lat)
        u.on_call_id = None

    def _pick_unit(self, call):
        # Policy returns (unit, resp_minutes, debug)
        return self.select_unit_fn(self.units, self.t, call)

    def _decide_transport(self, call: dict) -> bool:
        """
        Decide if the call results in a transport to hospital.
        Uses simple keyword/risk heuristics; falls back to base probability.
        """
        if not getattr(config, "USE_NON_TRANSPORT", True):
            return True  # always transport

        text = f"{call.get('description','')} {call.get('incidentType','')}".lower()
        prob = getattr(config, "NON_TRANSPORT_BASE", 0.1)

        if any(k in text for k in config.NON_TRANSPORT_KEYWORDS):
            prob = max(prob, getattr(config, "NON_TRANSPORT_HIGH_PROB", 0.8))
        if any(k in text for k in config.TRANSPORT_KEYWORDS):
            prob = min(prob, getattr(config, "NON_TRANSPORT_LOW_PROB", 0.0))

        return np.random.random() >= prob  # True = transport, False = non-transport

    def on_call(self, ev: Event):
        call = ev.payload
        if not call.get("_counted", False):
            self.metrics["n_calls"] += 1
            call["_counted"] = True

        units_needed = int(call.get("units_needed", 1) or 1)
        assigned = call.get("_assigned", 0)
        remaining = max(units_needed - assigned, 0)
        transport_flag = call.get("_transport_flag")
        if transport_flag is None:
            transport_flag = self._decide_transport(call)
            call["_transport_flag"] = transport_flag
        transport_unit_name = call.get("_transport_unit")

        # Try to assign as many units as needed at this event time
        local_assigned = 0
        while remaining > 0:
            # Temporarily filter out already assigned units in this loop
            # Policy sees actual units; we rely on busy_until and can_dispatch already set
            u, resp_minutes, debug = self._pick_unit(call)
            if u is None:
                break

            self.metrics["resp_times"].append(float(resp_minutes))
            self.metrics["wait_minutes"].append(float(call.get("_retries", 0)))
            # decision log
            self.metrics["decisions"].append({
                "tmin": float(self.t),
                "call_id": call["id"],
                "unit": u.name,
                "station": u.station,
                "resp_min": float(resp_minutes),
                "debug": debug,
            })

            # on-scene
            onscene = config.ONSCENE_MIN + np.random.gamma(2.0, config.ONSCENE_SCALE/2.0)
            abs_epoch_now = call["_abs_epoch_start"] + self.t*60.0

            # Decide if this unit transports (only one transports when SINGLE_TRANSPORT_PER_CALL is True)
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
                to_hosp = traffic.travel_minutes(
                    call["lon"], call["lat"], call["h_lon"], call["h_lat"],
                    config.HOSPITAL_SPEED_MPH, abs_epoch_now
                ) if call.get("h_lon") else 0.0

                if to_hosp > 0:
                    base_turn = config.TURNAROUND_MIN + np.random.gamma(2.0, config.TURNAROUND_SCALE/2.0)
                    turn = base_turn
                    hr = int((abs_epoch_now % (24*3600)) // 3600)
                    self.metrics["turnaround_hour"].setdefault(hr, []).append(turn)

            # return to base
            end_lon, end_lat = u.station_lon, u.station_lat
            start_lon = call["h_lon"] if (unit_will_transport and to_hosp > 0) else call["lon"]
            start_lat = call["h_lat"] if (unit_will_transport and to_hosp > 0) else call["lat"]
            ret_speed = config.HOSPITAL_SPEED_MPH if (unit_will_transport and to_hosp > 0) else config.SCENE_SPEED_MPH
            return_time = traffic.travel_minutes(start_lon, start_lat, end_lon, end_lat, ret_speed, abs_epoch_now)

            total_busy = resp_minutes + onscene + to_hosp + turn + return_time
            u.busy_until = self.t + total_busy
            u.on_call_id = call["id"]

            self.metrics["unit_util"][u.name] += total_busy
            self.metrics["on_scene"].append(onscene)
            if unit_will_transport and to_hosp > 0: self.metrics["transport"].append(to_hosp)
            if unit_will_transport and turn > 0:    self.metrics["turnaround"].append(turn)

            self.schedule(u.busy_until, "unit_free", unit=u, end_lon=end_lon, end_lat=end_lat)

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
        # else done
