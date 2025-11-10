from __future__ import annotations
from dataclasses import dataclass, field
import heapq, numpy as np
from collections import defaultdict
from . import config, traffic
from . import hospital

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
    # optional flag for duty windows; runner may set it
    can_dispatch: bool = True

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
            # hourly breakdowns for turnaround
            "turnaround_hour": defaultdict(list),  # hour:int -> list[minutes]
        }

    # core queue
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

    # events
    def on_unit_free(self, ev: Event):
        u: Unit = ev.payload["unit"]
        u.lon = ev.payload.get("end_lon", u.lon)
        u.lat = ev.payload.get("end_lat", u.lat)
        u.on_call_id = None

    def _pick_unit(self, call):
        # filter by duty if present
        eligible = [u for u in self.units if u.busy_until <= self.t and getattr(u, "can_dispatch", True)]
        if not eligible:
            return None, None
        # let policy choose among eligible
        return self.select_unit_fn(eligible, self.t, call)

    def on_call(self, ev: Event):
        call = ev.payload
        if not call.get("_counted", False):
            self.metrics["n_calls"] += 1
            call["_counted"] = True

        u, resp_minutes = self._pick_unit(call)
        if u is None:
            # retry queue with cap
            retry = call.get("_retries", 0)
            if retry < config.MAX_QUEUE_RETRIES:
                call["_retries"] = retry + 1
                self.schedule(self.t + 1.0, "call", **call)
            else:
                self.metrics["missed_calls"] += 1
            return

        # response time
        self.metrics["resp_times"].append(float(resp_minutes))
        self.metrics["wait_minutes"].append(float(call.get("_retries", 0)))

        # absolute "now" for this leg in epoch seconds
        abs_epoch_now = float(call["_abs_epoch_start"]) + self.t * 60.0

        # on-scene time (unchanged distribution)
        onscene = config.ONSCENE_MIN + np.random.gamma(2.0, config.ONSCENE_SCALE / 2.0)

        # transport to hospital using traffic heuristic
        if call.get("h_lon") is not None:
            to_hosp = traffic.travel_minutes(
                call["lon"], call["lat"], call["h_lon"], call["h_lat"],
                config.HOSPITAL_SPEED_MPH, abs_epoch_now
            )
        else:
            to_hosp = 0.0

        # hospital turnaround via hourly model
        turn = hospital.turnaround_minutes(
            abs_epoch_now=abs_epoch_now,
            base_min=config.TURNAROUND_MIN,
            scale=config.TURNAROUND_SCALE,
            hourly_map=config.TA_HOURLY,
            alpha=config.TA_ALPHA,
            ed_load_ratio=0.0,  # placeholder until ED load is wired
        ) if to_hosp > 0 else 0.0

        # return to base using traffic heuristic
        end_lon, end_lat = u.station_lon, u.station_lat
        ret_origin_lon = call["h_lon"] if to_hosp > 0 else call["lon"]
        ret_origin_lat = call["h_lat"] if to_hosp > 0 else call["lat"]
        return_time = traffic.travel_minutes(
            ret_origin_lon, ret_origin_lat, end_lon, end_lat,
            config.HOSPITAL_SPEED_MPH if to_hosp > 0 else config.SCENE_SPEED_MPH,
            abs_epoch_now
        )

        # commit metrics
        self.metrics["on_scene"].append(onscene)
        if to_hosp > 0:
            self.metrics["transport"].append(to_hosp)
        if turn > 0:
            self.metrics["turnaround"].append(turn)
            hour = int((abs_epoch_now % (24 * 3600)) // 3600)
            self.metrics["turnaround_hour"][hour].append(turn)

        total_busy = float(resp_minutes) + onscene + to_hosp + turn + return_time
        u.busy_until = self.t + total_busy
        u.on_call_id = call["id"]
        self.metrics["unit_util"][u.name] += total_busy

        # free at station
        self.schedule(u.busy_until, "unit_free", unit=u, end_lon=end_lon, end_lat=end_lat)