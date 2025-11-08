#!/usr/bin/env python3
# scripts/sim/sim_baseline.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import heapq, math, random
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ---- Inputs / outputs ----
CALLS_PARQUET = Path("data/processed/medical_calls_lemsa.parquet")
STATIONS_CSV  = Path("reference/lemsa_stations.csv")            # station_number,station_name,lat,lon
UNITS_CSV     = Path("reference/lemsa_units_consolidated.csv")  # unit_designator,unit_type,station_number,...
OUT_CSV       = Path("reports/sim_baseline_summary.csv")
PER_SHIFT_CSV = Path("reports/sim_baseline_per_shift.csv")

# ---- Field mapping ----
TIME_COL_CANDIDATES = [
    "t_received","received_ts","opened_ts","call_received_ts","epoch","ts","eventOpened_ts"
]
LON_CANDIDATES = ["lon","longitude","call_lon","x"]
LAT_CANDIDATES = ["lat","latitude","call_lat","y"]

HOSPITAL_LAT = 40.047144
HOSPITAL_LON = -76.304382

# ---- Models / params ----
RANDOM_SEED = 42
SCENE_SPEED_MPH = 38.0
HOSPITAL_SPEED_MPH = 40.0
ROAD_FACTOR = 1.35            # inflate crow-fly distance to road distance
DISPATCH_DELAY_MIN = 1.5      # call-processing + tone-out
ONSCENE_MIN, ONSCENE_SCALE = 8.0, 12.0
TURNAROUND_MIN, TURNAROUND_SCALE = 10.0, 15.0

SEGMENT_BY_SHIFT = True
SHIFT_WINDOWS = [(0, 480), (480, 960), (960, 1440)]  # [00-08), [08-16), [16-24)

# ---- Queue semantics ----
MAX_QUEUE_RETRIES = 15   # 1-min spaced retries before marking missed

# ----------------------------------------------------------------------

def haversine_mi(lon1, lat1, lon2, lat2):
    R = 3958.7613  # miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def travel_minutes_mph(mi, mph):
    return 60.0 * (mi / max(mph, 1e-6))

def rnd_onscene():
    # min + Gamma(k=2, theta=scale/2)
    return ONSCENE_MIN + np.random.gamma(2.0, ONSCENE_SCALE/2.0)

def rnd_turnaround():
    return TURNAROUND_MIN + np.random.gamma(2.0, TURNAROUND_SCALE/2.0)

def which_shift(min_in_day: int) -> int:
    for i, (a, b) in enumerate(SHIFT_WINDOWS):
        if a <= min_in_day < b:
            return i
    return len(SHIFT_WINDOWS) - 1

def segment_calls_by_shift(calls: list[dict]) -> tuple[dict[str, list[dict]], dict[str, float]]:
    """
    Group calls into YYYY-MM-DD_S{0,1,2}.
    Segment start is the true shift boundary.
    Recompute c["tmin"] relative to that boundary.
    """
    groups: dict[str, list[dict]] = {}
    seg_start_abs: dict[str, float] = {}

    for c in calls:
        abs_t = float(c["_abs_epoch"])
        d = datetime.fromtimestamp(abs_t, tz=timezone.utc)
        tod_min = int(abs_t % (24*3600) // 60)
        sidx = which_shift(tod_min)
        key = f"{d.date().isoformat()}_S{sidx}"
        groups.setdefault(key, []).append(c)

    for key, seg in groups.items():
        seg.sort(key=lambda x: x["_abs_epoch"])
        # compute real shift boundary
        day = datetime.fromtimestamp(float(seg[0]["_abs_epoch"]), tz=timezone.utc).date()
        sidx = int(key[-1])  # '..._S{0,1,2}'
        start_min, _ = SHIFT_WINDOWS[sidx]
        shift_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(minutes=start_min)
        seg_start_abs[key] = shift_start.timestamp()

        t0 = seg_start_abs[key]
        for c in seg:
            c["tmin"] = (float(c["_abs_epoch"]) - t0) / 60.0

    return groups, seg_start_abs

# ---------------- DES core ----------------
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
    busy_until: float = 0.0   # minutes from current segment start
    on_call_id: str | None = None

class DES:
    def __init__(self):
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
            "missed_calls": 0
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

    def on_call(self, ev: Event):
        call = ev.payload

        # count once per incident
        if not call.get("_counted", False):
            self.metrics["n_calls"] += 1
            call["_counted"] = True

        # available units
        avail = [u for u in self.units if u.busy_until <= self.t]
        if not avail:
            retry_count = call.get("_retries", 0)
            if retry_count < MAX_QUEUE_RETRIES:
                call["_retries"] = retry_count + 1
                self.schedule(self.t + 1.0, "call", **call)  # 1-min backoff
            else:
                self.metrics["missed_calls"] += 1
            return

        # nearest feasible with road inflation and dispatch delay
        best = None
        best_t_travel = 1e9
        for u in avail:
            mi = haversine_mi(u.lon, u.lat, call["lon"], call["lat"]) * ROAD_FACTOR
            tmin = DISPATCH_DELAY_MIN + travel_minutes_mph(mi, SCENE_SPEED_MPH)
            if tmin < best_t_travel:
                best_t_travel = tmin
                best = u
        if best is None:
            self.metrics["missed_calls"] += 1
            return

        u = best
        resp_minutes = best_t_travel
        self.metrics["resp_times"].append(resp_minutes)

        waited_min = float(call.get("_retries", 0)) * 1.0
        self.metrics["wait_minutes"].append(waited_min)

        # scene time
        onscene = rnd_onscene()

        # transport to hospital (inflated)
        to_hosp = travel_minutes_mph(
            haversine_mi(call["lon"], call["lat"], call["h_lon"], call["h_lat"]) * ROAD_FACTOR,
            HOSPITAL_SPEED_MPH
        ) if call.get("h_lon") else 0.0

        turn = rnd_turnaround() if to_hosp > 0 else 0.0

        # return to base (inflated)
        if to_hosp > 0:
            return_dist = haversine_mi(call["h_lon"], call["h_lat"], u.station_lon, u.station_lat) * ROAD_FACTOR
            return_time = travel_minutes_mph(return_dist, HOSPITAL_SPEED_MPH)
            end_lon, end_lat = u.station_lon, u.station_lat
        else:
            return_dist = haversine_mi(call["lon"], call["lat"], u.station_lon, u.station_lat) * ROAD_FACTOR
            return_time = travel_minutes_mph(return_dist, SCENE_SPEED_MPH)
            end_lon, end_lat = u.station_lon, u.station_lat

        total_busy = resp_minutes + onscene + to_hosp + turn + return_time
        u.busy_until = self.t + total_busy
        u.on_call_id = call["id"]

        self.metrics["unit_util"][u.name] += total_busy
        self.metrics["on_scene"].append(onscene)
        if to_hosp > 0: self.metrics["transport"].append(to_hosp)
        if turn > 0: self.metrics["turnaround"].append(turn)

        # free back at station
        self.schedule(u.busy_until, "unit_free", unit=u, end_lon=end_lon, end_lat=end_lat)

# ---------------- Runner ----------------
def load_calls():
    df = pd.read_parquet(CALLS_PARQUET)

    # time column detection
    time_col = None
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            time_col = c; break
        for cc in df.columns:
            if cc.lower() == c.lower():
                time_col = cc; break
        if time_col: break
    if time_col is None:
        for cc in df.columns:
            if "time" in cc.lower() or "open" in cc.lower() or "recv" in cc.lower():
                try:
                    df["_t"] = pd.to_datetime(df[cc]).astype("int64") // 10**9
                    time_col = "_t"; break
                except Exception:
                    pass
    if time_col is None:
        raise RuntimeError("Could not find a time column. Add one of TIME_COL_CANDIDATES or a datetime field.")

    # coords
    lon = next((c for c in LON_CANDIDATES if c in df.columns), None)
    lat = next((c for c in LAT_CANDIDATES if c in df.columns), None)
    if lon is None or lat is None:
        raise RuntimeError("Missing lon/lat columns in LEMSA calls parquet.")

    # hospital coords (Lancaster General for all calls)
    df["h_lon"] = HOSPITAL_LON
    df["h_lat"] = HOSPITAL_LAT

    # select and sort
    calls = (df[[time_col, lon, lat, "h_lon", "h_lat"]]
             .rename(columns={time_col: "t", lon: "lon", lat: "lat"})
             .sort_values("t")
             .reset_index(drop=True))

    # absolute epoch seconds
    if pd.api.types.is_datetime64_any_dtype(calls["t"]):
        abs_epoch = calls["t"].astype("int64") // 10**9
    else:
        abs_epoch = pd.to_numeric(calls["t"], errors="coerce")
        if abs_epoch.isna().any():
            parsed = pd.to_datetime(calls["t"], errors="coerce")
            if parsed.isna().all():
                raise RuntimeError("Time column is neither datetime nor numeric epoch.")
            abs_epoch = parsed.astype("int64") // 10**9

    calls["_abs_epoch"] = abs_epoch.astype(float)

    # global-relative minutes (not used for shifts but kept)
    t0 = float(calls["_abs_epoch"].iloc[0])
    calls["tmin"] = (calls["_abs_epoch"] - t0) / 60.0

    # time-of-day minutes [0,1440)
    calls["tod_min"] = (calls["_abs_epoch"] % (24*3600) // 60).astype(int)

    calls["id"] = calls.index.astype(str)
    return calls[["id", "tmin", "lon", "lat", "h_lon", "h_lat", "_abs_epoch", "tod_min"]].to_dict("records")

def load_units():
    st = pd.read_csv(STATIONS_CSV)
    # normalize columns
    cols_lower = {c.lower(): c for c in st.columns}
    if "latitude" in cols_lower: st = st.rename(columns={cols_lower["latitude"]:"lat"})
    if "longitude" in cols_lower: st = st.rename(columns={cols_lower["longitude"]:"lon"})
    if "station_number" not in {c.lower() for c in st.columns}:
        raise RuntimeError("stations CSV must include station_number, lat, lon")

    units_df = pd.read_csv(UNITS_CSV)
    units_df["station_number"] = units_df["station_number"].astype(str).str.upper()
    st["station_number"] = st["station_number"].astype(str).str.upper()

    m = units_df.merge(st[["station_number","lat","lon"]], on="station_number", how="left")
    m = m.dropna(subset=["lat","lon"])

    units = []
    for _, r in m.iterrows():
        station_lon = float(r["lon"])
        station_lat = float(r["lat"])
        units.append(Unit(
            name=str(r["unit_designator"]).upper(),
            utype=str(r.get("unit_type","")).upper(),
            station=str(r["station_number"]),
            lon=station_lon, lat=station_lat,
            station_lon=station_lon, station_lat=station_lat
        ))
    return units

def run_one_segment(
    calls_segment: list[dict],
    unit_state: dict[str, dict],
    seg_start_abs: float
) -> tuple[dict, dict]:
    """
    Build a DES for this shift. Units carry over busy state via abs_free_epoch.
    Return (row, updated_unit_state).
    """
    sim = DES()
    new_units: list[Unit] = []

    # materialize units for this segment
    for name, s in unit_state.items():
        remaining_min = max(0.0, (s["abs_free_epoch"] - seg_start_abs) / 60.0)
        if remaining_min > 0:
            lon, lat = s["lon"], s["lat"]
        else:
            lon, lat = s["station_lon"], s["station_lat"]

        u = Unit(
            name=name,
            utype=s["utype"],
            station=s["station"],
            lon=lon, lat=lat,
            station_lon=s["station_lon"], station_lat=s["station_lat"],
            busy_until=remaining_min if remaining_min > 0 else 0.0
        )
        new_units.append(u)
        sim.add_unit(u)
        if u.busy_until > 0.0:
            # schedule release at remaining time; keep current coords on release
            sim.schedule(u.busy_until, "unit_free", unit=u, end_lon=u.lon, end_lat=u.lat)

    # schedule calls
    for c in calls_segment:
        sim.schedule(c["tmin"], "call", **c)

    # run DES
    while sim.advance():
        pass

    # updated unit state to absolute epochs
    updated_state = {}
    for u in new_units:
        abs_free = seg_start_abs + max(u.busy_until, 0.0) * 60.0
        updated_state[u.name] = {
            "abs_free_epoch": abs_free,
            "lon": u.lon,
            "lat": u.lat,
            "station_lon": u.station_lon,
            "station_lat": u.station_lat,
            "utype": u.utype,
            "station": u.station,
        }

    # KPIs
    resp = np.array(sim.metrics["resp_times"]) if sim.metrics["resp_times"] else np.array([])
    row = {
        "n_calls": sim.metrics["n_calls"],
        "missed_calls": sim.metrics["missed_calls"],
        "p50_resp_min": np.percentile(resp, 50) if len(resp) else np.nan,
        "p90_resp_min": np.percentile(resp, 90) if len(resp) else np.nan,
        "avg_resp_min": float(resp.mean()) if len(resp) else np.nan,
        "p50_wait_min": np.percentile(sim.metrics["wait_minutes"], 50) if sim.metrics["wait_minutes"] else np.nan,
        "p90_wait_min": np.percentile(sim.metrics["wait_minutes"], 90) if sim.metrics["wait_minutes"] else np.nan,
        "avg_onscene_min": np.mean(sim.metrics["on_scene"]) if sim.metrics["on_scene"] else np.nan,
        "avg_transport_min": np.mean(sim.metrics["transport"]) if sim.metrics["transport"] else np.nan,
        "avg_turnaround_min": np.mean(sim.metrics["turnaround"]) if sim.metrics["turnaround"] else np.nan,
        "units": len(new_units),
    }
    return row, updated_state

def main():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    calls = load_calls()
    units = load_units()

    if not SEGMENT_BY_SHIFT:
        # original single-run behavior
        sim = DES()
        for u in units: sim.add_unit(u)
        for c in calls: sim.schedule(c["tmin"], "call", **c)
        while sim.advance(): pass

        resp = np.array(sim.metrics["resp_times"]) if sim.metrics["resp_times"] else np.array([])
        out = pd.DataFrame({
            "n_calls":[sim.metrics["n_calls"]],
            "p50_resp_min":[np.percentile(resp,50) if len(resp) else np.nan],
            "p90_resp_min":[np.percentile(resp,90) if len(resp) else np.nan],
            "avg_resp_min":[float(resp.mean()) if len(resp) else np.nan],
            "avg_onscene_min":[np.mean(sim.metrics["on_scene"]) if sim.metrics["on_scene"] else np.nan],
            "avg_transport_min":[np.mean(sim.metrics["transport"]) if sim.metrics["transport"] else np.nan],
            "avg_turnaround_min":[np.mean(sim.metrics["turnaround"]) if sim.metrics["turnaround"] else np.nan],
            "units":[len(units)],
        })
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
        print(out.to_string(index=False))
        print(f"✅ Baseline summary → {OUT_CSV}")
        return

    # shift-segmented runs
    groups, seg_start_abs = segment_calls_by_shift(calls)

    # initialize unit state once from load_units()
    unit_state = {
        u.name: {
            "abs_free_epoch": -float("inf"),   # free now
            "lon": u.lon,
            "lat": u.lat,
            "station_lon": u.station_lon,
            "station_lat": u.station_lat,
            "utype": u.utype,
            "station": u.station,
        }
        for u in units
    }

    rows = []
    for key, seg in sorted(groups.items()):
        start_abs = seg_start_abs[key]
        row, unit_state = run_one_segment(seg, unit_state, start_abs)
        row["segment"] = key
        rows.append(row)

    # sanity: units must exist for all segments
    assert all(r["units"] > 0 for r in rows), "No units materialized in some segments."

    print(f"📦 calls loaded: {len(calls)}  units: {len(units)}")
    print(f"SEGMENT_BY_SHIFT={SEGMENT_BY_SHIFT}")
    print(f"🗂 segments: {len(groups)}  sample: {list(sorted(groups.keys())[:3])}")

    # write per-shift table
    PER_SHIFT_CSV.parent.mkdir(parents=True, exist_ok=True)
    per_shift_df = pd.DataFrame(rows).sort_values("segment")
    per_shift_df.to_csv(PER_SHIFT_CSV, index=False)
    print(f"→ writing per-shift CSV to {PER_SHIFT_CSV.resolve()}")

    # weighted aggregate over shifts
    if len(per_shift_df):
        w = per_shift_df["n_calls"].values
        def wavg(col):
            v = per_shift_df[col].values
            m = ~np.isnan(v)
            return np.average(v[m], weights=w[m]) if m.any() else np.nan
        out = pd.DataFrame({
            "shifts":[len(per_shift_df)],
            "total_calls":[int(per_shift_df["n_calls"].sum())],
            "w_avg_p50_resp_min":[wavg("p50_resp_min")],
            "w_avg_p90_resp_min":[wavg("p90_resp_min")],
            "w_avg_avg_resp_min":[wavg("avg_resp_min")],
            "units":[len(units)],
        })
    else:
        out = pd.DataFrame({"shifts":[0], "total_calls":[0], "units":[len(units)]})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"→ writing aggregate CSV to {OUT_CSV.resolve()}")
    print(out.to_string(index=False))
    print(f"🕘 Per-shift summaries → {PER_SHIFT_CSV}")
    print(f"✅ Aggregated summary → {OUT_CSV}")

if __name__ == "__main__":
    main()