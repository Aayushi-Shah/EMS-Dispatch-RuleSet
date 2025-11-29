from __future__ import annotations
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Set, Tuple
from shapely.geometry import Point

from scripts.simulator import config
from scripts.simulator.geo import in_any_polygon, load_boundary, load_zones, zone_lookup_factory
from scripts.simulator.des import Unit


_BOUNDARIES = {
    "als": None,
    "bls": None,
    "overlap": None,
}

ZONES = load_zones({
    "ALS": "reference/lemsa_als_boundary.geojson",
    "BLS": "reference/lemsa_bls_boundary.geojson",
    "OVERLAP": "reference/lemsa_overlap_boundary.geojson"
})
ZONE_LOOKUP = zone_lookup_factory(ZONES)

def _pick_time_col(df: pd.DataFrame):
    for c in config.TIME_COL_CANDIDATES:
        if c in df.columns: return c
        for cc in df.columns:
            if cc.lower() == c.lower(): return cc
    for cc in df.columns:
        if "time" in cc.lower() or "open" in cc.lower() or "recv" in cc.lower():
            try:
                df["_t"] = pd.to_datetime(df[cc]).astype("int64") // 10**9
                return "_t"
            except Exception:
                pass
    raise RuntimeError("No time column found.")

def load_calls():
    df = pd.read_parquet(config.CALLS_PARQUET)

    tcol = _pick_time_col(df)
    lon = next((c for c in config.LON_CANDIDATES if c in df.columns), None)
    lat = next((c for c in config.LAT_CANDIDATES if c in df.columns), None)
    if lon is None or lat is None:
        raise RuntimeError("Missing lon/lat columns.")

    df["h_lon"] = config.HOSPITAL_LON
    df["h_lat"] = config.HOSPITAL_LAT

    calls = (df[[tcol, lon, lat, "h_lon", "h_lat", "numberOfUnits", "description", "incidentType"]]
             .rename(columns={tcol:"t", lon:"lon", lat:"lat", "numberOfUnits": "units_needed"})
             .sort_values("t")
             .reset_index(drop=True))

    if pd.api.types.is_datetime64_any_dtype(calls["t"]):
        abs_epoch = calls["t"].astype("int64") // 10**9
    else:
        abs_epoch = pd.to_numeric(calls["t"], errors="coerce")
        if abs_epoch.isna().any():
            parsed = pd.to_datetime(calls["t"], errors="coerce")
            if parsed.isna().all():
                raise RuntimeError("Time column not parseable.")
            abs_epoch = parsed.astype("int64") // 10**9

    calls["_abs_epoch"] = abs_epoch.astype(float)
    t0 = float(calls["_abs_epoch"].iloc[0])
    calls["tmin"] = (calls["_abs_epoch"] - t0) / 60.0
    calls["tod_min"] = (calls["_abs_epoch"] % (24*3600) // 60).astype(int)
    calls["id"] = calls.index.astype(str)

    # NEW: tag each call with zone based on incident lon/lat
    calls["zone"] = calls.apply(
        lambda r: ZONE_LOOKUP(r["lon"], r["lat"]),
        axis=1
    )

    return calls[["id","tmin","lon","lat","h_lon","h_lat","_abs_epoch","tod_min","zone","units_needed","description","incidentType"]].to_dict("records")

def load_units():
    st = pd.read_csv(config.STATIONS_CSV)
    cols_lower = {c.lower(): c for c in st.columns}
    if "latitude" in cols_lower: st = st.rename(columns={cols_lower["latitude"]:"lat"})
    if "longitude" in cols_lower: st = st.rename(columns={cols_lower["longitude"]:"lon"})
    if "station_number" not in {c.lower() for c in st.columns}:
        raise RuntimeError("stations CSV must include station_number, lat, lon")

    units_df = pd.read_csv(config.UNITS_CSV)
    units_df["station_number"] = units_df["station_number"].astype(str).str.upper()
    st["station_number"] = st["station_number"].astype(str).str.upper()

    m = units_df.merge(st[["station_number","lat","lon"]], on="station_number", how="left").dropna(subset=["lat","lon"])

    
    units = []
    for _, r in m.iterrows():
        station_lon = float(r["lon"]); station_lat = float(r["lat"])
        zone = ZONE_LOOKUP(station_lon, station_lat)

        u = Unit(
            name=str(r["unit_designator"]).upper(),
            utype=str(r.get("unit_type","")).upper(),
            station=str(r["station_number"]),
            lon=station_lon, lat=station_lat,
            station_lon=station_lon, station_lat=station_lat
        )
        # NEW: attach zone to unit
        u.zone = zone

        units.append(u)
    return units

def which_shift(min_in_day: int) -> int:
    for i, (a, b) in enumerate(config.SHIFT_WINDOWS):
        if a <= min_in_day < b:
            return i
    return len(config.SHIFT_WINDOWS) - 1

def load_unit_duty() -> dict[str, list[tuple[set[int], int, int]]]:
    """
    Returns: { UNIT -> [ (days_set, start_min, end_min), ... ] }
    """
    try:
        df = pd.read_csv(config.DUTY_CSV)
    except FileNotFoundError:
        return {}  # no enforcement if file missing

    need = {"unit_designator","day","window_start_min","window_end_min"}
    if not need.issubset({c.lower() for c in df.columns}):
        # be tolerant to case
        cols = {c.lower(): c for c in df.columns}
        df = df.rename(columns={cols.get("unit_designator","unit_designator"): "unit_designator",
                                cols.get("day","day"): "day",
                                cols.get("window_start_min","window_start_min"): "window_start_min",
                                cols.get("window_end_min","window_end_min"): "window_end_min"})
    duty: dict[str, list[tuple[set[int], int, int]]] = {}
    for _, r in df.iterrows():
        u = str(r["unit_designator"]).upper()
        day = str(r["day"]).strip().lower()
        if day == "all":
            days = set(range(7))
        else:
            # allow comma list like "0,1,2"
            days = set(int(d) for d in str(day).split(","))
        s = int(r["window_start_min"]); e = int(r["window_end_min"])
        duty.setdefault(u, []).append((days, s, e))
    return duty

def segment_calls_by_shift(calls: list[dict]):
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
        day = datetime.fromtimestamp(float(seg[0]["_abs_epoch"]), tz=timezone.utc).date()
        sidx = int(key.split("_S")[-1])
        start_min, _ = config.SHIFT_WINDOWS[sidx]
        shift_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(minutes=start_min)
        seg_start_abs[key] = shift_start.timestamp()
        t0 = seg_start_abs[key]
        for c in seg:
            c["tmin"] = (float(c["_abs_epoch"]) - t0) / 60.0

    return groups, seg_start_abs

def tag_calls_in_bounds(calls: List[Dict], geoms) -> List[Dict]:
    if not geoms:
        for c in calls: c["in_bounds"] = True
        return calls
    for c in calls:
        c["in_bounds"] = in_any_polygon(c["lon"], c["lat"], geoms)
    return calls

def filter_units_in_bounds(units, geoms):
    if not geoms:
        return units
    keep = []
    from .geo import in_any_polygon
    for u in units:
        if in_any_polygon(u.station_lon, u.station_lat, geoms):
            keep.append(u)
    return keep

def load_boundaries_if_needed(config):
    """Lazy load ALS/BLS/Overlap boundaries once per program."""
    if _BOUNDARIES["als"] is None:
        _BOUNDARIES["als"] = load_boundary(str(config.ALS_BOUNDARY))
    if _BOUNDARIES["bls"] is None:
        _BOUNDARIES["bls"] = load_boundary(str(config.BLS_BOUNDARY))
    if _BOUNDARIES["overlap"] is None:
        _BOUNDARIES["overlap"] = load_boundary(str(config.OVERLAP_BOUNDARY))


def tag_calls_with_boundaries(calls: list[dict], config):
    """
    For each call, add:
        call["in_als_boundary"]
        call["in_bls_boundary"]
        call["in_overlap_boundary"]
    """
    load_boundaries_if_needed(config)

    als_poly = _BOUNDARIES["als"]
    bls_poly = _BOUNDARIES["bls"]
    overlap_poly = _BOUNDARIES["overlap"]

    for c in calls:
        p = Point(c["lon"], c["lat"])
        c["in_als_boundary"] = als_poly.contains(p) if als_poly else False
        c["in_bls_boundary"] = bls_poly.contains(p) if bls_poly else False
        c["in_overlap_boundary"] = overlap_poly.contains(p) if overlap_poly else False

    return calls
