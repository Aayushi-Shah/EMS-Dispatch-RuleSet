from __future__ import annotations
import pandas as pd
from datetime import datetime, timezone, timedelta
from . import config

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

    calls = (df[[tcol, lon, lat, "h_lon", "h_lat"]]
             .rename(columns={tcol:"t", lon:"lon", lat:"lat"})
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

    return calls[["id","tmin","lon","lat","h_lon","h_lat","_abs_epoch","tod_min"]].to_dict("records")

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

    from .des import Unit
    units = []
    for _, r in m.iterrows():
        station_lon = float(r["lon"]); station_lat = float(r["lat"])
        units.append(Unit(
            name=str(r["unit_designator"]).upper(),
            utype=str(r.get("unit_type","")).upper(),
            station=str(r["station_number"]),
            lon=station_lon, lat=station_lat,
            station_lon=station_lon, station_lat=station_lat
        ))
    return units

def which_shift(min_in_day: int) -> int:
    for i, (a, b) in enumerate(config.SHIFT_WINDOWS):
        if a <= min_in_day < b:
            return i
    return len(config.SHIFT_WINDOWS) - 1

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