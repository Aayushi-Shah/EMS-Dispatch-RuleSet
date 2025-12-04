from __future__ import annotations

import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict
from pathlib import Path

from scripts.simulator import config
from scripts.simulator.des import Unit

def _pick_time_col(df: pd.DataFrame) -> str:
    """
    Pick a time column from the DataFrame using config.TIME_COL_CANDIDATES
    and a few fallbacks.
    """
    for c in config.TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
        for cc in df.columns:
            if cc.lower() == c.lower():
                return cc
    for cc in df.columns:
        if "time" in cc.lower() or "open" in cc.lower() or "recv" in cc.lower():
            try:
                df["_t"] = pd.to_datetime(df[cc]).astype("int64") // 10**9
                return "_t"
            except Exception:
                pass
    raise RuntimeError("No time column found in calls parquet.")


def load_calls() -> list[dict]:
    """
    Load pre-tagged LEMSA calls from CALLS_PARQUET and adapt them into
    the simulator's call dict format.

    Assumptions:
      - CALLS_PARQUET points to the tagged file produced by build_lemsa_tagged
        (e.g., data/processed/medical_calls_lemsa_tagged.parquet).
      - That file already contains ALS/BLS boundary flags, zone, and
        urban/rural and risk annotations.

    Returns a list of dicts with, at minimum, the following keys:
      id, tmin, lon, lat, h_lon, h_lat, _abs_epoch, tod_min,
      zone, urban_rural, call_area,
      in_als_boundary, in_bls_boundary, in_overlap_boundary,
      risk_score, severity_bucket, preferred_unit_type,
      units_needed, description, incidentType.
    """
    df = pd.read_parquet(config.CALLS_PARQUET)

    # --- Core coordinate + time columns ---
    tcol = _pick_time_col(df)
    lon = next((c for c in config.LON_CANDIDATES if c in df.columns), None)
    lat = next((c for c in config.LAT_CANDIDATES if c in df.columns), None)
    if lon is None or lat is None:
        raise RuntimeError("Missing lon/lat columns in calls parquet.")

    # Hospital coordinates (single receiving hospital for now).
    df["h_lon"] = float(config.HOSPITAL_LON)
    df["h_lat"] = float(config.HOSPITAL_LAT)

    # --- Boundary flags: trust pre-tagged cols, but backfill if missing ---
    for col, default in {
        "in_als_boundary": False,
        "in_bls_boundary": False,
        "in_overlap_boundary": False,
    }.items():
        if col not in df.columns:
            df[col] = default

    # --- Zone + area (must exist in tagged data) ---
    required_area_cols = ["zone", "urban_rural", "call_area"]
    missing_area = [c for c in required_area_cols if c not in df.columns]
    if missing_area:
        raise RuntimeError(f"Missing required columns in calls parquet: {missing_area}")

    # --- Risk / severity annotations (backfill if missing) ---
    if "risk_score" not in df.columns:
        df["risk_score"] = None
    if "severity_bucket" not in df.columns:
        df["severity_bucket"] = "unknown"
    if "preferred_unit_type" not in df.columns:
        df["preferred_unit_type"] = None

    # --- Units-needed column (CAD "numberOfUnits" or equivalent) ---
    units_col = None
    for cand in ["numberOfUnits", "units_needed", "num_units"]:
        if cand in df.columns:
            units_col = cand
            break
    if units_col is None:
        df["units_needed"] = 1
        units_col = "units_needed"

    # --- Incident identifier: prefer CAD incidentID if present ---
    if "incidentID" in df.columns:
        df["id"] = df["incidentID"].astype(str)
    elif "id" in df.columns:
        df["id"] = df["id"].astype(str)
    else:
        df["id"] = df.index.astype(str)

    # Ensure description / incidentType exist for debugging, even if empty.
    if "description" not in df.columns:
        df["description"] = ""
    if "incidentType" not in df.columns:
        df["incidentType"] = ""

    # --- Project to working DataFrame ---
    calls = (
        df[
            [
                "id",
                tcol,
                lon,
                lat,
                "h_lon",
                "h_lat",
                units_col,
                "description",
                "incidentType",
                "in_als_boundary",
                "in_bls_boundary",
                "in_overlap_boundary",
                "zone",
                "urban_rural",
                "call_area",
                "risk_score",
                "severity_bucket",
                "preferred_unit_type",
            ]
        ]
        .rename(
            columns={
                tcol: "t",
                lon: "lon",
                lat: "lat",
                units_col: "units_needed",
            }
        )
        .sort_values("t")
        .reset_index(drop=True)
    )

    # --- Absolute time + relative tmin + time-of-day ---
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

    # tmin is relative to first call; segment_calls_by_shift will recompute
    # relative to shift start for each segment.
    t0 = float(calls["_abs_epoch"].iloc[0])
    calls["tmin"] = (calls["_abs_epoch"] - t0) / 60.0
    calls["tod_min"] = (calls["_abs_epoch"] % (24 * 3600) // 60).astype(int)

    return calls[
        [
            "id",
            "tmin",
            "lon",
            "lat",
            "h_lon",
            "h_lat",
            "_abs_epoch",
            "tod_min",
            "zone",
            "urban_rural",
            "call_area",
            "in_als_boundary",
            "in_bls_boundary",
            "in_overlap_boundary",
            "risk_score",
            "severity_bucket",
            "preferred_unit_type",
            "units_needed",
            "description",
            "incidentType",
        ]
    ].to_dict("records")


def load_units() -> list[Unit]:
    """
    Load LEMSA units from UNITS_CSV and adapt them into Unit objects.

    Expected primary schema for UNITS_CSV (from build_lemsa_units_from_calls):
      - unit_designator     : unit name (e.g., "MEDIC 06-1")
      - station_number      : station identifier
      - unit_type / utype   : unit type (ALS/BLS/MEDIC/AMB/etc.)
      - station_lon         : station longitude (EPSG:4326)
      - station_lat         : station latitude (EPSG:4326)
      - unit_zone           : "ALS" / "BLS" / "OVERLAP" (optional)
      - unit_area           : "urban" / "rural" / "unknown" (optional)
    """
    units_df = pd.read_csv(config.UNITS_CSV)

    if "station_number" not in units_df.columns:
        raise RuntimeError("UNITS_CSV must contain 'station_number'.")
    units_df["station_number"] = units_df["station_number"].astype(str).str.upper()

    if not {"station_lon", "station_lat"}.issubset(units_df.columns):
        raise RuntimeError("UNITS_CSV must contain 'station_lon' and 'station_lat'.")

    m = units_df.copy()

    has_unit_zone = "unit_zone" in m.columns
    has_unit_area = "unit_area" in m.columns

    units: list[Unit] = []
    for _, r in m.iterrows():
        station_lon = float(r["station_lon"])
        station_lat = float(r["station_lat"])

        zone = r["unit_zone"] if has_unit_zone else None
        unit_area = r["unit_area"] if has_unit_area else None

        u = Unit(
            name=str(r["unit_designator"]).upper(),
            utype=str(r.get("utype") or r.get("unit_type", "")).upper(),
            station=str(r["station_number"]),
            lon=station_lon,
            lat=station_lat,
        )
        # Attach extra attributes for downstream fairness / KPIs
        u.station_lon = station_lon
        u.station_lat = station_lat
        u.zone = zone
        u.unit_area = unit_area

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

    need = {"unit_designator", "day", "window_start_min", "window_end_min"}
    if not need.issubset({c.lower() for c in df.columns}):
        cols = {c.lower(): c for c in df.columns}
        df = df.rename(
            columns={
                cols.get("unit_designator", "unit_designator"): "unit_designator",
                cols.get("day", "day"): "day",
                cols.get("window_start_min", "window_start_min"): "window_start_min",
                cols.get("window_end_min", "window_end_min"): "window_end_min",
            }
        )
    duty: dict[str, list[tuple[set[int], int, int]]] = {}
    for _, r in df.iterrows():
        u = str(r["unit_designator"]).upper()
        day = str(r["day"]).strip().lower()
        if day == "all":
            days = set(range(7))
        else:
            days = set(int(d) for d in str(day).split(","))
        s = int(r["window_start_min"])
        e = int(r["window_end_min"])
        duty.setdefault(u, []).append((days, s, e))
    return duty


def segment_calls_by_shift(calls: list[dict]):
    """
    Segment calls into day/shift buckets based on _abs_epoch and
    config.SHIFT_WINDOWS, and recompute tmin relative to shift start.

    Returns:
      groups: {segment_key -> list[call_dict]}
      seg_start_abs: {segment_key -> shift_start_epoch}
    """
    groups: dict[str, list[dict]] = {}
    seg_start_abs: dict[str, float] = {}

    for c in calls:
        abs_t = float(c["_abs_epoch"])
        d = datetime.fromtimestamp(abs_t, tz=timezone.utc)
        tod_min = int(abs_t % (24 * 3600) // 60)
        sidx = which_shift(tod_min)
        key = f"{d.date().isoformat()}_S{sidx}"
        groups.setdefault(key, []).append(c)

    for key, seg in groups.items():
        seg.sort(key=lambda x: x["_abs_epoch"])
        day = datetime.fromtimestamp(float(seg[0]["_abs_epoch"]), tz=timezone.utc).date()
        sidx = int(key.split("_S")[-1])
        start_min, _ = config.SHIFT_WINDOWS[sidx]
        shift_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(
            minutes=start_min
        )
        seg_start_abs[key] = shift_start.timestamp()
        t0 = seg_start_abs[key]
        for c in seg:
            c["tmin"] = (float(c["_abs_epoch"]) - t0) / 60.0

    return groups, seg_start_abs
