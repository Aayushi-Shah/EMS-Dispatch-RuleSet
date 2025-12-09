# experiments/rulelist_policies/simulator/io.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import config
from .des import Unit


def _pick_time_col(df: pd.DataFrame) -> str:
    """
    Pick a time column from the DataFrame using config.TIME_COL_CANDIDATES
    and a few fallbacks.
    """
    # 1) Try config-specified candidates (case-insensitive)
    for c in getattr(config, "TIME_COL_CANDIDATES", []):
        if c in df.columns:
            return c
        for cc in df.columns:
            if cc.lower() == c.lower():
                return cc

    # 2) Fallback: any column with "time"/"open"/"recv" that parses as datetime
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
      - config.CALLS_PARQUET points to the tagged file produced by
        build_lemsa_tagged (e.g., data/processed/medical_calls_lemsa_tagged.parquet).
      - That file already contains ALS/BLS boundary flags, zone, and
        urban/rural and risk annotations.

    Returns a list of dicts with, at minimum, the following keys:
      id, tmin, lon, lat, h_lon, h_lat, _abs_epoch, tod_min,
      zone, urban_rural, call_area,
      in_als_boundary, in_bls_boundary, in_overlap_boundary,
      risk_score, severity_bucket, preferred_unit_type,
      units_needed, description, incidentType.
    """
    calls_path = Path(config.CALLS_PARQUET)
    if not calls_path.exists():
        raise RuntimeError(f"CALLS_PARQUET not found: {calls_path}")

    df = pd.read_parquet(calls_path)

    # --- Core coordinate + time columns ---
    tcol = _pick_time_col(df)

    lon = next((c for c in getattr(config, "LON_CANDIDATES", []) if c in df.columns), None)
    lat = next((c for c in getattr(config, "LAT_CANDIDATES", []) if c in df.columns), None)
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
    base_cols = [
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
    optional_cols = [
        c
        for c in [
            "incidentID",
            "municipality",
            "municipality_std",
            "is_critical_municipality",
            "critical_zone_type",
        ]
        if c in df.columns
    ]
    calls = (
        df[base_cols + optional_cols]
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

    # tmin relative to first call in this run
    t0 = float(calls["_abs_epoch"].iloc[0])
    calls["tmin"] = (calls["_abs_epoch"] - t0) / 60.0

    # time-of-day in minutes (for potential hourly/diurnal KPIs)
    calls["tod_min"] = (calls["_abs_epoch"] % (24 * 3600) // 60).astype(int)

    calls["call_id"] = calls["id"]

    base_return = [
        "id",
        "call_id",
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
    final_cols = list(dict.fromkeys(base_return + optional_cols))
    return calls[final_cols].to_dict("records")


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
    units_path = Path(config.UNITS_CSV)
    if not units_path.exists():
        raise RuntimeError(f"UNITS_CSV not found: {units_path}")

    units_df = pd.read_csv(units_path)

    if "station_number" not in units_df.columns:
        raise RuntimeError("UNITS_CSV must contain 'station_number'.")
    units_df["station_number"] = units_df["station_number"].astype(str).str.upper()

    if not {"station_lon", "station_lat"}.issubset(units_df.columns):
        raise RuntimeError("UNITS_CSV must contain 'station_lon' and 'station_lat'.")

    m = units_df.copy()

    has_unit_zone = "unit_zone" in m.columns
    has_unit_area = "unit_area" in m.columns
    has_muni = "municipality" in m.columns
    has_muni_std = "municipality_std" in m.columns
    has_crit_muni = "is_critical_municipality" in m.columns

    units: list[Unit] = []
    for _, r in m.iterrows():
        station_lon = float(r["station_lon"])
        station_lat = float(r["station_lat"])

        zone = r["unit_zone"] if has_unit_zone else None
        unit_area = r["unit_area"] if has_unit_area else None
        muni = r["municipality"] if has_muni else None
        muni_std = r["municipality_std"] if has_muni_std else None
        crit_raw = r["is_critical_municipality"] if has_crit_muni else None

        u = Unit(
            name=str(r["unit_designator"]).upper(),
            utype=str(r.get("utype") or r.get("unit_type", "")).upper(),
            station=str(r["station_number"]),
            lon=station_lon,
            lat=station_lat,
        )
        # Attach extra attributes we care about for ALS/BLS and coverage rules
        u.station_lon = station_lon
        u.station_lat = station_lat
        u.zone = zone
        u.unit_zone = zone
        u.unit_area = unit_area
        if muni is not None:
            u.municipality = str(muni)
        if muni_std is not None:
            u.municipality_std = str(muni_std)
        if crit_raw is not None:
            if isinstance(crit_raw, str):
                crit_val = crit_raw.strip().lower() in ("1", "true", "yes", "y")
            else:
                try:
                    crit_val = bool(int(crit_raw))
                except Exception:
                    crit_val = bool(crit_raw)
            u.is_critical_municipality = crit_val

        units.append(u)

    return units
