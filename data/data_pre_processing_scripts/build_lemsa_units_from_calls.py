#!/usr/bin/env python3
from __future__ import annotations

"""
build_lemsa_units_from_calls.py

Derive a clean, LEMSA-only ALS/BLS unit catalog from:
- Call-level data (medical_calls_lemsa_tagged.parquet),
- Station locations (lemsa_stations_points.geojson),
- ALS/BLS boundaries + urban/rural polygons,
- Critical municipalities derived from historical CAD.

For each unit_designator (e.g. 'MEDIC 06-3', 'AMB 06-1'), we produce:
- unit_designator
- unit_type (MEDIC/MICU/AMB)
- utype (ALS/BLS)
- station_number
- station_name
- station_lon, station_lat
- municipality_std (majority municipality from calls, OR UNKNOWN)
- is_critical_municipality (bool)
- critical_zone_type ("municipality" / "none")
- unit_zone (ALS/BLS/OVERLAP)
- unit_area (urban/rural/unknown)
- cad_calls (# distinct calls involving this unit)
- source ('derived_from_calls')

Output: reference/lemsa_units_from_calls.csv
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, List

import geopandas as gpd
import pandas as pd


def _add_repo_root_to_sys_path() -> Path:
    """
    Find the project root (directory containing 'simulator') and add it to sys.path.
    This makes imports like 'simulator.runner' work when running the script directly.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "simulator").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    # Fallback: use the immediate parent of this file
    fallback = here.parent
    if str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))
    return fallback


ROOT = _add_repo_root_to_sys_path()

import config
from pre_processing.geo_utils import load_boundary


# -----------------------------
# Config
# -----------------------------

# Unit prefixes we actually model in DES
ALLOWED_UNIT_TYPES = {"AMB", "MEDIC", "MICU"}

UNIT_TYPE_TO_UTYPE = {
    "MEDIC": "ALS",
    "MICU": "ALS",
    "AMB": "BLS",
    # If you ever need others, add here explicitly
}


def pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Return the first column whose name matches any candidate (case-insensitive).
    Raises if none found.
    """
    for c in candidates:
        if c in df.columns:
            return c
        for cc in df.columns:
            if cc.lower() == c.lower():
                return cc
    raise RuntimeError(f"None of {candidates} found in columns: {df.columns}")


def _read_union(path: Path | str | None):
    """
    Read a GeoJSON/shapefile and return the union geometry in WGS84.
    Returns None if path missing/unreadable.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        gdf = gpd.read_file(p)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        try:
            return gdf.union_all()
        except AttributeError:
            return gdf.unary_union
    except Exception:
        return None


def extract_unit_tokens(units_str: Any) -> List[str]:
    """
    Parse unitsString into a list of unit_designator strings.

    Assumes unitsString is like 'MEDIC 06-3, AMB 06-1'.
    Splits on commas/semicolons.
    """
    if pd.isna(units_str):
        return []
    s = str(units_str).strip()
    if not s:
        return []
    parts = re.split(r"[;,]", s)
    tokens = [p.strip() for p in parts if p.strip()]
    return tokens


def parse_unit_designator(designator: str) -> tuple[str, str]:
    """
    Given designator like 'MEDIC 06-3' or 'AMB 06-1', return:
    - unit_type (e.g. 'MEDIC', 'AMB')
    - station_number (e.g. '06-3', '06-1')

    For messy cases like 'AMB 189-1 CHESTER':
    - unit_type = 'AMB'
    - station_number = '189-1'
    Extra trailing tokens are ignored at this stage.
    """
    s = designator.strip()
    if not s:
        return "", ""

    tokens = s.split()
    if len(tokens) == 1:
        # Only a call-sign; no explicit station number
        return tokens[0].upper(), ""

    unit_type = tokens[0].upper()
    station_number = tokens[1]

    return unit_type, station_number


def unit_type_to_utype(unit_type: str) -> str:
    unit_type = (unit_type or "").upper()
    return UNIT_TYPE_TO_UTYPE.get(unit_type, "BLS")  # conservative default


def main():
    ap = argparse.ArgumentParser(
        description="Build LEMSA unit catalog from calls + station geojson."
    )
    ap.add_argument(
        "--calls",
        type=str,
        default="data/processed/medical_calls_lemsa_tagged.parquet",
        help="Calls parquet with unitsString (default: data/processed/medical_calls_lemsa_tagged.parquet)",
    )
    ap.add_argument(
        "--stations",
        type=str,
        default="maps/processed/lemsa_stations_points.geojson",
        help="Station points GeoJSON (default: maps/processed/lemsa_stations_points.geojson)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="reference/lemsa_units_from_calls.csv",
        help="Output units CSV (default: reference/lemsa_units_from_calls.csv)",
    )
    args = ap.parse_args()

    calls_path = Path(args.calls)
    stations_path = Path(args.stations)
    out_path = Path(args.out)

    if not calls_path.exists():
        raise FileNotFoundError(f"Calls file not found: {calls_path}")
    if not stations_path.exists():
        raise FileNotFoundError(f"Stations GeoJSON not found: {stations_path}")

    # -----------------------------
    # Critical municipalities table
    # -----------------------------
    crit_path = Path("tools/critical_zones/critical_zones_municipality.csv")
    critical_municipalities: set[str] = set()
    if crit_path.exists():
        crit_df = pd.read_csv(crit_path)
        muni_col_crit = pick_col(crit_df, ["municipality_std", "municipality"])
        critical_municipalities = {
            str(v).upper().strip()
            for v in crit_df[muni_col_crit].dropna().unique()
        }
        print(f"Loaded {len(critical_municipalities)} critical municipalities from {crit_path}")
    else:
        print(f"⚠️ Critical municipality file not found: {crit_path} — all units will be non-critical.")

    # -----------------------------
    # 1) Load calls and explode units
    # -----------------------------
    calls = pd.read_parquet(calls_path)
    print(f"Loaded calls: {len(calls):,} from {calls_path}")

    units_col = pick_col(calls, ["unitsString", "units", "unit_list"])
    incident_col = pick_col(calls, ["incidentID", "incident_id"])

    try:
        muni_col_calls = pick_col(calls, ["municipality_std", "municipality"])
        print(f"Using call municipality column: {muni_col_calls}")
    except RuntimeError:
        muni_col_calls = None
        print("⚠️ No municipality column found in calls; "
              "unit municipality_std will be 'UNKNOWN' and no units will match critical municipalities.")

    rows = []
    subset_cols = [incident_col, units_col]
    if muni_col_calls:
        subset_cols.append(muni_col_calls)

    for _, row in calls[subset_cols].iterrows():
        units_str = row[units_col]
        incident_id = row[incident_col]
        municipality_val = row[muni_col_calls] if muni_col_calls else None

        for u in extract_unit_tokens(units_str):
            rec = {
                "unit_designator": u,
                incident_col: incident_id,
            }
            if muni_col_calls:
                rec["municipality_std"] = municipality_val
            rows.append(rec)

    if not rows:
        raise RuntimeError("No units could be parsed from calls; check unitsString format.")

    calls_units = pd.DataFrame(rows)
    print(f"Parsed {len(calls_units):,} (incident, unit) pairs")

    # Distinct units and cad_calls
    cad_counts = (
        calls_units
        .groupby("unit_designator")[incident_col]
        .nunique()
        .reset_index()
        .rename(columns={incident_col: "cad_calls"})
    )

    # Majority municipality per unit (from calls)
    if muni_col_calls and "municipality_std" in calls_units.columns:
        muni_counts = (
            calls_units
            .dropna(subset=["municipality_std"])
            .groupby(["unit_designator", "municipality_std"])[incident_col]
            .nunique()
            .reset_index(name="n_calls")
        )
        if len(muni_counts) > 0:
            idx = muni_counts.groupby("unit_designator")["n_calls"].idxmax()
            unit_muni = muni_counts.loc[idx, ["unit_designator", "municipality_std"]]
        else:
            unit_muni = cad_counts[["unit_designator"]].copy()
            unit_muni["municipality_std"] = "UNKNOWN"
    else:
        unit_muni = cad_counts[["unit_designator"]].copy()
        unit_muni["municipality_std"] = "UNKNOWN"

    cad_counts = cad_counts.merge(unit_muni, on="unit_designator", how="left")
    cad_counts["municipality_std"] = cad_counts["municipality_std"].fillna("UNKNOWN")

    # Parse unit_type + station_number from designator
    cad_counts["unit_type"], cad_counts["station_number"] = zip(
        *cad_counts["unit_designator"].apply(parse_unit_designator)
    )

    # Restrict to ALS/BLS unit prefixes we model
    mask_allowed_type = cad_counts["unit_type"].isin(ALLOWED_UNIT_TYPES)
    cad_counts = cad_counts[mask_allowed_type].copy()

    cad_counts["utype"] = cad_counts["unit_type"].apply(unit_type_to_utype)
    cad_counts["source"] = "derived_from_calls"

    # -----------------------------
    # 2) Load station points and join lat/lon + station_name
    # -----------------------------
    gstations = gpd.read_file(stations_path)
    if gstations.crs is not None and gstations.crs.to_epsg() != 4326:
        gstations = gstations.to_crs("EPSG:4326")

    # Extract lon/lat from geometry
    gstations["station_lon"] = gstations.geometry.x
    gstations["station_lat"] = gstations.geometry.y

    # Identify station_number and station_name columns in geojson
    station_num_col = pick_col(gstations, ["station_number", "station_num", "station_id", "id"])
    station_name_col = pick_col(gstations, ["station_name", "name", "station"])

    stations_core = (
        gstations[[station_num_col, station_name_col, "station_lon", "station_lat"]]
        .drop_duplicates(subset=[station_num_col])
        .rename(columns={station_num_col: "station_number", station_name_col: "station_name"})
    )

    # Merge to get station geo + name
    units = cad_counts.merge(stations_core, on="station_number", how="left")

    # Restrict to units that actually have a station coordinate in the station GeoJSON
    before_geo_filter = len(units)
    units = units[units["station_lon"].notna()].copy()
    after_geo_filter = len(units)
    dropped = before_geo_filter - after_geo_filter
    if dropped > 0:
        print(f"Dropped {dropped} units with no matching station in {stations_path} (non-LEMSA/mutual-aid/etc.)")

    # -----------------------------
    # 3) Tag unit_zone and unit_area from station position
    # -----------------------------
    if units.empty:
        print("⚠️ No units remaining after geo filter; check station_number matching.")
        units["unit_zone"] = []
        units["unit_area"] = []
    else:
        gdf_u = gpd.GeoDataFrame(
            units,
            geometry=gpd.points_from_xy(units["station_lon"], units["station_lat"]),
            crs="EPSG:4326",
        )

        # ALS/BLS boundaries from config
        als_geom = load_boundary(str(config.ALS_BOUNDARY)) if getattr(config, "ALS_BOUNDARY", None) else None
        bls_geom = load_boundary(str(config.BLS_BOUNDARY)) if getattr(config, "BLS_BOUNDARY", None) else None

        if als_geom is not None:
            in_als = gdf_u.geometry.within(als_geom)
        else:
            in_als = pd.Series(False, index=gdf_u.index)

        if bls_geom is not None:
            in_bls = gdf_u.geometry.within(bls_geom)
        else:
            in_bls = pd.Series(False, index=gdf_u.index)

        in_overlap = in_als & in_bls

        unit_zone = pd.Series("unknown", index=gdf_u.index, dtype="object")
        unit_zone = unit_zone.where(~in_als, other="ALS")
        unit_zone = unit_zone.where(~in_bls, other="BLS")
        unit_zone.loc[in_overlap] = "OVERLAP"

        # Urban/rural for units (reuse call tagging geometry)
        urban_union = _read_union("maps/processed/urban_areas_lancaster.geojson")
        rural_union = _read_union("maps/processed/rural_areas_lancaster.geojson")
        county_union = _read_union("maps/processed/lancaster_county_boundary.geojson")

        unit_area = pd.Series("unknown", index=gdf_u.index, dtype="object")
        if urban_union is not None:
            mask = gdf_u.geometry.within(urban_union)
            unit_area = unit_area.where(~mask, other="urban")
        if rural_union is not None:
            mask = gdf_u.geometry.within(rural_union) & (unit_area == "unknown")
            unit_area = unit_area.where(~mask, other="rural")
        if county_union is not None:
            mask = gdf_u.geometry.within(county_union) & (unit_area == "unknown")
            unit_area = unit_area.where(~mask, other="rural")

        units["unit_zone"] = unit_zone.values
        units["unit_area"] = unit_area.values

    # -----------------------------
    # 4) Critical municipality flags (per unit)
    # -----------------------------
    if critical_municipalities:
        muni_norm = units["municipality_std"].astype(str).str.upper().str.strip()
        units["is_critical_municipality"] = muni_norm.isin(critical_municipalities)
        units["critical_zone_type"] = units["is_critical_municipality"].map(
            lambda x: "municipality" if x else "none"
        )
    else:
        units["is_critical_municipality"] = False
        units["critical_zone_type"] = "none"

    # -----------------------------
    # 5) Save canonical units table as CSV
    # -----------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    units.to_csv(out_path, index=False)

    print(f"\n✅ Wrote {len(units):,} units to {out_path}")
    print("Columns:", units.columns.tolist())

    print("\n📊 Critical municipality breakdown (units):")
    print(units["is_critical_municipality"].value_counts(dropna=False))

    print("\n📊 Sample municipalities for units:")
    print(units["municipality_std"].value_counts(dropna=False).head(10))


if __name__ == "__main__":
    main()
