#!/usr/bin/env python3
from __future__ import annotations

"""
build_lemsa_tagged.py

End-to-end pipeline to build canonical LEMSA calls parquet:

1) Load Lancaster medical calls from parquet.
2) Filter to calls within LEMSA coverage (ALS ∪ BLS).
3) Tag each call with:
   - in_als_boundary / in_bls_boundary / in_overlap_boundary
   - zone ∈ {ALS, BLS, OVERLAP}
   - urban_rural + call_area
   - municipality_std
   - is_critical_municipality + critical_zone_type
   - risk_score from (incidentType, description)
   - severity_bucket ∈ {low, medium, high}
   - preferred_unit_type ∈ {ALS, BLS} from risk_score

Input:
  data/processed/medical_calls_lancaster.parquet (by default)

Output:
  data/processed/medical_calls_lemsa_tagged.parquet
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

# Try to use simulator config ONLY for ALS/BLS boundaries
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import config
    HAVE_CONFIG = True
except Exception:
    HAVE_CONFIG = False

# Default paths
LANCASTER_IN_DEFAULT = Path("data/processed/medical_calls_lancaster.parquet")
ALS_BOUNDARY_DEFAULT = Path("maps/processed/lemsa_als_boundary_wgs84.geojson")
BLS_BOUNDARY_DEFAULT = Path("maps/processed/lemsa_bls_boundary_wgs84.geojson")
LEMSA_OUT_DEFAULT = Path("data/processed/medical_calls_lemsa_tagged.parquet")

# NEW: explicit defaults for urban/rural/county (edit these to your actual files)
URBAN_GEOJSON_DEFAULT = Path("maps/processed/urban_areas_lancaster.geojson")
RURAL_GEOJSON_DEFAULT = Path("maps/processed/rural_areas_lancaster.geojson")
COUNTY_BOUNDARY_DEFAULT = Path("maps/processed/lancaster_county_boundary.geojson")

# NEW: default path for critical municipalities (precomputed CSV)
CRITICAL_MUNICIPALITY_DEFAULT = Path("tools/critical_zones/critical_zones_municipality.csv")

# -----------------------------
# Risk model configuration
# -----------------------------
DESC_BASE_RISK: dict[str, float] = {
    "EMERGENCY TRANSFER-CLASS 1": 0.85,
    "EMERGENCY TRANSFER-CLASS 2": 0.65,
    "EMS ACTIVITY": 0.20,
    "MEDICAL ASSIST-EMERGENCY": 0.75,
    "MEDICAL EMERGENCY": 0.80,
    "RESCUE-COLLAPSE-CONFINED SPACE-TRENCH": 0.95,
    "RESCUE-COLLAPSE-CONFINED SPACE-TRENCH-1A": 1.00,
    "STANDBY-PREARRANGED EMS": 0.10,
    "STANDBY-TRANSFER EMS": 0.10,
    "VEHICLE ACCIDENT-CLASS 1": 0.80,
    "VEHICLE ACCIDENT-CLASS 2": 0.75,
    "VEHICLE ACCIDENT-CLASS 3-EMS ONLY": 0.55,
    "VEHICLE ACCIDENT-COMMERCIAL": 0.70,
    "VEHICLE ACCIDENT-ENTRAPMENT": 0.95,
    "VEHICLE ACCIDENT-FIRE": 0.95,
    "VEHICLE ACCIDENT-HIT RUN-JUST OCC": 0.55,
    "VEHICLE ACCIDENT-MASS TRANSIT": 0.95,
    "VEHICLE ACCIDENT-NO INJURIES": 0.30,
    "VEHICLE ACCIDENT-STANDBY": 0.40,
    "VEHICLE ACCIDENT-TRAIN": 0.95,
    "VEHICLE ACCIDENT-UNKNOWN INJURY": 0.75,
}

INCIDENTTYPE_RISK_ADJ: dict[int, float] = {
    1: 0.10,
    2: 0.00,
    3: -0.10,
}


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
        for cc in df.columns:
            if cc.lower() == c.lower():
                return cc
    raise RuntimeError(f"None of {candidates} found in columns: {df.columns}")


def _read_union(path: Path | str | None):
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


def compute_risk_score(incident_type: Any, description: Any) -> float:
    desc = (description or "").strip().upper()
    base = DESC_BASE_RISK.get(desc, 0.5)

    try:
        itype = int(incident_type)
    except Exception:
        itype = None
    adj = INCIDENTTYPE_RISK_ADJ.get(itype, 0.0)

    score = base + adj
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return float(score)


def risk_to_severity_bucket(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score >= 0.75:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def preferred_unit_type_from_risk(score: float, threshold: float = 0.75) -> str:
    if pd.isna(score):
        return "ALS"
    return "ALS" if score >= threshold else "BLS"


def load_critical_municipality_set(path: Path | str) -> set[str]:
    """
    Load set of municipality names considered "critical" from CSV.
    Expects a 'municipality' column.
    Normalizes to UPPERCASE + stripped.
    """
    p = Path(path)
    if not p.exists():
        print(f"⚠️  Critical municipality file not found at {p}; treating all as non-critical.")
        return set()

    df = pd.read_csv(p)
    if "municipality" not in df.columns:
        raise RuntimeError(
            f"Critical municipality file {p} has no 'municipality' column. "
            f"Columns: {list(df.columns)}"
        )
    crit = (
        df["municipality"]
        .astype(str)
        .str.strip()
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )
    crit_set = set(crit)
    print(f"  Loaded {len(crit_set)} critical municipalities from {p}")
    return crit_set


def main():
    ap = argparse.ArgumentParser(
        description="Build canonical LEMSA calls parquet: filter + geo tags + risk + ALS/BLS preference."
    )
    ap.add_argument(
        "--src",
        type=str,
        default=str(LANCASTER_IN_DEFAULT),
        help="Input Lancaster medical calls parquet",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(LEMSA_OUT_DEFAULT),
        help="Output LEMSA tagged parquet",
    )
    ap.add_argument(
        "--urban-geojson",
        type=str,
        default=str(URBAN_GEOJSON_DEFAULT),
        help="Urban area GeoJSON path",
    )
    ap.add_argument(
        "--rural-geojson",
        type=str,
        default=str(RURAL_GEOJSON_DEFAULT),
        help="Rural area GeoJSON path",
    )
    ap.add_argument(
        "--county-geojson",
        type=str,
        default=str(COUNTY_BOUNDARY_DEFAULT),
        help="County boundary GeoJSON path",
    )
    ap.add_argument(
        "--critical-municipalities",
        type=str,
        default=str(CRITICAL_MUNICIPALITY_DEFAULT),
        help="CSV with precomputed critical municipalities (column: municipality)",
    )
    ap.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Optional limit on number of calls (for quick test)",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.out)

    # -----------------------------
    # Load Lancaster calls
    # -----------------------------
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    print(f"Loading Lancaster calls from {src}...")
    calls = pd.read_parquet(src)
    print(f"  Total Lancaster calls: {len(calls):,}")

    if args.max_calls:
        calls = calls.head(args.max_calls)
        print(f"  Using first {len(calls):,} calls (max-calls).")

    lon_col = pick_col(calls, ["longitude", "lon"])
    lat_col = pick_col(calls, ["latitude", "lat"])

    valid_coords = calls[lon_col].notna() & calls[lat_col].notna()
    calls_valid = calls[valid_coords].copy()
    if len(calls_valid) < len(calls):
        print(f"  Dropped {len(calls) - len(calls_valid):,} calls with missing coordinates")

    gdf = gpd.GeoDataFrame(
        calls_valid,
        geometry=gpd.points_from_xy(calls_valid[lon_col], calls_valid[lat_col]),
        crs="EPSG:4326",
    )

    # -----------------------------
    # ALS/BLS boundaries and filter to LEMSA
    # -----------------------------
    if HAVE_CONFIG and hasattr(config, "ALS_BOUNDARY"):
        als_boundary_path = Path(config.ALS_BOUNDARY)
    else:
        als_boundary_path = ALS_BOUNDARY_DEFAULT

    if HAVE_CONFIG and hasattr(config, "BLS_BOUNDARY"):
        bls_boundary_path = Path(config.BLS_BOUNDARY)
    else:
        bls_boundary_path = BLS_BOUNDARY_DEFAULT

    if not als_boundary_path.exists():
        raise FileNotFoundError(f"ALS boundary not found: {als_boundary_path}")
    if not bls_boundary_path.exists():
        raise FileNotFoundError(f"BLS boundary not found: {bls_boundary_path}")

    print(f"\nLoading LEMSA boundaries...")
    als = gpd.read_file(als_boundary_path)
    bls = gpd.read_file(bls_boundary_path)

    if als.crs is not None and als.crs.to_epsg() != 4326:
        als = als.to_crs("EPSG:4326")
    if bls.crs is not None and bls.crs.to_epsg() != 4326:
        bls = bls.to_crs("EPSG:4326")

    als_geom = als.geometry.unary_union
    bls_geom = bls.geometry.unary_union

    print("  Creating ALS ∪ BLS union boundary...")
    union_geom = unary_union([als_geom, bls_geom])

    print("\nFiltering calls within LEMSA boundaries (ALS ∪ BLS)...")
    in_union = gdf.geometry.within(union_geom)
    gdf = gdf[in_union].copy()
    print(f"  LEMSA calls after union filter: {len(gdf):,}")

    if "incidentID" in gdf.columns:
        before = len(gdf)
        gdf = gdf.drop_duplicates(subset=["incidentID"])
        if len(gdf) < before:
            print(f"  Dropped {before - len(gdf):,} duplicate incidentIDs")
    else:
        gdf = gdf.drop_duplicates()
        print("  No incidentID column; dropped exact duplicate rows")

    # -----------------------------
    # Boundary flags + zone
    # -----------------------------
    print("\nTagging ALS/BLS boundary flags and zone...")
    gdf["in_als_boundary"] = gdf.geometry.within(als_geom)
    gdf["in_bls_boundary"] = gdf.geometry.within(bls_geom)
    gdf["in_overlap_boundary"] = gdf["in_als_boundary"] & gdf["in_bls_boundary"]

    zone = pd.Series(pd.NA, index=gdf.index, dtype="object")
    zone = zone.where(~gdf["in_als_boundary"], other="ALS")
    zone = zone.where(~gdf["in_bls_boundary"], other="BLS")
    overlap_mask = gdf["in_overlap_boundary"]
    zone.loc[overlap_mask] = "OVERLAP"
    gdf["zone"] = zone

    # -----------------------------
    # Urban/rural tagging (NO config – explicit file paths)
    # -----------------------------
    print("\nTagging urban/rural classification...")

    urban_path = Path(args.urban_geojson)
    rural_path = Path(args.rural_geojson)
    county_path = Path(args.county_geojson)

    urban_union = _read_union(urban_path)
    rural_union = _read_union(rural_path)
    county_union = _read_union(county_path)

    print(f"  Urban polygons loaded:  {urban_union is not None} from {urban_path}")
    print(f"  Rural polygons loaded:  {rural_union is not None} from {rural_path}")
    print(f"  County boundary loaded: {county_union is not None} from {county_path}")

    if urban_union is None and rural_union is None and county_union is None:
        raise RuntimeError(
            "No urban/rural/county geometry could be loaded. "
            "Check --urban-geojson, --rural-geojson, and --county-geojson."
        )

    urban_rural = pd.Series("unknown", index=gdf.index, dtype="object")

    if urban_union is not None:
        mask = gdf.geometry.within(urban_union)
        urban_rural = urban_rural.where(~mask, other="urban")

    if rural_union is not None:
        mask = gdf.geometry.within(rural_union) & (urban_rural == "unknown")
        urban_rural = urban_rural.where(~mask, other="rural")

    if county_union is not None:
        mask = gdf.geometry.within(county_union) & (urban_rural == "unknown")
        urban_rural = urban_rural.where(~mask, other="rural")

    gdf["urban_rural"] = urban_rural
    gdf["call_area"] = gdf["urban_rural"]

    # -----------------------------
    # Critical municipality tagging
    # -----------------------------
    print("\nTagging critical municipalities...")

    critical_set = load_critical_municipality_set(Path(args.critical_municipalities))

    try:
        muni_col = pick_col(gdf, ["municipality", "municipality_name", "munic"])
        gdf["municipality_std"] = (
            gdf[muni_col]
            .astype(str)
            .str.strip()
            .str.upper()
        )
    except RuntimeError:
        print("  Municipality column not found; tagging all calls as non-critical.")
        gdf["municipality_std"] = "UNKNOWN"

    gdf["is_critical_municipality"] = gdf["municipality_std"].isin(critical_set)
    gdf["critical_zone_type"] = gdf["is_critical_municipality"].map(
        lambda x: "critical" if x else "non_critical"
    )

    # -----------------------------
    # Risk + severity + ALS/BLS preference
    # -----------------------------
    print("\nDeriving risk, severity, and preferred ALS/BLS from incidentType + description...")

    desc_col = pick_col(gdf, ["description"])
    itype_col = pick_col(gdf, ["incidentType", "incident_type"])

    gdf["risk_score"] = gdf.apply(
        lambda row: compute_risk_score(row[itype_col], row[desc_col]),
        axis=1,
    )
    gdf["severity_bucket"] = gdf["risk_score"].apply(risk_to_severity_bucket)
    gdf["preferred_unit_type"] = gdf["risk_score"].apply(preferred_unit_type_from_risk)

    # -----------------------------
    # Write tagged parquet
    # -----------------------------
    out_df = gdf.drop(columns="geometry")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(dst, index=False)

    print(f"\n✅ LEMSA tagged calls saved: {len(out_df):,}")
    print(f"   Output: {dst}")
    print(f"   Coverage: {len(out_df) / len(calls_valid) * 100:.1f}% of valid Lancaster calls")

    print("\n📊 Severity bucket breakdown:")
    print(out_df["severity_bucket"].value_counts(dropna=False))

    print("\n📊 Urban/rural breakdown:")
    print(out_df["urban_rural"].value_counts(dropna=False))

    print("\n📊 Critical municipality breakdown:")
    print(out_df["is_critical_municipality"].value_counts(dropna=False))
    print("\n📊 Sample of municipalities (standardized):")
    print(out_df["municipality_std"].value_counts().head(10))


if __name__ == "__main__":
    main()
