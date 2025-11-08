#!/usr/bin/env python3
# Build LEMSA-only dataset + discover units from CAD, config-driven (no CLI).

from pathlib import Path
import re
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

# ---- config (prefers ems_config; falls back to defaults) ----
try:
    from config.ems_config import settings
except Exception:
    class _S: pass
    settings = _S()
    settings.LANCASTER_MEDICAL = Path("data/processed/medical_calls_lancaster.parquet")
    settings.LEMSA_MEDICAL     = Path("data/processed/medical_calls_lemsa.parquet")
    settings.LEMSA_UNITS_MANUAL= Path("reference/lemsa_units_manual_normalized.csv")

ALS_WGS = Path("reference/lemsa_als_boundary_wgs84.geojson")
BLS_WGS = Path("reference/lemsa_bls_boundary_wgs84.geojson")

OUT_CSV = Path(str(settings.LEMSA_MEDICAL).replace(".parquet", ".csv"))
UNITS_FROM_CAD = Path("reference/lemsa_units_from_cad.csv")
UNITS_COMBINED = Path("reference/lemsa_units_combined.csv")

# ---- helpers ----
def load_calls(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_parquet(path)
    # find lon/lat columns robustly
    cols = {c.lower(): c for c in df.columns}
    lon_col = next((cols[k] for k in ["lon","longitude","x","geo_longitude","lng"] if k in cols), None)
    lat_col = next((cols[k] for k in ["lat","latitude","y","geo_latitude"] if k in cols), None)
    if not lon_col or not lat_col:
        raise ValueError("Could not locate lon/lat columns in CAD parquet.")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    return gdf

def load_lemsa_union() -> gpd.GeoSeries:
    g_als = gpd.read_file(ALS_WGS).to_crs(4326) if ALS_WGS.exists() else None
    g_bls = gpd.read_file(BLS_WGS).to_crs(4326) if BLS_WGS.exists() else None
    parts = [g for g in [g_als, g_bls] if g is not None and not g.empty]
    if not parts:
        raise FileNotFoundError("Missing ALS/BLS WGS84 boundaries. Generate them first.")
    geom = unary_union(pd.concat(parts, ignore_index=True).geometry)
    return gpd.GeoSeries([geom], crs="EPSG:4326")

UNIT_TOKEN = re.compile(r"\b(?:(AMBULANCE|AMB|MICU|MEDIC|ALS|BLS)\s*)?(\d{2})-(\d+)\b", re.I)

def norm_unit(s: str) -> str:
    m = UNIT_TOKEN.search(s or "")
    if not m: return ""
    head = (m.group(1) or "").upper()
    if head == "AMBULANCE": head = "AMB"
    st, num = m.group(2), m.group(3)
    return (f"{head} {int(st):02d}-{int(num)}").strip()

# ---- main ----
def main():
    Path(settings.LEMSA_MEDICAL).parent.mkdir(parents=True, exist_ok=True)
    UNITS_FROM_CAD.parent.mkdir(parents=True, exist_ok=True)

    print("• Loading Lancaster medical calls…")
    calls = load_calls(Path(settings.LANCASTER_MEDICAL))
    print(f"  rows: {len(calls):,}")

    print("• Loading LEMSA ALS/BLS boundaries (WGS84)…")
    union = load_lemsa_union().iloc[0]

    print("• Spatial filter → LEMSA-only calls…")
    inside_mask = calls.within(union)
    calls_lemsa = calls[inside_mask].copy()
    print(f"  kept: {len(calls_lemsa):,} ({len(calls_lemsa)/len(calls):.1%})")

    # Save parquet + CSV
    calls_lemsa.drop(columns=["geometry"]).to_parquet(settings.LEMSA_MEDICAL, index=False)
    calls_lemsa.drop(columns=["geometry"]).to_csv(OUT_CSV, index=False)
    print(f"  → {settings.LEMSA_MEDICAL}")
    print(f"  → {OUT_CSV}")

    print("• Discovering units from CAD (inside LEMSA)…")
    units_col = "unitsString" if "unitsString" in calls_lemsa.columns else None
    if not units_col:
        raise ValueError("Expected 'unitsString' column not found in CAD calls.")

    all_units = []
    for s in calls_lemsa[units_col].astype(str):
        for tok in re.split(r"[;,/]| {2,}", s):
            u = norm_unit(tok.strip())
            if u: all_units.append(u)

    seen = (pd.Series(all_units, name="unit_designator")
              .value_counts()
              .rename_axis("unit_designator")
              .reset_index(name="calls_inside"))

    # split prefix for quick scan
    seen["prefix"] = seen["unit_designator"].str.extract(r"(\d{2})-", expand=False)

    seen.to_csv(UNITS_FROM_CAD, index=False)
    print(f"  → {UNITS_FROM_CAD} (rows: {len(seen)})")

    # Merge with manual list (if present)
    combined = seen.copy()
    if Path(settings.LEMSA_UNITS_MANUAL).exists():
        manual = pd.read_csv(settings.LEMSA_UNITS_MANUAL)
        manual["unit_designator"] = manual["unit_designator"].astype(str).str.upper().str.replace(r"\s+"," ", regex=True)
        combined = (manual.merge(seen, on="unit_designator", how="outer")
                           .sort_values(["unit_designator"]))
    combined.to_csv(UNITS_COMBINED, index=False)
    print(f"  → {UNITS_COMBINED} (merged manual + discovered)\n")

    # Tiny report
    top = seen.head(15)
    print("Top 15 units inside LEMSA by call count:")
    for _, r in top.iterrows():
        print(f"  {r['unit_designator']:<12}  {int(r['calls_inside']):>6}")

if __name__ == "__main__":
    main()