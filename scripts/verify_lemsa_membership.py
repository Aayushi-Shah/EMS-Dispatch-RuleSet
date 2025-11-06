#!/usr/bin/env python3
# scripts/qa_verify_lemsa_membership.py (buffer-aware)
from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ALS_GEOJSON = Path("reference/maps/lemsa_als_boundary_wgs84.geojson")
BLS_GEOJSON = Path("reference/maps/lemsa_bls_boundary_wgs84.geojson")
CALLS_PARQUET = Path("data/processed/medical_calls_lancaster.parquet")

OUT_DIR = Path("reports"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_OUT = OUT_DIR / "lemsa_point_audit_sample.csv"
BORDER_OUT = OUT_DIR / "lemsa_false_negatives_near_boundary.csv"

BORDER_BAND_M = 1500           # for triage report
BUFFER_FOR_INSIDE_M = 2000     # <= try 1500/2000/2500 and compare

LEMSA_UNIT = re.compile(r"\b(AMB|MICU|MEDIC)?\s*0?6-\d+\b|\b(AMB|MICU|MEDIC)?\s*56-\d+\b", re.IGNORECASE)

def load_union():
    gdf_als = gpd.read_file(ALS_GEOJSON)
    gdf_bls = gpd.read_file(BLS_GEOJSON)
    if gdf_als.crs is None: gdf_als.set_crs("EPSG:4326", inplace=True)
    if gdf_bls.crs is None: gdf_bls.set_crs("EPSG:4326", inplace=True)
    union_geom = gpd.GeoSeries([gdf_als.unary_union, gdf_bls.unary_union], crs="EPSG:4326").unary_union
    return gpd.GeoDataFrame({"coverage":["ALS_or_BLS"]}, geometry=[union_geom], crs="EPSG:4326")

def load_calls():
    df = pd.read_parquet(CALLS_PARQUET)
    if {"longitude","latitude"}.issubset(df.columns):
        lon = df["longitude"].astype(float); lat = df["latitude"].astype(float)
    elif "geoLocation" in df.columns:
        arr = df["geoLocation"].apply(lambda v: (float(v[0]), float(v[1])) if isinstance(v,(list,tuple)) and len(v)==2 else (np.nan,np.nan))
        lon = arr.apply(lambda t:t[0]); lat = arr.apply(lambda t:t[1])
    else:
        raise ValueError("No longitude/latitude or geoLocation columns found.")
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None for xy in zip(lon,lat)], crs="EPSG:4326")
    return gdf.dropna(subset=["geometry"])

def main():
    union = load_union()
    calls = load_calls()

    # Raw inside via spatial join
    raw = gpd.sjoin(calls, union, how="left", predicate="within")
    raw["inside_lemsa_raw"] = raw["coverage"].notna()

    # Metric space for distances + buffered membership
    calls_m = calls.to_crs(3857)
    union_m = union.to_crs(3857)
    border_poly = union_m.geometry.iloc[0]
    calls_m["dist_m_to_boundary"] = calls_m.geometry.distance(border_poly)

    # Buffered "inside"
    buffered_poly = border_poly.buffer(BUFFER_FOR_INSIDE_M)
    calls_m["inside_lemsa_buffered"] = calls_m.geometry.within(buffered_poly)

    # Bring flags/distances back
    raw = raw.join(calls_m[["dist_m_to_boundary","inside_lemsa_buffered"]])

    # Unit code heuristic
    def looks_lemsa(s: str) -> bool:
        if not isinstance(s, str): return False
        for tok in re.split(r"[;,/]| {2,}", s.upper()):
            if tok and LEMSA_UNIT.search(tok): return True
        return False
    raw["unit_looks_lemsa"] = raw.get("unitsString","").apply(looks_lemsa)

    # Crosstabs
    print("\n=== Inside vs LEMSA-like unit code (RAW) ===")
    print(pd.crosstab(raw["unit_looks_lemsa"], raw["inside_lemsa_raw"]))

    print(f"\n(Recheck with {BUFFER_FOR_INSIDE_M} m buffer)")
    print(pd.crosstab(raw["unit_looks_lemsa"], raw["inside_lemsa_buffered"]))

    # Sample CSV for eyeballing
    rng = np.random.default_rng(42)
    sample_in  = raw[raw["inside_lemsa_buffered"]].sample(n=min(50, raw["inside_lemsa_buffered"].sum()), random_state=42)
    sample_out = raw[~raw["inside_lemsa_buffered"]].sample(n=min(50, (~raw["inside_lemsa_buffered"]).sum()), random_state=42)
    sample = pd.concat([sample_in, sample_out])
    if {"longitude","latitude"}.issubset(raw.columns):
        sample["google_maps"] = sample.apply(lambda r: f"https://maps.google.com/?q={r['latitude']},{r['longitude']}", axis=1)
    else:
        sample["google_maps"] = ""
    keep = ["incidentID","description","unitsString","longitude","latitude","inside_lemsa_raw","inside_lemsa_buffered","dist_m_to_boundary","google_maps"]
    for c in keep:
        if c not in sample.columns: sample[c] = ""
    sample[keep].to_csv(SAMPLE_OUT, index=False)
    print(f"\n📄 Sample written: {SAMPLE_OUT}")

    # Near-border potential FNs (LEM SA-looking but outside raw; within band)
    band = raw[(~raw["inside_lemsa_raw"]) & (raw["unit_looks_lemsa"]) & (raw["dist_m_to_boundary"] <= BORDER_BAND_M)].copy()
    if {"longitude","latitude"}.issubset(band.columns):
        band["google_maps"] = band.apply(lambda r: f"https://maps.google.com/?q={r['latitude']},{r['longitude']}", axis=1)
    cols = ["incidentID","description","unitsString","longitude","latitude","dist_m_to_boundary","google_maps"]
    for c in cols:
        if c not in band.columns: band[c] = ""
    band[cols].to_csv(BORDER_OUT, index=False)
    print(f"📄 Near-border (≤{BORDER_BAND_M} m) written: {BORDER_OUT}  Count={len(band)}")

    # Summary
    rate_raw = raw["inside_lemsa_raw"].mean()*100
    rate_buf = raw["inside_lemsa_buffered"].mean()*100
    print(f"\nSummary: RAW inside={rate_raw:.1f}%  |  BUFFER({BUFFER_FOR_INSIDE_M} m) inside={rate_buf:.1f}%")
    print("Union bounds (WGS84):", union.total_bounds)

if __name__ == "__main__":
    main()