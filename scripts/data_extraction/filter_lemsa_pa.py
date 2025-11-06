#!/usr/bin/env python3
"""
filter_lemsa_pa.py
Filters Lancaster medical calls to only those within LEMSA boundaries (ALS or BLS).
Uses WGS84 georeferenced boundaries to classify calls.
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

# Input/output paths
LANCASTER_IN = Path("data/processed/medical_calls_lancaster.parquet")
ALS_BOUNDARY = Path("reference/maps/lemsa_als_boundary_wgs84.geojson")
BLS_BOUNDARY = Path("reference/maps/lemsa_bls_boundary_wgs84.geojson")
LEMSA_OUT = Path("data/processed/medical_calls_lemsa.parquet")

def main():
    # Check input files
    if not LANCASTER_IN.exists():
        raise FileNotFoundError(f"Input file not found: {LANCASTER_IN}")
    if not ALS_BOUNDARY.exists():
        raise FileNotFoundError(f"ALS boundary not found: {ALS_BOUNDARY}")
    if not BLS_BOUNDARY.exists():
        raise FileNotFoundError(f"BLS boundary not found: {BLS_BOUNDARY}")
    
    # Load Lancaster calls
    print(f"Loading Lancaster calls from {LANCASTER_IN}...")
    calls = pd.read_parquet(LANCASTER_IN)
    print(f"  Total Lancaster calls: {len(calls):,}")
    
    # Check for required columns
    if "longitude" not in calls.columns or "latitude" not in calls.columns:
        raise ValueError("Lancaster data must have 'longitude' and 'latitude' columns")
    
    # Filter out missing coordinates
    valid_coords = calls["longitude"].notna() & calls["latitude"].notna()
    calls_valid = calls[valid_coords].copy()
    if len(calls_valid) < len(calls):
        print(f"  Dropped {len(calls) - len(calls_valid):,} calls with missing coordinates")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        calls_valid,
        geometry=gpd.points_from_xy(calls_valid["longitude"], calls_valid["latitude"]),
        crs="EPSG:4326"
    )
    
    # Load boundaries
    print(f"\nLoading LEMSA boundaries...")
    als = gpd.read_file(ALS_BOUNDARY)
    bls = gpd.read_file(BLS_BOUNDARY)
    
    # Ensure CRS is set
    if als.crs is None:
        als.set_crs("EPSG:4326", inplace=True)
    if bls.crs is None:
        bls.set_crs("EPSG:4326", inplace=True)
    
    # Create union of ALS and BLS (calls in either coverage area)
    print("  Creating ALS ∪ BLS union boundary...")
    union_geom = unary_union([als.geometry.iloc[0], bls.geometry.iloc[0]])
    union_gdf = gpd.GeoDataFrame({"coverage": ["ALS_or_BLS"]}, geometry=[union_geom], crs="EPSG:4326")
    
    # Spatial join to find calls within LEMSA boundaries
    print("\nFiltering calls within LEMSA boundaries...")
    gdf_joined = gpd.sjoin(gdf, union_gdf, how="inner", predicate="within")
    
    # Remove join columns and convert back to DataFrame
    # Use incidentID to drop duplicates (if a call appears in both ALS/BLS, keep one)
    if "incidentID" in gdf_joined.columns:
        lemsa_calls = gdf_joined.drop(columns=["geometry", "coverage", "index_right"]).drop_duplicates(subset=["incidentID"])
    else:
        # Fallback: drop all duplicates
        lemsa_calls = gdf_joined.drop(columns=["geometry", "coverage", "index_right"]).drop_duplicates()
    
    # Save output
    LEMSA_OUT.parent.mkdir(parents=True, exist_ok=True)
    lemsa_calls.to_parquet(LEMSA_OUT, index=False)
    
    # Summary
    print(f"\n✅ LEMSA calls saved: {len(lemsa_calls):,}")
    print(f"   Output: {LEMSA_OUT}")
    print(f"   Coverage: {len(lemsa_calls) / len(calls_valid) * 100:.1f}% of valid Lancaster calls")
    
    # Classification breakdown (ALS/BLS/BOTH)
    if "incidentID" in gdf.columns:
        als_ids = set(gpd.sjoin(gdf, als, how="inner", predicate="within")["incidentID"].unique())
        bls_ids = set(gpd.sjoin(gdf, bls, how="inner", predicate="within")["incidentID"].unique())
        
        als_only_ids = als_ids - bls_ids
        bls_only_ids = bls_ids - als_ids
        both_ids = als_ids & bls_ids
        
        print(f"\n📊 Classification breakdown:")
        print(f"   ALS only: {len(als_only_ids):,}")
        print(f"   BLS only: {len(bls_only_ids):,}")
        print(f"   Both (overlap): {len(both_ids):,}")
        print(f"   Total unique calls: {len(als_ids | bls_ids):,}")

if __name__ == "__main__":
    main()