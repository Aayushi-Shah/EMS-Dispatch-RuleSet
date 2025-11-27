#!/usr/bin/env python

"""
Build Lancaster County boundary and Urban/Rural masks from TIGER data.

Inputs (with defaults):
  - Counties:  data/raw/tl_2023_us_county/tl_2023_us_county.shp
  - Urban:     data/raw/tl_2023_us_uac20/tl_2023_us_uac20.shp

Outputs:
  - reference/lancaster_county_boundary.geojson
  - reference/urban_areas_lancaster.geojson
  - reference/rural_area_lancaster.geojson
"""

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union


STATEFP_PA = "42"
COUNTYFP_LANCASTER = "071"

DEFAULT_COUNTY_SHP = "data/raw/tl_2023_us_county/tl_2023_us_county.shp"
DEFAULT_URBAN_SHP = "data/raw/tl_2023_us_uac20/tl_2023_us_uac20.shp"

DEFAULT_OUT_BOUNDARY = "reference/lancaster_county_boundary.geojson"
DEFAULT_OUT_URBAN = "reference/urban_areas_lancaster.geojson"
DEFAULT_OUT_RURAL = "reference/rural_area_lancaster.geojson"


def build_lancaster_boundary(county_shp: str, out_boundary: Path) -> gpd.GeoDataFrame:
    print("[1/5] Loading US counties …")
    counties = gpd.read_file(county_shp)

    if "STATEFP" not in counties.columns or "COUNTYFP" not in counties.columns:
        raise RuntimeError(
            f"Unexpected schema in {county_shp}: missing STATEFP/COUNTYFP columns. "
            f"Columns present: {list(counties.columns)}"
        )

    print("[2/5] Filtering to Lancaster County, PA (STATEFP=42, COUNTYFP=071) …")
    lanc = counties[
        (counties["STATEFP"] == STATEFP_PA) &
        (counties["COUNTYFP"] == COUNTYFP_LANCASTER)
    ]

    if len(lanc) == 0:
        raise RuntimeError("Lancaster County not found in county shapefile.")
    if len(lanc) > 1:
        print(f"  Found {len(lanc)} features; dissolving into a single boundary.")
        lanc = lanc.dissolve()

    out_boundary.parent.mkdir(parents=True, exist_ok=True)
    lanc.to_file(out_boundary, driver="GeoJSON")

    print(f"[3/5] Wrote Lancaster County boundary → {out_boundary}")
    print(f"       Feature count: {len(lanc)}; CRS: {lanc.crs}")
    return lanc


def clip_urban_to_lancaster(
    urban_shp: str,
    lanc_gdf: gpd.GeoDataFrame,
    out_urban: Path,
    out_rural: Path,
) -> None:
    print("[4/5] Loading TIGER Urban Areas …")
    urban = gpd.read_file(urban_shp)

    # Align CRS
    if urban.crs != lanc_gdf.crs:
        print(f"  Reprojecting urban areas from {urban.crs} to {lanc_gdf.crs}")
        urban = urban.to_crs(lanc_gdf.crs)

    lanc_geom = lanc_gdf.geometry.unary_union

    print("[4.1/5] Clipping urban areas to Lancaster boundary …")
    urban_clip = gpd.clip(urban, lanc_geom)

    # Drop empties just in case
    urban_clip = urban_clip[~urban_clip.geometry.is_empty & urban_clip.geometry.notnull()]

    out_urban.parent.mkdir(parents=True, exist_ok=True)
    urban_clip.to_file(out_urban, driver="GeoJSON")
    print(f"  Wrote clipped urban areas → {out_urban}")
    print(f"  Urban polygons in Lancaster: {len(urban_clip)}")

    # Optional: print a small summary of names if present
    name_cols = [c for c in urban_clip.columns if c.lower().startswith("name")]
    if name_cols:
        nc = name_cols[0]
        print(f"  Urban area names ({nc}):")
        print(urban_clip[nc].value_counts().head(10))

    # Build rural = county boundary minus union(urban_clip)
    print("[4.2/5] Deriving rural remainder (Lancaster minus urban union) …")
    if len(urban_clip) > 0:
        urban_union = unary_union(urban_clip.geometry)
        rural_geom = lanc_geom.difference(urban_union)
    else:
        print("  No urban polygons intersecting Lancaster; rural = full county.")
        rural_geom = lanc_geom

    rural_gdf = gpd.GeoDataFrame(
        {"name": ["Lancaster rural"], "geometry": [rural_geom]},
        crs=lanc_gdf.crs,
    )
    rural_gdf.to_file(out_rural, driver="GeoJSON")
    print(f"  Wrote rural area geometry → {out_rural}")
    print(f"  Rural geometry type: {rural_geom.geom_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Lancaster County boundary and urban/rural masks from TIGER data."
    )
    parser.add_argument("--county-shp", default=DEFAULT_COUNTY_SHP,
                        help=f"US counties TIGER shapefile (default: {DEFAULT_COUNTY_SHP})")
    parser.add_argument("--urban-shp", default=DEFAULT_URBAN_SHP,
                        help=f"TIGER Urban Areas shapefile (default: {DEFAULT_URBAN_SHP})")
    parser.add_argument("--out-boundary", default=DEFAULT_OUT_BOUNDARY,
                        help=f"Output Lancaster boundary GeoJSON (default: {DEFAULT_OUT_BOUNDARY})")
    parser.add_argument("--out-urban", default=DEFAULT_OUT_URBAN,
                        help=f"Output clipped urban areas GeoJSON (default: {DEFAULT_OUT_URBAN})")
    parser.add_argument("--out-rural", default=DEFAULT_OUT_RURAL,
                        help=f"Output rural remainder GeoJSON (default: {DEFAULT_OUT_RURAL})")

    args = parser.parse_args()

    out_boundary = Path(args.out_boundary)
    out_urban = Path(args.out_urban)
    out_rural = Path(args.out_rural)

    lanc = build_lancaster_boundary(args.county_shp, out_boundary)
    clip_urban_to_lancaster(args.urban_shp, lanc, out_urban, out_rural)

    print("[5/5] All done.")


if __name__ == "__main__":
    main()