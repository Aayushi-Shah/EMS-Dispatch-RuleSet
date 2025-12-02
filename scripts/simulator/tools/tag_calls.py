from __future__ import annotations

"""Pre-tag calls with boundary/zone/urban-rural metadata using vectorized geospatial ops.

Outputs a new parquet next to CALLS_PARQUET with suffix "_tagged".
"""

import argparse
import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys

# Ensure project root on PYTHONPATH when run as a script
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulator import config
from scripts.simulator.geo import load_boundary
from scripts.simulator.io import ZONES


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
        for cc in df.columns:
            if cc.lower() == c.lower():
                return cc
    raise RuntimeError(f"None of {candidates} found in columns: {df.columns}")


def main():
    parser = argparse.ArgumentParser(description="Pre-tag calls with boundaries/zone/urban_rural")
    parser.add_argument("--max-calls", type=int, default=None, help="Limit number of calls to tag (for quick test)")
    args = parser.parse_args()

    src = Path(config.CALLS_PARQUET)
    df = pd.read_parquet(src)
    if args.max_calls:
        df = df.head(args.max_calls)

    lon_col = pick_col(df, config.LON_CANDIDATES)
    lat_col = pick_col(df, config.LAT_CANDIDATES)

    # Normalize lon/lat to canonical columns.
    df = df.copy()
    df["lon"] = df[lon_col].astype(float)
    df["lat"] = df[lat_col].astype(float)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326"),
    )

    # Vectorized boundary tagging.
    als_geom = load_boundary(str(config.ALS_BOUNDARY))
    bls_geom = load_boundary(str(config.BLS_BOUNDARY))
    overlap_geom = load_boundary(str(config.OVERLAP_BOUNDARY))

    gdf["in_als_boundary"] = gdf.geometry.within(als_geom) if als_geom else False
    gdf["in_bls_boundary"] = gdf.geometry.within(bls_geom) if bls_geom else False
    gdf["in_overlap_boundary"] = gdf.geometry.within(overlap_geom) if overlap_geom else False

    # Vectorized zone tagging with priority ALS -> BLS -> OVERLAP.
    zone = pd.Series(pd.NA, index=gdf.index, dtype="object")
    for name in ("ALS", "BLS", "OVERLAP"):
        poly = ZONES.get(name)
        if poly is None:
            continue
        mask = gdf.geometry.within(poly) & zone.isna()
        zone = zone.where(~mask, other=name)
    gdf["zone"] = zone

    # Vectorized urban/rural tagging: urban > rural > county>unknown.
    def _read_union(path: Path | str | None):
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        try:
            geodf = gpd.read_file(p)
            if geodf.crs is not None and geodf.crs.to_epsg() != 4326:
                geodf = geodf.to_crs("EPSG:4326")
            try:
                return geodf.union_all()
            except AttributeError:
                # Fallback for older geopandas versions
                return geodf.unary_union
        except Exception:
            return None

    urban_union = _read_union(getattr(config, "URBAN_GEOJSON_PATH", None))
    rural_union = _read_union(getattr(config, "RURAL_GEOJSON_PATH", None))
    county_union = _read_union(getattr(config, "LANCASTER_BOUNDARY_PATH", "reference/lancaster_county_boundary.geojson"))

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

    out_df = gdf.drop(columns="geometry")
    suffix = "_tagged" if not args.max_calls else f"_tagged_sample{args.max_calls}"
    dst = src.with_name(f"{src.stem}{suffix}{src.suffix}")
    out_df.to_parquet(dst, index=False)
    print(f"Wrote tagged calls to {dst} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
