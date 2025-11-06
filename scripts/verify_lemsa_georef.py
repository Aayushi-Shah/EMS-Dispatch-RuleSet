#!/usr/bin/env python3
"""
verify_lemsa_georef.py

What this does:
- Loads ALS/BLS/Overlap WGS84 GeoJSONs and validates geometries.
- Computes bounds, area (km²), and perimeter (km) in a local projected CRS.
- Verifies station points: which are inside ALS, inside BLS, inside union, or outside.
- Samples CAD points and reports fraction inside vs outside LEMSA union.
- Exports boundary vertices sample for manual coordinate checks.
- Builds an interactive Folium map (with mouse lat/lon readout) to eyeball alignment.

Outputs (in ./reports):
- lemsa_bounds_areas.txt
- stations_in_out.csv
- cad_inside_outside_summary.txt
- cad_points_sample.csv
- lemsa_boundary_vertices_sample.csv
- lemsa_boundary_qc.html
"""

from pathlib import Path
import os
import random
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import folium
from folium.plugins import MousePosition

# --------------- config ---------------

ALS_GJ = Path("reference/maps/lemsa_als_boundary_wgs84.geojson")
BLS_GJ = Path("reference/maps/lemsa_bls_boundary_wgs84.geojson")
OVL_GJ = Path("reference/maps/lemsa_overlap_boundary_wgs84.geojson")  # may or may not exist

STATIONS_CSV = Path("reference/lemsa_stations.csv")  # needs columns: station_number, lon, lat
CALLS_PARQUET = Path("data/processed/medical_calls_lancaster.parquet")  # your CAD calls subset
CALLS_SAMPLE_N = 3000  # keep the folium map snappy

# Local projected CRS for area/perimeter (Lancaster County ≈ UTM zone 18N)
AREA_CRS = "EPSG:32618"

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------- helpers ---------------

def load_poly(path: Path, label: str) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame({"coverage":[]}, geometry=[], crs="EPSG:4326")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    # fix invalid rings (if any) via buffer(0)
    gdf["geometry"] = gdf["geometry"].buffer(0)
    gdf["coverage"] = label
    return gdf[["coverage", "geometry"]]

def union_nonempty(*frames) -> gpd.GeoDataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return gpd.GeoDataFrame({"coverage":[], "geometry":[]}, crs="EPSG:4326")
    g = pd.concat(frames, ignore_index=True)
    geom = unary_union(g.geometry)
    return gpd.GeoDataFrame({"coverage":["UNION"]}, geometry=[geom], crs=g.crs)

def load_stations(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    # be forgiving with column names
    cols = {c.lower(): c for c in df.columns}
    lon_col = cols.get("lon") or cols.get("longitude")
    lat_col = cols.get("lat") or cols.get("latitude")
    if not lon_col or not lat_col:
        raise ValueError("stations CSV must have lon/lat (or longitude/latitude) columns.")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    return gdf

def load_calls_points(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    df = pd.read_parquet(path)
    # Prefer explicit longitude/latitude columns; else try geoLocation [lon,lat]
    lon = None; lat = None
    for c in df.columns:
        if c.lower() == "longitude": lon = c
        if c.lower() == "latitude":  lat = c
    if lon and lat:
        pts = gpd.points_from_xy(df[lon], df[lat])
    elif "geoLocation" in df.columns:
        # expect [lon, lat]
        def glon(x): 
            try: return float(x[0])
            except: return None
        def glat(x): 
            try: return float(x[1])
            except: return None
        df["_lon"] = df["geoLocation"].apply(glon)
        df["_lat"] = df["geoLocation"].apply(glat)
        df = df.dropna(subset=["_lon","_lat"]).copy()
        pts = gpd.points_from_xy(df["_lon"], df["_lat"])
    else:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(df, geometry=pts, crs="EPSG:4326")

def area_perimeter_km(gdf: gpd.GeoDataFrame) -> tuple[float,float]:
    if gdf.empty:
        return (0.0, 0.0)
    pg = gdf.to_crs(AREA_CRS)
    area_km2 = float(pg.area.sum()) / 1_000_000.0
    perimeter_km = float(pg.length.sum()) / 1_000.0
    return (area_km2, perimeter_km)

def sample_boundary_vertices(gdf: gpd.GeoDataFrame, every_n: int = 100) -> pd.DataFrame:
    if gdf.empty:
        return pd.DataFrame(columns=["lon","lat","coverage"])
    rows = []
    for cov, geom in zip(gdf["coverage"], gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        geoms = [geom] if isinstance(geom, (Polygon,)) else list(geom.geoms)
        for gg in geoms:
            xs, ys = gg.exterior.coords.xy
            for i in range(0, len(xs), every_n):
                rows.append({"lon": xs[i], "lat": ys[i], "coverage": cov})
    return pd.DataFrame(rows)

# --------------- main ---------------

def main():
    # 1) Load boundaries (WGS84)
    als = load_poly(ALS_GJ, "ALS")
    bls = load_poly(BLS_GJ, "BLS")
    ovl = load_poly(OVL_GJ, "BOTH")
    union = union_nonempty(als, bls)

    # 2) Basic metrics & bounds
    with open(REPORTS_DIR / "lemsa_bounds_areas.txt", "w") as f:
        for name, g in [("ALS", als), ("BLS", bls), ("BOTH(overlap)", ovl), ("UNION(ALS∪BLS)", union)]:
            a, p = area_perimeter_km(g)
            b = g.total_bounds if not g.empty else (None, None, None, None)
            line = f"{name:16} area_km2={a:.2f}  perim_km={p:.2f}  bounds(lon_min,lat_min,lon_max,lat_max)={b}\n"
            f.write(line)
            print(line.strip())

    # 3) Stations inside/outside
    try:
        st = load_stations(STATIONS_CSV)
        st["inside_als"] = False if als.empty else st.within(als.unary_union)
        st["inside_bls"] = False if bls.empty else st.within(bls.unary_union)
        st["inside_union"] = False if union.empty else st.within(union.unary_union)
        st_out = st[~st["inside_union"]].copy()
        st.to_csv(REPORTS_DIR / "stations_in_out.csv", index=False)
        print(f"\nStations total={len(st)}, outside union={len(st_out)} → reports/stations_in_out.csv")
    except Exception as e:
        print(f"\n[WARN] Station check skipped: {e}")

    # 4) CAD calls: inside vs outside (sample for speed on map)
    calls = load_calls_points(CALLS_PARQUET)
    inside_count = 0
    if not calls.empty and not union.empty:
        inside = calls.within(union.unary_union)
        inside_count = int(inside.sum())
        summary_txt = (
            f"CAD total={len(calls)}  inside_union={inside_count}  "
            f"outside_union={len(calls)-inside_count}  inside_pct={100*inside_count/len(calls):.1f}%\n"
        )
        (REPORTS_DIR / "cad_inside_outside_summary.txt").write_text(summary_txt, encoding="utf-8")
        print("\n" + summary_txt.strip())

    # 5) Export a small CAD sample for manual checks & map
    calls_sample = calls.sample(min(CALLS_SAMPLE_N, len(calls)), random_state=42) if not calls.empty else calls
    if not calls_sample.empty and not union.empty:
        calls_sample = calls_sample.copy()
        calls_sample["inside_union"] = calls_sample.within(union.unary_union)
        # write CSV with lat/lon for quick eyeballing
        out_csv = REPORTS_DIR / "cad_points_sample.csv"
        # try to serialize geometry to lon/lat
        calls_sample["_lon"] = calls_sample.geometry.x
        calls_sample["_lat"] = calls_sample.geometry.y
        calls_sample[["_lon","_lat","inside_union"]].to_csv(out_csv, index=False)
        print(f"CAD sample → {out_csv}")

    # 6) Export boundary vertices sample for manual lat/lon spot checks
    verts = pd.concat([
        sample_boundary_vertices(als, every_n=75),
        sample_boundary_vertices(bls, every_n=75)
    ], ignore_index=True)
    verts_csv = REPORTS_DIR / "lemsa_boundary_vertices_sample.csv"
    verts.to_csv(verts_csv, index=False)
    print(f"Boundary vertices sample → {verts_csv}")

    # 7) Folium map for visual verification
    if not union.empty:
        # center map roughly on union bounds
        minx, miny, maxx, maxy = union.total_bounds
        ctr = [(miny+maxy)/2.0, (minx+maxx)/2.0]
        m = folium.Map(location=ctr, zoom_start=11, control_scale=True)

        def add_layer(g, color, name, fill_opacity=0.25):
            if g.empty: return
            gj = folium.GeoJson(
                data=g.to_json(),
                name=name,
                style_function=lambda _: {"color": color, "weight": 2, "fillColor": color, "fillOpacity": fill_opacity},
                highlight_function=lambda _: {"weight": 3}
            )
            gj.add_to(m)

        add_layer(als, "#8a2be2", "ALS")    # purple-ish
        add_layer(bls, "#2ecc71", "BLS")    # green-ish
        if not ovl.empty:
            add_layer(ovl, "#f39c12", "Overlap", fill_opacity=0.35)  # orange

        # stations
        try:
            if not st.empty:
                for _, r in st.iterrows():
                    folium.CircleMarker(
                        location=[r.geometry.y, r.geometry.x],
                        radius=4,
                        color="#34495e" if r.get("inside_union", False) else "#e74c3c",
                        fill=True,
                        popup=f"{r.get('station_number','?')}",
                    ).add_to(m)
        except Exception:
            pass

        # small CAD sample (inside vs outside)
        if not calls_sample.empty:
            for _, r in calls_sample.iterrows():
                folium.CircleMarker(
                    location=[r.geometry.y, r.geometry.x],
                    radius=2,
                    color="#2980b9" if r.get("inside_union", False) else "#c0392b",
                    fill=True,
                    opacity=0.7,
                ).add_to(m)

        # mouse lat/lon readout
        MousePosition(
            position="bottomright",
            separator=" | ",
            empty_string="",
            num_digits=6,
            prefix="lat/lon"
        ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        out_html = REPORTS_DIR / "lemsa_boundary_qc.html"
        m.save(out_html)
        print(f"Interactive map → {out_html}")

if __name__ == "__main__":
    main()