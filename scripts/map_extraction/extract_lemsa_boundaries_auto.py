#!/usr/bin/env python3
"""
extract_lemsa_boundaries_auto.py (merged: extraction + georeferencing)
- Extracts polygons from ALS/BLS PDF maps using fill color detection
- Writes page-space GeoJSONs (no CRS)
- Automatically georeferences to WGS84 using unit labels as GCPs
- Exports ALS/BLS/overlap boundaries in both PDF and WGS84 coordinates
"""

from pathlib import Path
from collections import Counter
import re
import math
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.affinity import affine_transform
from shapely.ops import unary_union

# ------------ config ------------
PDFS = {
    "ALS": Path("maps/alsMap.pdf"),
    "BLS": Path("maps/blsMap.pdf"),
}
OUT_DIR = Path("reference/maps")
OUT_ALS = OUT_DIR / "lemsa_als_boundary.geojson"
OUT_BLS = OUT_DIR / "lemsa_bls_boundary.geojson"
OUT_OVL = OUT_DIR / "lemsa_overlap_boundary.geojson"

# WGS84 output paths
OUT_ALS_WGS84 = OUT_DIR / "lemsa_als_boundary_wgs84.geojson"
OUT_BLS_WGS84 = OUT_DIR / "lemsa_bls_boundary_wgs84.geojson"
OUT_OVL_WGS84 = OUT_DIR / "lemsa_overlap_boundary_wgs84.geojson"

STATIONS_CSV = Path("reference/lemsa_stations.csv")

# Polygon extraction parameters
COLOR_TOL = 0.04         # color tolerance in 0..1 space
MIN_AREA = 3_000         # drop tiny slivers in page units
MAX_HOLES_RATIO = 0.98   # discard polygons that are almost all holes

# Georeferencing parameters
MIN_GCPS = 7             # minimum GCPs needed
RANSAC_ITERS = 1500      # RANSAC iterations
INLIER_THRESH_M = 300.0  # inlier threshold in meters
REFIT_WITH_HUBER = True  # use Huber weighting for robustness

UNIT_HEAD = {"AMB", "AMBULANCE", "MEDIC", "MICU", "ALS", "BLS"}
STATION_RE = re.compile(r"^0*(\d{1,2})-(\d+)$")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ Polygon Extraction Functions ============

def norm_color(c):
    if not c: return None
    return tuple(round(max(0, min(1, v)), 3) for v in c)

def close_color(a, b, tol=COLOR_TOL):
    return all(abs(x - y) <= tol for x, y in zip(a, b))

def dominant_nonwhite(colors: Counter):
    for c, _ in colors.most_common():
        if c != (1.0, 1.0, 1.0):
            return c
    return colors.most_common(1)[0][0]

def color_stats_cdrawings(pdf_path: Path) -> Counter:
    doc = fitz.open(pdf_path)
    cnt = Counter()
    for page in doc:
        drawings = page.get_cdrawings()
        if drawings is None:
            continue
        for d in drawings:
            fill = norm_color(d.get("fill"))
            if fill is not None:
                cnt[fill] += 1
    print(f"\n🎨 Fill colors in {pdf_path.name}:")
    for c, n in cnt.most_common():
        print(f"  {c} — {n} occurrences")
    return cnt

def extract_polygons(pdf_path: Path, target_color):
    """
    Extract polygons directly from drawings with matching fill color.
    Each drawing's path is traced in order to form a polygon.
    """
    def ptxy(p):
        # p can be fitz.Point or (x, y) tuple
        try:
            return (p.x, p.y)
        except AttributeError:
            return (float(p[0]), float(p[1]))

    doc = fitz.open(pdf_path)
    polygons = []
    ops_seen = Counter()
    matched_drawings = 0

    for page in doc:
        drawings = page.get_cdrawings()
        if drawings is None:
            continue
        for d in drawings:
            fill = norm_color(d.get("fill"))
            if fill is None or not close_color(fill, target_color):
                continue
            matched_drawings += 1
            
            # Collect points from this drawing in order
            # Line segments in a drawing form a continuous path
            points = []
            for it in d.get("items", []):
                op = it[0]
                ops_seen[op] += 1

                if op == "l":  # ('l', p1, p2) - line segment
                    p1, p2 = ptxy(it[1]), ptxy(it[2])
                    # First segment: add both points
                    if not points:
                        points.append(p1)
                        points.append(p2)
                    else:
                        # Subsequent segments: check if p1 connects to last point
                        last = points[-1]
                        dist = ((p1[0] - last[0])**2 + (p1[1] - last[1])**2)**0.5
                        if dist < 1e-6:
                            # p1 connects to last point, just add p2
                            points.append(p2)
                        else:
                            # Disconnected segment, add both points
                            points.append(p1)
                            points.append(p2)

                elif op == "re":  # ('re', x, y, w, h) OR ('re', (x,y,w,h)) - rectangle
                    # Normalize to numbers
                    if isinstance(it[1], (tuple, list)) and len(it[1]) >= 4:
                        x, y, w, h = it[1][:4]
                    else:
                        x, y, w, h = it[1], it[2], it[3], it[4]
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    # Rectangle forms a closed polygon - replace points
                    points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)]

            # Try to form polygon from collected points
            if len(points) >= 3:
                # Remove duplicate consecutive points
                cleaned_points = [points[0]]
                for p in points[1:]:
                    last = cleaned_points[-1]
                    if abs(p[0] - last[0]) > 1e-6 or abs(p[1] - last[1]) > 1e-6:
                        cleaned_points.append(p)
                
                # Ensure closed (first == last)
                if len(cleaned_points) >= 3:
                    if (abs(cleaned_points[0][0] - cleaned_points[-1][0]) > 1e-6 or
                        abs(cleaned_points[0][1] - cleaned_points[-1][1]) > 1e-6):
                        cleaned_points.append(cleaned_points[0])
                    
                    try:
                        poly = Polygon(cleaned_points)
                        # Fix invalid polygons with buffer(0)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if poly.is_valid and not poly.is_empty and poly.area >= MIN_AREA:
                            # Check hole ratio
                            if isinstance(poly, Polygon):
                                outer = poly.area
                                holes = sum(Polygon(r).area for r in poly.interiors)
                                if outer > 0 and (holes / outer) <= MAX_HOLES_RATIO:
                                    polygons.append(poly)
                    except Exception:
                        # Skip invalid polygons
                        continue

    print(f"✅ {pdf_path.name}: matched drawings={matched_drawings}, polygons={len(polygons)}")
    if ops_seen:
        print("   ops seen:", dict(ops_seen))
    return polygons

def write_geojson(polys, out_path: Path, label: str):
    if not polys:
        gpd.GeoDataFrame({"coverage": []}, geometry=[]).to_file(out_path, driver="GeoJSON")
        print(f"  → wrote {out_path} (empty)")
        return 0
    gdf = gpd.GeoDataFrame({"coverage": [label]*len(polys)}, geometry=polys, crs=None)
    # dissolve to one
    gdf = gdf.dissolve(by="coverage")
    gdf.to_file(out_path, driver="GeoJSON")
    area = float(gdf.geometry.iloc[0].area)
    print(f"  → wrote {out_path}  (area≈{int(area)})")
    return area

def process_extraction(label: str, pdf_path: Path):
    stats = color_stats_cdrawings(pdf_path)
    tgt = dominant_nonwhite(stats)
    print(f"🎯 Auto-selected {label} color: {tgt}")
    polys = extract_polygons(pdf_path, tgt)
    
    # Union all polygons from matching drawings
    if polys:
        union_poly = unary_union(polys)
        if isinstance(union_poly, Polygon):
            return [union_poly]
        elif isinstance(union_poly, MultiPolygon):
            return list(union_poly.geoms)
        else:
            return [union_poly]
    return []

# ============ Georeferencing Functions ============

def canon_station(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().upper().replace(" ", "")
    m = STATION_RE.match(s)
    if not m: return ""
    a, b = m.group(1), m.group(2).lstrip("0") or "0"
    return f"{int(a)}-{int(b)}"

def words_on_page(pdf_path: Path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    for (x0, y0, x1, y1, txt, b, l, w) in page.get_text("words"):
        yield (x0, y0, x1, y1, txt.strip(), b, l, w)

def collect_unit_gcps(pdf_path: Path, stations_df: pd.DataFrame):
    words = list(words_on_page(pdf_path))
    words.sort(key=lambda t: (t[5], t[6], t[7]))  # block,line,word

    hits = []
    for i, (x0, y0, x1, y1, txt, b, l, w) in enumerate(words):
        if txt.upper() in UNIT_HEAD and i + 1 < len(words):
            x0b, y0b, x1b, y1b, nxt, *_ = words[i + 1]
            m = STATION_RE.match(nxt)
            if m:
                st = f"{int(m.group(1))}-{int(m.group(2))}"
                X0, Y0, X1, Y1 = min(x0, x0b), min(y0, y0b), max(x1, x1b), max(y1, y1b)
                cx, cy = (X0 + X1) / 2.0, (Y0 + Y1) / 2.0
                hits.append((st, cx, cy, f"{txt} {nxt}"))

    if not hits:
        return pd.DataFrame(columns=["station_number","page_x","page_y","lon","lat","label"])

    df = pd.DataFrame(hits, columns=["station_number","page_x","page_y","label"])
    df["station_number_norm"] = df["station_number"].apply(canon_station)

    stations_df = stations_df.copy()
    stations_df["station_number_norm"] = stations_df["station_number"].apply(canon_station)

    merged = df.merge(stations_df, on="station_number_norm", how="left", suffixes=("", "_st"))
    merged = merged[merged["lon"].notna() & merged["lat"].notna()].copy()
    merged = merged.sort_values(["station_number_norm"]).drop_duplicates("station_number_norm", keep="first")

    return merged[["station_number_norm","page_x","page_y","lon","lat","label"]].rename(
        columns={"station_number_norm":"station_number"}
    )

def affine_from_gcps(gcp_df: pd.DataFrame):
    X = np.column_stack([gcp_df["page_x"].values, gcp_df["page_y"].values, np.ones(len(gcp_df))])
    lon = gcp_df["lon"].values
    lat = gcp_df["lat"].values
    a, b, tx = np.linalg.lstsq(X, lon, rcond=None)[0]
    d, e, ty = np.linalg.lstsq(X, lat, rcond=None)[0]
    return [a, b, d, e, tx, ty]

def huber_weights(res, delta=1.5):
    # res in meters; weight in [0,1]
    r = np.asarray(res)
    w = np.ones_like(r, dtype=float)
    big = np.abs(r) > delta
    w[big] = delta / np.abs(r[big])
    return w

def refine_affine_weighted(gcp_df: pd.DataFrame, params, use_huber=True):
    # weighted least-squares 1–2 iterations
    cur = params
    for _ in range(2 if use_huber else 1):
        # predict & residuals in meters
        rlon, rlat, rm = residuals_m(gcp_df, cur)
        if not use_huber:
            break
        # Huber on meter residual magnitude
        w = huber_weights(rm, delta=INLIER_THRESH_M / 2.5)  # tighter than inlier threshold
        # build weighted design
        X = np.column_stack([gcp_df["page_x"], gcp_df["page_y"], np.ones(len(gcp_df))]).astype(float)
        W = np.diag(w)
        a,b,tx = np.linalg.lstsq(W @ X, W @ gcp_df["lon"].values, rcond=None)[0]
        d,e,ty = np.linalg.lstsq(W @ X, W @ gcp_df["lat"].values, rcond=None)[0]
        cur = [a,b,d,e,tx,ty]
    return cur

def residuals_deg(gcp_df: pd.DataFrame, params):
    a,b,d,e,tx,ty = params
    lon_pred = a * gcp_df["page_x"].values + b * gcp_df["page_y"].values + tx
    lat_pred = d * gcp_df["page_x"].values + e * gcp_df["page_y"].values + ty
    return lon_pred - gcp_df["lon"].values, lat_pred - gcp_df["lat"].values

def residuals_m(gcp_df: pd.DataFrame, params):
    dlon, dlat = residuals_deg(gcp_df, params)
    lat0 = float(np.mean(gcp_df["lat"].values))
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    dx = dlon * m_per_deg_lon
    dy = dlat * m_per_deg_lat
    rm = np.sqrt(dx*dx + dy*dy)
    return dx, dy, rm

def ransac_affine(gcp_df: pd.DataFrame):
    if len(gcp_df) < MIN_GCPS:
        return affine_from_gcps(gcp_df), gcp_df.index.tolist(), []

    best_params, best_inliers = None, []
    idx_all = gcp_df.index.tolist()
    idx_arr = np.array(idx_all)

    # RANSAC loop
    for _ in range(RANSAC_ITERS):
        # sample minimal set: 3 non-collinear points
        sample_idx = np.random.choice(idx_arr, size=3, replace=False)
        sub = gcp_df.loc[sample_idx]
        try:
            params = affine_from_gcps(sub)
        except Exception:
            continue
        _, _, rm = residuals_m(gcp_df, params)
        inliers = idx_arr[rm <= INLIER_THRESH_M].tolist()
        if len(inliers) > len(best_inliers):
            best_inliers, best_params = inliers, params

    # Refit on inliers (and optionally apply Huber weighting)
    if best_inliers:
        final = affine_from_gcps(gcp_df.loc[best_inliers])
        if REFIT_WITH_HUBER:
            final = refine_affine_weighted(gcp_df.loc[best_inliers], final, use_huber=True)
        return final, best_inliers, [i for i in idx_all if i not in best_inliers]
    else:
        # fallback
        return affine_from_gcps(gcp_df), idx_all, []

def xform_gdf(gdf: gpd.GeoDataFrame, params):
    g = gdf.copy()
    g.crs = None  # page space
    g["geometry"] = g["geometry"].apply(lambda geom: affine_transform(geom, params))
    g.set_crs("EPSG:4326", inplace=True)
    return g

def safe_dissolve(gdf, label):
    geom = unary_union(gdf.geometry)
    return gpd.GeoDataFrame({"coverage":[label]}, geometry=[geom], crs=gdf.crs)

def georeference_map(label: str, pdf_path: Path, boundary_in: Path, out_path: Path, stations_df: pd.DataFrame):
    print(f"\n=== {label} Georeferencing ===")
    if not pdf_path.exists():
        print(f"⚠️  Missing PDF: {pdf_path}")
        return None
    if not boundary_in.exists():
        print(f"⚠️  Missing boundary file: {boundary_in}")
        return None

    gcp = collect_unit_gcps(pdf_path, stations_df)
    print(f"Found {len(gcp)} GCPs on {label} map")
    if len(gcp) < MIN_GCPS:
        print(f"❌ Need at least {MIN_GCPS} GCPs; found {len(gcp)}.")
        return None

    params_raw = affine_from_gcps(gcp)
    _, _, rm_raw = residuals_m(gcp, params_raw)
    rms_raw = float(np.sqrt(np.mean(rm_raw**2)))

    params, inliers, outliers = ransac_affine(gcp)
    dx, dy, rm = residuals_m(gcp.loc[inliers], params)
    rms = float(np.sqrt(np.mean(rm**2)))

    # diagnostics
    diag = gcp.copy()
    diag["dx_m"], diag["dy_m"], diag["resid_m"] = residuals_m(gcp, params)
    diag["inlier"] = diag.index.isin(inliers)
    diag_path = Path("reports") / f"lemsa_gcp_diagnostics_{label.lower()}.csv"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(diag_path, index=False)

    print(f"RMS before RANSAC: ~{rms_raw:.1f} m   |   after: ~{rms:.1f} m  (inliers {len(inliers)}/{len(gcp)})")
    if outliers:
        worst = diag.sort_values("resid_m", ascending=False).head(5)[["label","station_number","resid_m"]]
        print("Top outliers (m):")
        for _, r in worst.iterrows():
            print(f"  {r['label']:<12} {r['station_number']:<6}  {r['resid_m']:.1f}")

    # transform boundaries
    gdf = gpd.read_file(boundary_in)
    if gdf.empty:
        print(f"⚠️  Empty boundary file: {boundary_in}")
        return None
    gdf_wgs = xform_gdf(gdf, params)
    gdf_wgs = safe_dissolve(gdf_wgs, label)
    gdf_wgs.to_file(out_path, driver="GeoJSON")
    print(f"✅ {label} boundary → {out_path}")
    return gdf_wgs

# ============ Main Function ============

def main():
    # Step 1: Extract polygons from PDFs
    print("=" * 60)
    print("STEP 1: Extracting polygons from PDFs")
    print("=" * 60)
    
    als_polys = process_extraction("ALS", PDFS["ALS"])
    bls_polys = process_extraction("BLS", PDFS["BLS"])

    a_area = write_geojson(als_polys, OUT_ALS, "ALS")
    b_area = write_geojson(bls_polys, OUT_BLS, "BLS")

    # quick page-space overlap for sanity
    if als_polys and bls_polys:
        inter = unary_union(als_polys).intersection(unary_union(bls_polys))
        gpd.GeoDataFrame(geometry=[] if inter.is_empty else [inter]).to_file(OUT_OVL, driver="GeoJSON")
        print(f"  → wrote {OUT_OVL} (empty={inter.is_empty})")

    print("\n📊 Summary (page-space):")
    print(f"  ALS polys: {len(als_polys)}  area≈{int(a_area or 0)}")
    print(f"  BLS polys: {len(bls_polys)}  area≈{int(b_area or 0)}")

    # Step 2: Georeference to WGS84 (if stations CSV exists)
    if STATIONS_CSV.exists():
        print("\n" + "=" * 60)
        print("STEP 2: Georeferencing to WGS84")
        print("=" * 60)
        
        # load stations
        st = pd.read_csv(STATIONS_CSV)
        cols = {c.lower(): c for c in st.columns}
        if "lon" not in cols and "longitude" in cols: st["lon"] = st[cols["longitude"]]
        if "lat" not in cols and "latitude"  in cols: st["lat"] = st[cols["latitude"]]
        need = {"station_number","lon","lat"}
        if need - set(c.lower() for c in st.columns):
            print(f"⚠️  Stations CSV missing required columns: {need}")
            print("   Skipping georeferencing.")
        else:
            als_w = georeference_map("ALS", PDFS["ALS"], OUT_ALS, OUT_ALS_WGS84, st)
            bls_w = georeference_map("BLS", PDFS["BLS"], OUT_BLS, OUT_BLS_WGS84, st)

            if als_w is not None and bls_w is not None:
                overlap = gpd.overlay(als_w, bls_w, how="intersection", keep_geom_type=True)
                if overlap.empty:
                    print("⚠️  No ALS∩BLS overlap after georef.")
                else:
                    overlap["coverage"] = "BOTH"
                    overlap.to_file(OUT_OVL_WGS84, driver="GeoJSON")
                    print(f"✅ BOTH overlap → {OUT_OVL_WGS84}")

                both = gpd.GeoDataFrame(pd.concat([als_w, bls_w], ignore_index=True), crs="EPSG:4326")
                print("\n📊 WGS84 Bounds (lon_min, lat_min, lon_max, lat_max):")
                print(f"   {both.total_bounds}")
    else:
        print(f"\n⚠️  Stations CSV not found: {STATIONS_CSV}")
        print("   Skipping georeferencing. PDF-coordinate boundaries saved.")

if __name__ == "__main__":
    np.random.seed(42)  # deterministic RANSAC
    main()
