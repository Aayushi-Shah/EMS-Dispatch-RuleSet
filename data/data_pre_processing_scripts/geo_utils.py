# scripts/pre_processing/geo_utils.py
from __future__ import annotations

import json
import os
from typing import List, Union

from shapely.geometry import Point, shape
from shapely.ops import unary_union


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    # ray casting; ring: [[lon,lat], ...]
    x, y = lon, lat
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def _point_in_polygon(lon: float, lat: float, poly: dict) -> bool:
    # GeoJSON Polygon or MultiPolygon
    if poly["type"] == "Polygon":
        rings = poly["coordinates"]
        if not _point_in_ring(lon, lat, rings[0]):  # outer
            return False
        # holes
        for hole in rings[1:]:
            if _point_in_ring(lon, lat, hole):
                return False
        return True
    elif poly["type"] == "MultiPolygon":
        for pg in poly["coordinates"]:
            outer = pg[0]
            if _point_in_ring(lon, lat, outer):
                ok = True
                for hole in pg[1:]:
                    if _point_in_ring(lon, lat, hole):
                        ok = False
                        break
                if ok:
                    return True
        return False
    else:
        raise ValueError("Unsupported geometry type")


def load_boundary(paths: Union[str, List[str], None]):
    """
    Accepts:
      - None or [] -> return None (no filtering)
      - str -> single GeoJSON path
      - list[str] -> union of all GeoJSON feature geometries
    Returns a unioned shapely geometry, or None.
    """
    if not paths:
        return None
    if isinstance(paths, str):
        paths = [paths]

    geoms = []
    for p in paths:
        if not p:
            continue
        path_str = str(p)
        if not os.path.exists(path_str):
            continue
        with open(path_str, "r") as f:
            gj = json.load(f)
        feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
        for ft in feats:
            geom = ft.get("geometry")
            if geom:
                geoms.append(shape(geom))

    if not geoms:
        return None

    union_geom = unary_union(geoms)
    return union_geom


def in_any_polygon(lon: float, lat: float, geoms) -> bool:
    if not geoms:
        return True
    pt = Point(lon, lat)
    if isinstance(geoms, (list, tuple)):
        return any(g and g.contains(pt) for g in geoms)
    return geoms.contains(pt)


def load_zones(zone_files: dict) -> dict:
    """
    zone_files = {
        "ALS": "reference/lemsa_als_boundary.geojson",
        "BLS": "reference/lemsa_bls_boundary.geojson",
        "OVERLAP": "reference/lemsa_overlap_boundary.geojson"
    }
    Returns dict of shapely polygons (may be None).
    """
    zones = {}
    for name, path in zone_files.items():
        try:
            with open(path, "r") as f:
                gj = json.load(f)
            zones[name] = shape(gj["features"][0]["geometry"])
        except Exception:
            zones[name] = None
    return zones


def zone_lookup_factory(zones: dict):
    """Return (lon,lat) → zone_name lookup."""
    def lookup(lon, lat):
        if lon is None or lat is None:
            return None
        p = Point(lon, lat)
        for name, poly in zones.items():
            if poly and poly.contains(p):
                return name
        return None
    return lookup
