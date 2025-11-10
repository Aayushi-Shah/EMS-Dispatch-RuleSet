# scripts/simulator/geo.py
from __future__ import annotations
import json, os
from typing import List, Optional, Union
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    # ray casting; ring: [[lon,lat], ...]
    x, y = lon, lat
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i+1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1 + 1e-12) + x1):
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
    Returns a FeatureCollection dict with one unioned geometry, or None.
    """
    if not paths:
        return None
    if isinstance(paths, str):
        paths = [paths]

    geoms = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, "r") as f:
            gj = json.load(f)
        feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
        for ft in feats:
            if not ft or "geometry" not in ft or ft["geometry"] is None:
                continue
            geoms.append(shape(ft["geometry"]))

    if not geoms:
        return None

    uni = unary_union(geoms)
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": mapping(uni), "properties": {}}],
    }

def in_any_polygon(lon: float, lat: float, geoms) -> bool:
    if not geoms:
        return True
    for g in geoms:
        if _point_in_polygon(lon, lat, g):
            return True
    return False