# utils.py
import math

def haversine_mi(lon1, lat1, lon2, lat2):
    # force scalars to avoid numpy/dict bleed-through
    lon1 = float(lon1); lat1 = float(lat1); lon2 = float(lon2); lat2 = float(lat2)
    R = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))