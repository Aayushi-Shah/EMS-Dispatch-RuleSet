# rulelist_policies/config/config.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from tools.critical_zones.critical_zones import load_critical_municipalities

load_dotenv()

# ---------- I/O ----------
CALLS_PARQUET = Path("data/processed/medical_calls_lemsa_tagged.parquet")
UNITS_CSV = Path("data/processed/lemsa_units_from_calls.csv")

# ---------- Field mapping ----------
TIME_COL_CANDIDATES = [
    "t_received",
    "received_ts",
    "opened_ts",
    "call_received_ts",
    "epoch",
    "ts",
    "eventOpened_ts",
]
LON_CANDIDATES = ["lon", "longitude", "call_lon", "x"]
LAT_CANDIDATES = ["lat", "latitude", "call_lat", "y"]

# ---------- Hospital ----------
HOSPITAL_LAT = 40.047144
HOSPITAL_LON = -76.304382

# ---------- Simulation controls ----------
SHIFT_WINDOWS = [(0, 480), (480, 960), (960, 1440)]  # [00-08), [08-16), [16-24)
MAX_QUEUE_RETRIES = 15

# -----------------------------
# Scene / transport speeds
# -----------------------------

# Existing single-speed defaults (keep them for fallback)
SCENE_SPEED_MPH = 45.0
HOSPITAL_SPEED_MPH = 50.0

# Urban/rural variants for more realism
SCENE_SPEED_MPH_URBAN = 35.0
SCENE_SPEED_MPH_RURAL = 55.0
SCENE_SPEED_MPH_DEFAULT = SCENE_SPEED_MPH  # fallback

HOSPITAL_SPEED_MPH_URBAN = 35.0
HOSPITAL_SPEED_MPH_RURAL = 60.0
HOSPITAL_SPEED_MPH_DEFAULT = HOSPITAL_SPEED_MPH  # fallback

# ---------- Travel model settings ----------
ROAD_FACTOR = 1.35
ROAD_FACTOR_BY_SHIFT = {
    0: 1.35,
    1: 1.55,
    2: 1.45,
}

HOUR_CONGESTION = {
    0: 1.00,
    1: 1.00,
    2: 1.00,
    3: 1.00,
    4: 1.00,
    5: 1.05,
    6: 1.10,
    7: 1.20,
    8: 1.25,
    9: 1.15,
    10: 1.10,
    11: 1.05,
    12: 1.05,
    13: 1.05,
    14: 1.10,
    15: 1.15,
    16: 1.20,
    17: 1.25,
    18: 1.20,
    19: 1.15,
    20: 1.10,
    21: 1.05,
    22: 1.00,
    23: 1.00,
}

ZONES = [
    (-999, -999, 999, 999, "default"),
]
ZONE_MULTIPLIER = {
    "default": 1.00,
}

TT_NOISE_SIGMA = 0.05

ORS_USE = False
ORS_API_KEY = os.getenv("ORS_API_KEY")
ORS_PROFILE = "driving-car"
ORS_BASE_URL = "https://api.openrouteservice.org/v2/matrix"
ORS_CACHE_PATH = Path(".cache/ors_matrix_cache.json")

# ---------- Hospital turnaround ----------
TA_HOURLY = {
    0: 1.00,
    1: 1.00,
    2: 1.00,
    3: 1.00,
    4: 1.00,
    5: 1.05,
    6: 1.10,
    7: 1.15,
    8: 1.20,
    9: 1.20,
    10: 1.15,
    11: 1.10,
    12: 1.10,
    13: 1.10,
    14: 1.15,
    15: 1.20,
    16: 1.25,
    17: 1.30,
    18: 1.25,
    19: 1.20,
    20: 1.15,
    21: 1.10,
    22: 1.05,
    23: 1.00,
}
TA_ALPHA = 0.10

# ---------- Transport / non-transport heuristics ----------
USE_NON_TRANSPORT = True
NON_TRANSPORT_BY_SEVERITY = {
    "high": 0.05,
    "medium": 0.15,
    "low": 0.30,
    "unknown": 0.10,
}
NON_TRANSPORT_BASE = 0.10
NON_TRANSPORT_HIGH_PROB = 0.80
NON_TRANSPORT_LOW_PROB = 0.02
NON_TRANSPORT_KEYWORDS = [
    "ems activity",
    "standby-prearranged ems",
    "standby-transfer ems",
    "vehicle accident-no injuries",
    "vehicle accident-standby",
]
TRANSPORT_KEYWORDS = [
    "emergency transfer-class 1",
    "emergency transfer-class 2",
    "rescue-collapse-confined space-trench",
    "rescue-collapse-confined space-trench-1a",
    "vehicle accident-entrapment",
    "vehicle accident-fire",
    "vehicle accident-mass transit",
    "vehicle accident-train",
]
SINGLE_TRANSPORT_PER_CALL = False

# ---------- Critical municipalities ----------
CRITICAL_ZONES_PATH = Path("tools/critical_zones/critical_zones_municipality.csv")
CRITICAL_MUNICIPALITIES = load_critical_municipalities(CRITICAL_ZONES_PATH)

# -----------------------------
# On-scene + turnaround times
# -----------------------------

ONSCENE_MIN = 10.0       # baseline mins on scene
ONSCENE_SCALE = 10.0     # gamma scale (so mean on-scene ≈ ONSCENE_MIN + ONSCENE_SCALE)

TURNAROUND_MIN = 10.0    # nominal hospital turnaround baseline
TURNAROUND_SCALE = 10.0  # gamma scale

# Long hospital delays (overcrowding / offload issues)
LONG_TRANSPORT_MIN_MINUTES = 20.0   # if transport leg ≥ this, eligible for long delay
LONG_TURNAROUND_PROB = 0.3          # chance of long delay (rural / long transports)
LONG_TURNAROUND_EXTRA_MIN = 30.0    # additional mins if long delay triggers
LONG_TURNAROUND_EXTRA_MAX = 90.0

# Congestion tail (rare but painful)
HOSPITAL_CONGESTION_PROB = 0.10          # 10% of transports get extra delay
HOSPITAL_CONGESTION_MAX_EXTRA_MIN = 60.0 # up to +60 min extra

STRESS_RESTRICT_TO_PEAK_WINDOW = True  
PEAK_WINDOW_MINUTES = 180
N_PEAK_WINDOWS = 5            
