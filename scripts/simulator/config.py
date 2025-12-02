# scripts/simulator/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- I/O ----------
CALLS_PARQUET = Path("data/processed/medical_calls_lemsa_tagged.parquet")
STATIONS_CSV  = Path("reference/lemsa_stations.csv")
UNITS_CSV     = Path("reference/lemsa_units_consolidated.csv")
DUTY_CSV      = Path("reference/lemsa_unit_duty.csv")

RUNS_DIR   = Path("reports/runs")
RUNLOG_CSV = Path("reports/runlog.csv")

# Optional boundary files (WGS84).
ALS_BOUNDARY      = Path("reference/maps/lemsa_als_boundary_wgs84.geojson")
BLS_BOUNDARY      = Path("reference/maps/lemsa_bls_boundary_wgs84.geojson")
OVERLAP_BOUNDARY  = Path("reference/maps/lemsa_overlap_boundary_wgs84.geojson")
URBAN_GEOJSON_PATH = Path("reference/maps/urban_areas_lancaster.geojson")
RURAL_GEOJSON_PATH = Path("reference/maps/rural_area_lancaster.geojson")

# ---------- Field mapping ----------
TIME_COL_CANDIDATES = [
    "t_received","received_ts","opened_ts","call_received_ts","epoch","ts","eventOpened_ts"
]
LON_CANDIDATES = ["lon","longitude","call_lon","x"]
LAT_CANDIDATES = ["lat","latitude","call_lat","y"]

# ---------- Hospital ----------
HOSPITAL_LAT = 40.047144
HOSPITAL_LON = -76.304382

# ---------- Simulation controls ----------
RANDOM_SEED = 42

SEGMENT_BY_SHIFT = False
SHIFT_WINDOWS = [(0, 480), (480, 960), (960, 1440)]  # [00-08), [08-16), [16-24)

# Queue semantics
MAX_QUEUE_RETRIES = 15

# ---------- Base speeds and delays ----------
SCENE_SPEED_MPH    = 38.0
HOSPITAL_SPEED_MPH = 40.0
DISPATCH_DELAY_MIN = 1.5

# Scene/hospital dwell distributions
ONSCENE_MIN, ONSCENE_SCALE       = 8.0, 12.0
TURNAROUND_MIN, TURNAROUND_SCALE = 10.0, 15.0

# ---------- Distance inflation (baseline straight-line → road) ----------
# Keep simple global factor for backward compatibility.
ROAD_FACTOR = 1.35

# Shift-specific inflation (heuristic). If used by policies, this overrides ROAD_FACTOR.
ROAD_FACTOR_BY_SHIFT = {
    0: 1.35,  # 00-08
    1: 1.55,  # 08-16
    2: 1.45,  # 16-24
}

# Hour-of-day congestion multiplier (0..23). 1.0 means no change.
HOUR_CONGESTION = {
    0:1.00, 1:1.00, 2:1.00, 3:1.00, 4:1.00, 5:1.05,
    6:1.10, 7:1.20, 8:1.25, 9:1.15,10:1.10,11:1.05,
    12:1.05,13:1.05,14:1.10,15:1.15,16:1.20,17:1.25,
    18:1.20,19:1.15,20:1.10,21:1.05,22:1.00,23:1.00,
}

# Optional spatial zoning (all calls fall into "default" unless you add more zones)
# ZONES entries: (lon_min, lat_min, lon_max, lat_max, name)
ZONES = [
    (-999, -999, 999, 999, "default"),
]
ZONE_MULTIPLIER = {
    "default": 1.00,
    # e.g., "urban": 1.10, "rural": 0.95
}

# Small multiplicative noise on computed minutes to avoid ties; set 0 to disable
TT_NOISE_SIGMA = 0.05

# ---- Optional OpenRouteService integration (road network travel) ----
ORS_USE = False  # set True to attempt ORS calls
ORS_API_KEY = os.getenv("ORS_API_KEY")
ORS_PROFILE = "driving-car"
ORS_BASE_URL = "https://api.openrouteservice.org/v2/matrix"
ORS_CACHE_PATH = Path(".cache/ors_matrix_cache.json")

# ---- Hospital turnaround (heuristic) ----
# Hour-of-day multipliers (0..23). Tweak later when you have real ED data.
TA_HOURLY = {
    0:1.00, 1:1.00, 2:1.00, 3:1.00, 4:1.00, 5:1.05,
    6:1.10, 7:1.15, 8:1.20, 9:1.20,10:1.15,11:1.10,
    12:1.10,13:1.10,14:1.15,15:1.20,16:1.25,17:1.30,
    18:1.25,19:1.20,20:1.15,21:1.10,22:1.05,23:1.00,
}
# Sensitivity to ED load (0..1). Keep small until calibrated.
TA_ALPHA = 0.10
# Lookback window if you later compute rolling ED load (minutes). Not used yet.
TA_LOOKBACK_MIN = 90

# ---------- Transport / non-transport heuristics ----------
USE_NON_TRANSPORT = False
NON_TRANSPORT_BASE = 0.1          # base probability a call does NOT transport
NON_TRANSPORT_HIGH_PROB = 0.8     # if keywords indicate likely non-transport
NON_TRANSPORT_LOW_PROB = 0.0      # if keywords indicate definite transport
NON_TRANSPORT_KEYWORDS = [
    "lift assist", "public assist", "minor injury", "no patient",
    "refused", "cancel", "cancelled", "no transport", "ntp",
    "fall", "ankle", "wrist", "sick person",
]
TRANSPORT_KEYWORDS = [
    "cardiac", "chest pain", "no pulse", "not breathing", "respiratory",
    "gunshot", "stabbing", "overdose", "stroke", "seizure",
]
SINGLE_TRANSPORT_PER_CALL = False  # for multi-unit calls, only one unit transports by default

BOUNDARY_GEOJSONS = [
    "reference/maps/lemsa_als_boundary_wgs84.geojson",
    "reference/maps/lemsa_bls_boundary_wgs84.geojson",
    "reference/maps/lemsa_overlap_boundary_wgs84.geojson",
]
POLICY_NAME = "nearest_eta"  # or "StationBiasETA", "MaxRadiusCap"
POLICY_KWARGS = {"penalty_min": 2.0}  # or {"max_mi": 12.0}
DUTY_ENFORCEMENT = True  # set False to ignore duty windows
