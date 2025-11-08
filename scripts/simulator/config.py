# scripts/simulator/config.py
from pathlib import Path

# ---------- I/O ----------
CALLS_PARQUET = Path("data/processed/medical_calls_lemsa.parquet")
STATIONS_CSV  = Path("reference/lemsa_stations.csv")
UNITS_CSV     = Path("reference/lemsa_units_consolidated.csv")

RUNS_DIR   = Path("reports/runs")
RUNLOG_CSV = Path("reports/runlog.csv")

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

SEGMENT_BY_SHIFT = True
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

# Optional bottlenecks with fixed penalties (minutes) if a call lies inside bbox
BOTTLENECKS = [
    # {"bbox": (lon_min, lat_min, lon_max, lat_max), "penalty_min": 1.5}
]

# Small multiplicative noise on computed minutes to avoid ties; set 0 to disable
TT_NOISE_SIGMA = 0.05