# ems/config/critical_zones.py
from pathlib import Path
import pandas as pd

def load_critical_municipalities(path: str | Path) -> set[str]:
    df = pd.read_csv(path)
    # Normalize to avoid “LANCASTER ” vs “Lancaster” headaches
    return set(df["municipality"].astype(str).str.strip().str.upper())

CRITICAL_MUNICIPALITIES = load_critical_municipalities(
    Path("tools/critical_zones/critical_zones_municipality.csv")
)
