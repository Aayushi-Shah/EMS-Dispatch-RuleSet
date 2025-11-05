from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env from project root (where config/ is located)
# If .env doesn't exist, that's fine - defaults will be used
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)
else:
    # Try loading from current directory as fallback (for backwards compatibility)
    load_dotenv(override=False)

def _p(key: str, default: str) -> Path:
    return Path(os.getenv(key, default)).expanduser().resolve()

def _b(key: str, default: bool) -> bool:
    v = os.getenv(key, str(default)).strip().lower()
    return v in ("1","true","yes","y","on")

@dataclass(frozen=True)
class Settings:
    RAW_JSON: Path = _p("RAW_JSON", "./data/raw/all_incidents.json")
    PA_MEDICAL: Path = _p("PA_MEDICAL", "./data/processed/medical_calls_pa_clean.parquet")
    LANCASTER_MEDICAL: Path = _p("LANCASTER_MEDICAL", "./data/processed/medical_calls_lancaster.parquet")
    LANCASTER_MEDICAL_CSV: Path = _p("LANCASTER_MEDICAL_CSV", "./data/processed/medical_calls_lancaster_preview.csv")
    LEMSA_MEDICAL: Path = _p("LEMSA_MEDICAL", "./data/processed/medical_calls_lemsa.parquet")

    LEMSA_STATIONS: Path = _p("LEMSA_STATIONS", "./reference/lemsa_stations.csv")
    LEMSA_UNITS_MANUAL: Path = _p("LEMSA_UNITS_MANUAL", "./reference/lemsa_units_manual.csv")
    LEMSA_BOUNDARY: Path = _p("LEMSA_BOUNDARY", "./reference/lemsa_boundary.geojson")
    
    UNITS_VS_CAD_REPORT: Path = _p("UNITS_VS_CAD_REPORT", "./reports/lemsa_units_manual_vs_cad.txt")
    CALLS_FOR_VERIFICATION: Path = _p("CALLS_FOR_VERIFICATION", "./data/processed/medical_calls_lancaster.parquet")
    STATIONS_CSV: Path       = _p("STATIONS_CSV", "./reference/lemsa_stations.csv")
    MANUAL_UNITS_CSV: Path   = _p("MANUAL_UNITS_CSV", "./reference/lemsa_units_manual.csv")

    USER_AGENT: str = os.getenv("USER_AGENT", "Mozilla/5.0")
    HTTP_TIMEOUT: int = int(os.getenv("HTTP_TIMEOUT", "30"))
    ALLOW_EXTERNAL: bool = _b("ALLOW_EXTERNAL", True)
    HTTP_FALLBACK: bool = _b("HTTP_FALLBACK", True)
    RESPECT_ROBOTS: bool = _b("RESPECT_ROBOTS", True)

    GEOCODE_ENABLE: bool = _b("GEOCODE_ENABLE", False)
    GEOCODE_EMAIL: str = os.getenv("GEOCODE_EMAIL", "")
    GEOCODE_RATE_PER_MIN: int = int(os.getenv("GEOCODE_RATE_PER_MIN", "45"))

    TIMEZONE: str = os.getenv("TIMEZONE", "America/New_York")
    FORCE_UTC: bool = _b("FORCE_UTC", True)

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
