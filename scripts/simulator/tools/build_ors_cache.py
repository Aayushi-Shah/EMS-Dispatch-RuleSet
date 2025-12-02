from __future__ import annotations

"""
Prefetch OpenRouteService travel times for common origin/destination pairs.

This builds/updates the ORS cache used by scripts.simulator.traffic.
It loads:
  - Station coordinates from config.STATIONS_CSV
  - Hospital coordinate from config
  - Call origins from CALLS_PARQUET (sampled/quantized to reduce size)

It then queries ORS (matrix API) for:
  - station -> call
  - call -> hospital

Results are stored in .cache/ors_matrix_cache.json by default.
Requires ORS_API_KEY env var (or config.ORS_API_KEY) and ORS_USE can remain False
in config; the cache is used regardless.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib import request

import pandas as pd

from scripts.simulator import config
from scripts.simulator.traffic import _key_for_coords


def load_stations() -> List[Tuple[float, float]]:
    st = pd.read_csv(config.STATIONS_CSV)
    cols_lower = {c.lower(): c for c in st.columns}
    if "latitude" in cols_lower:
        st = st.rename(columns={cols_lower["latitude"]: "lat"})
    if "longitude" in cols_lower:
        st = st.rename(columns={cols_lower["longitude"]: "lon"})
    return [(float(r["lon"]), float(r["lat"])) for _, r in st.iterrows()]


def load_calls(sample_n: int | None = None, round_digits: int = 3) -> List[Tuple[float, float]]:
    df = pd.read_parquet(config.CALLS_PARQUET, columns=["longitude", "latitude"])
    if sample_n is not None and sample_n < len(df):
        df = df.sample(sample_n, random_state=42)
    df["lon_r"] = df["longitude"].round(round_digits)
    df["lat_r"] = df["latitude"].round(round_digits)
    uniq = df[["lon_r", "lat_r"]].dropna().drop_duplicates()
    return [(float(r.lon_r), float(r.lat_r)) for r in uniq.itertuples(index=False)]


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


def ors_minutes(orig: Tuple[float, float], dest: Tuple[float, float], api_key: str) -> float | None:
    url = f"{config.ORS_BASE_URL}/{config.ORS_PROFILE}"
    payload = {
        "locations": [
            [orig[0], orig[1]],
            [dest[0], dest[1]],
        ],
        "metrics": ["duration"],
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }
    req = request.Request(url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    durations = obj.get("durations") or obj.get("matrix") or []
    seconds = durations[0][1]
    return float(seconds) / 60.0


def main(sample_calls: int = 200, round_digits: int = 3, sleep_sec: float = 0.2):
    api_key = getattr(config, "ORS_API_KEY", "") or os.getenv("ORS_API_KEY", "")
    if not api_key:
        print("ORS_API_KEY not set; aborting prefetch.")
        return

    cache_path = getattr(config, "ORS_CACHE_PATH", Path(".cache/ors_matrix_cache.json"))
    cache = load_cache(Path(cache_path))

    stations = load_stations()
    calls = load_calls(sample_calls, round_digits)
    hospital = (float(config.HOSPITAL_LON), float(config.HOSPITAL_LAT))

    pairs = []
    for s in stations:
        for c in calls:
            pairs.append((s, c))
    for c in calls:
        pairs.append((c, hospital))

    print(f"Prefetching {len(pairs)} pairs (stations={len(stations)}, calls={len(calls)})")

    done = 0
    for orig, dest in pairs:
        key = _key_for_coords(orig[0], orig[1], dest[0], dest[1])
        if key in cache:
            done += 1
            continue
        try:
            minutes = ors_minutes(orig, dest, api_key)
            cache[key] = minutes
            done += 1
            if done % 50 == 0:
                print(f"Cached {done}/{len(pairs)}")
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"Failed {orig}->{dest}: {e}")
            # Skip on failure; continue
            continue

    save_cache(Path(cache_path), cache)
    print(f"Saved cache with {len(cache)} entries to {cache_path}")


if __name__ == "__main__":
    main()
