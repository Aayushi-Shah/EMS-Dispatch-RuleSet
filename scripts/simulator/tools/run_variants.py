from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


def run_variants(csv_path: Path, max_variants: int | None = None, max_segments: int | None = None) -> None:
    """
    Loop through variant_candidates.csv and run the simulator for each variant.
    """
    csv_path = csv_path.expanduser()
    with csv_path.open() as f:
        reader = list(csv.DictReader(f))

    if max_variants is not None:
        reader = reader[:max_variants]

    for row in reader:
        vid = row["variant_id"]
        policy = row["policy"]
        kwargs_raw = row["kwargs"]
        try:
            kwargs = json.loads(kwargs_raw.replace("'", '"')) if kwargs_raw and kwargs_raw != "{}" else {}
        except Exception:
            kwargs = {}

        cmd = ["python3", "-m", "scripts.simulator.runner", "--policy", policy]
        if kwargs:
            cmd += ["--policy-kwargs", json.dumps(kwargs)]
        if max_segments is not None:
            cmd += ["--max-segments", str(max_segments)]

        print(f"[{vid}] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_variants(Path("reports/variant_candidates.csv"))
