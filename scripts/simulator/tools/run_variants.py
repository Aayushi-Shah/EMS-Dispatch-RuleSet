from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def run_variants(csv_path: Path, max_variants: int | None = None, max_segments: int | None = None) -> None:
    """
    Loop through variant_candidates.csv and run the simulator for each variant.
    Uses the current Python interpreter (sys.executable) so venv deps are honored.
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
        kwargs = {}
        if kwargs_raw and kwargs_raw != "{}":
            try:
                # kwargs are stored as Python dict repr in the CSV; parse safely
                parsed = ast.literal_eval(kwargs_raw)
                if isinstance(parsed, dict):
                    kwargs = parsed
            except Exception:
                kwargs = {}

        cmd = [sys.executable, "-m", "scripts.simulator.runner", "--policy", policy]
        if kwargs:
            cmd += ["--policy-kwargs", json.dumps(kwargs)]
        if max_segments is not None:
            cmd += ["--max-segments", str(max_segments)]

        print(f"[{vid}] running: {' '.join(cmd)}")
        env = os.environ.copy()
        env["VARIANT_ID"] = vid
        env["VARIANT_POLICY"] = policy
        env["VARIANT_COMPLEXITY"] = str(row.get("complexity", "") or "")
        env["RUN_ID_PREFIX"] = vid  # encode variant in run_id for downstream matching
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulator variants from CSV.")
    parser.add_argument("--variants-csv", type=str, default="reports/variant_candidates.csv")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--max-segments", type=int, default=None, help="Limit number of segments per run for fast pass")
    args = parser.parse_args()

    run_variants(Path(args.variants_csv), max_variants=args.max_variants, max_segments=args.max_segments)
