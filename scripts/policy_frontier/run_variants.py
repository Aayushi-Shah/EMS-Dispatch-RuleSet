from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

# Default locations
VARIANTS_DEFAULT = Path("reference") / "variant_candidates.csv"
RUNS_ROOT_DEFAULT = Path("reports") / "runs" / "variant_runs_2"

def parse_kwargs_cell(cell: Any) -> Dict[str, Any]:
    """Parse kwargs from CSV into a dict."""
    if cell is None:
        return {}
    if isinstance(cell, dict):
        return cell
    s = str(cell).strip()
    if not s or s == "{}":
        return {}
    try:
        val = ast.literal_eval(s)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def canonical_kwargs_json(d: Dict[str, Any]) -> str:
    """Canonical JSON for kwargs: sorted keys, compact separators."""
    d = d or {}
    canon = {k: d[k] for k in sorted(d.keys())}
    return json.dumps(canon, sort_keys=True, separators=(",", ":"))


def make_variant_key(policy: str, kwargs_json: str) -> str:
    """Variant identity key consistent with KPI pipeline."""
    return f"{policy}::{kwargs_json}"


def load_variants(csv_path: Path):
    """Yield variant rows as dicts from variant_candidates.csv."""
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def run_variants(
    csv_path: Path,
    max_variants: int | None = None,
    max_segments: int | None = None,
) -> None:
    """
    Loop through variant_candidates.csv and run the simulator for each variant.

    - Computes canonical variant_key = policy :: canonical(kwargs_json).
    - Sets VARIANT_KEY / VARIANT_ID / RUN_ID_PREFIX / RUNS_ROOT in env.
    - Uses 'command' column from CSV to build the simulator command.
    """
    count = 0
    for row in load_variants(csv_path):
        if max_variants is not None and count >= max_variants:
            break

        vid = row.get("variant_id", "").strip()
        policy = row.get("policy", "").strip()
        kwargs_str = row.get("kwargs", "") or "{}"
        complexity = row.get("complexity", "")

        if not policy:
            print(f"⚠️  Skipping row with empty policy (variant_id={vid!r})")
            continue

        # Canonical kwargs + variant_key
        kwargs_dict = parse_kwargs_cell(kwargs_str)
        kwargs_json = canonical_kwargs_json(kwargs_dict)
        variant_key = make_variant_key(policy, kwargs_json)

        # Build simulator command from 'command' column
        cmd_str = row.get("command", "").strip()
        if not cmd_str:
            # fallback: build simple command from policy/kwargs if needed
            parts = [f"--policy {policy}"]
            if kwargs_json != "{}":
                parts.append(f'--policy-kwargs "{kwargs_json}"')
            cmd_str = " ".join(parts)

        # Base command: simulator entrypoint
        base_cmd = [sys.executable, "-m", "scripts.simulator.runner"]
        cmd = base_cmd + shlex.split(cmd_str)

        if max_segments is not None:
            cmd += ["--max-segments", str(max_segments)]

        # Env tagging
        env = os.environ.copy()
        env["VARIANT_ID"] = vid
        env["VARIANT_POLICY"] = policy
        env["VARIANT_COMPLEXITY"] = str(complexity or "")
        env["VARIANT_KEY"] = variant_key

        # Use variant_id as a stable, filesystem-safe prefix.
        # The true identity (policy+kwargs) is in VARIANT_KEY inside meta.json.
        run_id_prefix = vid or policy
        env["RUN_ID_PREFIX"] = run_id_prefix

        print(f"▶ Running variant {vid or '[no-id]'}")
        print(f"   policy={policy}, kwargs={kwargs_json}")
        print(f"   variant_key={variant_key}")
        subprocess.run(cmd, check=True, env=env)

        count += 1

    print(f"✅ Completed {count} variants from {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run simulator variants from CSV.")
    parser.add_argument(
        "--variants-csv",
        type=str,
        default=str(VARIANTS_DEFAULT),
        help="Path to variant_candidates.csv (default: reference/variant_candidates.csv)",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Optional cap on number of variants to run",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Optional limit on segments per run for fast passes",
    )
    args = parser.parse_args()

    run_variants(
        csv_path=Path(args.variants_csv),
        max_variants=args.max_variants,
        max_segments=args.max_segments,
    )


if __name__ == "__main__":
    main()
