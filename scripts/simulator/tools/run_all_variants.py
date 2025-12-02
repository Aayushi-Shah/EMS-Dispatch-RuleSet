from __future__ import annotations

"""
Run baselines (P1–P5) plus all variant candidates.

Usage:
  python -m scripts.simulator.tools.run_all_variants --max-segments 100
  # optionally limit candidates:
  python -m scripts.simulator.tools.run_all_variants --max-variants 50
"""

import argparse
import csv
import json
import tempfile
from pathlib import Path

from scripts.simulator.tools.run_variants import run_variants


BASELINES = [
    {"variant_id": "baseline_p1", "policy": "nearest_eta", "kwargs": "{}", "complexity": "1", "note": "baseline"},
    {"variant_id": "baseline_p2", "policy": "p2_als_bls", "kwargs": "{}", "complexity": "2", "note": "baseline"},
    {"variant_id": "baseline_p3", "policy": "p3_coverage", "kwargs": "{}", "complexity": "3", "note": "baseline"},
    {"variant_id": "baseline_p4", "policy": "p4_fairness", "kwargs": "{}", "complexity": "3", "note": "baseline"},
    {"variant_id": "baseline_p5", "policy": "p5_hybrid", "kwargs": "{}", "complexity": "4", "note": "baseline"},
]


def _load_candidates(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _write_candidates(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _combine_candidates(orig_rows: list[dict], include_baselines: bool) -> list[dict]:
    rows = orig_rows.copy()
    if include_baselines:
        existing = {r["variant_id"] for r in rows}
        for b in BASELINES:
            if b["variant_id"] not in existing:
                rows.append(b)
    return rows


def _load_existing_variants(runs_dir: Path) -> set[str]:
    """Return variant_ids already present in existing run artifacts."""
    variants: set[str] = set()
    if not runs_dir.exists():
        return variants
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        variant_id = meta.get("variant_id") or (meta.get("config") or {}).get("VARIANT_ID")
        if variant_id:
            variants.add(str(variant_id))
    return variants


def main():
    parser = argparse.ArgumentParser(description="Run baselines + candidates, then summarize.")
    parser.add_argument("--variants-csv", type=str, default="reports/variant_candidates.csv")
    parser.add_argument("--max-variants", type=int, default=None, help="Limit number of candidate rows (after baselines).")
    parser.add_argument("--max-segments", type=int, default=None, help="Limit segments per run (pass to runner).")
    parser.add_argument("--include-baselines", action="store_true", default=True, help="Include P1–P5 baselines.")
    parser.add_argument("--start-from", type=str, default=None, help="Variant_id to start from (inclusive) in the candidates list.")
    parser.add_argument("--runs-dir", type=str, default="reports/runs", help="Directory containing prior run outputs.")
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Skip variants that already have a run in --runs-dir (default).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Run variants even if they already exist in --runs-dir.",
    )
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    variants_path = Path(args.variants_csv)
    orig_rows = _load_candidates(variants_path)
    combined_rows = _combine_candidates(orig_rows, include_baselines=args.include_baselines)

    rows_to_run = combined_rows
    if args.start_from:
        started = False
        filtered = []
        for row in rows_to_run:
            if not started:
                if row.get("variant_id") == args.start_from:
                    started = True
                    filtered.append(row)
                continue
            filtered.append(row)
        rows_to_run = filtered
        if not started:
            print(f"[WARN] start-from '{args.start_from}' not found; no variants will run.")

    if args.skip_existing:
        existing_variants = _load_existing_variants(Path(args.runs_dir))
        before_rows = rows_to_run
        before = len(before_rows)
        rows_to_run = [row for row in before_rows if row.get("variant_id") not in existing_variants]
        skipped = before - len(rows_to_run)
        if skipped:
            skipped_ids = sorted({row["variant_id"] for row in before_rows if row["variant_id"] in existing_variants})
            print(f"[INFO] Skipping {skipped} variants already in {args.runs_dir}: {', '.join(skipped_ids)}")

    if not rows_to_run:
        print("No variants to run after filtering; exiting.")
        return

    # Write combined candidates to a temp file so we don't modify the original.
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp_path = Path(tmp.name)
        _write_candidates(rows_to_run, tmp_path)

    # Run all variants (baselines + candidates) with env tagging handled by run_variants.
    run_variants(tmp_path, max_variants=args.max_variants, max_segments=args.max_segments)
    print("Done.")


if __name__ == "__main__":
    main()
