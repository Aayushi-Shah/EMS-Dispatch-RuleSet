#!/usr/bin/env python3
from __future__ import annotations

"""
analyze_delta_eta.py

Analyze ΔETA from delta-ETA JSONL logs and suggest a K.

Input:
  One or more JSONL files produced by config.delta_eta_logging.log_delta_eta_choice,
  typically in data/experiments/delta_eta/*.jsonl.

For each record where:
  - There are >= 2 candidate units with ETA,
  - A "nearest" unit can be identified,

we compute ΔETA = ETA(alt) - ETA(nearest) for cases where an alternative exists
that helps:
  1) Protect critical municipalities, or
  2) Protect ALS when the call does not require ALS.

We then summarize distributions and suggest K (in minutes) based on percentiles
(50th / 75th / 90th).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

# -----------------------------
# Core logic
# -----------------------------


def iter_records(paths: List[Path]) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a list of JSONL files."""
    for p in paths:
        if not p.exists():
            print(f"⚠️  Skipping missing file: {p}")
            continue
        if p.is_dir():
            # Recurse: all *.jsonl in dir
            for f in sorted(p.glob("*.jsonl")):
                yield from iter_records([f])
            continue

        print(f"Reading {p} ...")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        yield rec
                except Exception:
                    # Skip malformed lines
                    continue


def pick_nearest_candidate(
    rec: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """
    Identify the nearest unit candidate for a record.

    Prefer the candidate whose unit_id == nearest_unit_id.
    If that is missing or doesn't match, fall back to the lowest ETA.
    """
    candidates = rec.get("candidates") or []
    if not candidates:
        return None, None

    nearest_unit_id = rec.get("nearest_unit_id")
    nearest: Optional[Dict[str, Any]] = None
    nearest_eta: Optional[float] = None

    # First pass: try to match nearest_unit_id explicitly
    if nearest_unit_id is not None:
        for c in candidates:
            if c.get("unit_id") == nearest_unit_id and c.get("eta") is not None:
                nearest = c
                nearest_eta = float(c["eta"])
                break

    # Fallback: minimum ETA
    if nearest is None:
        for c in candidates:
            eta = c.get("eta")
            if eta is None:
                continue
            eta = float(eta)
            if nearest_eta is None or eta < nearest_eta:
                nearest_eta = eta
                nearest = c

    return nearest, nearest_eta


def find_best_alternative(
    rec: Dict[str, Any],
    nearest: Dict[str, Any],
    nearest_eta: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[float], bool, bool]:
    """
    Given a record and its nearest candidate, select the "best alternative"
    and return:

      (alt_candidate, alt_eta, is_protect_critical_case, is_protect_als_case)

    Heuristics:
      - Protect critical municipalities:
          nearest is in critical muni, choose alt not in critical muni.
      - Protect ALS:
          nearest.utype == 'ALS' but call_preferred_unit_type != 'ALS',
          choose alt with utype != 'ALS'.

    If multiple alternatives qualify, pick the one with smallest ETA.
    """
    candidates = rec.get("candidates") or []
    if len(candidates) < 2:
        return None, None, False, False

    call_pref = rec.get("call_preferred_unit_type")
    nearest_crit = bool(nearest.get("unit_is_critical_municipality", False))
    nearest_utype = (nearest.get("utype") or "").upper()

    # Define which "modes" this call is relevant for
    is_protect_critical_case = nearest_crit
    is_protect_als_case = nearest_utype == "ALS" and call_pref != "ALS"

    if not (is_protect_critical_case or is_protect_als_case):
        return None, None, False, False

    # Build candidate sets
    alt_candidates: List[Dict[str, Any]] = []

    for c in candidates:
        if c is nearest:
            continue
        eta = c.get("eta")
        if eta is None:
            continue
        eta = float(eta)

        # Skip alternatives that are actually closer than "nearest" due to bad logging
        # (should not happen in a correct nearest-ETA policy, but be defensive).
        if nearest_eta is not None and eta < nearest_eta:
            continue

        c_utype = (c.get("utype") or "").upper()
        c_crit = bool(c.get("unit_is_critical_municipality", False))

        # Critical-protection alternative: nearest is critical, alt is non-critical.
        crit_ok = is_protect_critical_case and not c_crit

        # ALS-protection alternative: nearest is ALS, call doesn't require ALS, alt is non-ALS.
        als_ok = is_protect_als_case and c_utype != "ALS"

        if crit_ok or als_ok:
            alt_candidates.append(c)

    if not alt_candidates:
        return None, None, is_protect_critical_case, is_protect_als_case

    # Pick alt with smallest ETA
    best_alt = None
    best_eta = None
    for c in alt_candidates:
        eta = float(c["eta"])
        if best_eta is None or eta < best_eta:
            best_eta = eta
            best_alt = c

    return best_alt, best_eta, is_protect_critical_case, is_protect_als_case


def summarize(values: List[float]) -> Dict[str, float]:
    """
    Compute basic stats on a list of floats: count, mean, median, p75, p90.
    """
    if not values:
        return {}

    values_sorted = sorted(values)
    n = len(values_sorted)

    def pct(p: float) -> float:
        if n == 1:
            return values_sorted[0]
        # index in [0, n-1]
        idx = int(round(p * (n - 1)))
        return values_sorted[idx]

    mean = sum(values_sorted) / n
    median = pct(0.5)
    p75 = pct(0.75)
    p90 = pct(0.90)

    return {
        "count": float(n),
        "mean": mean,
        "median": median,
        "p75": p75,
        "p90": p90,
    }


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze ΔETA from delta-ETA JSONL logs and suggest K."
    )
    ap.add_argument(
        "paths",
        nargs="+",
        help="JSONL files or directories containing JSONL logs "
             "(e.g. data/experiments/delta_eta/S0_baseline.jsonl).",
    )
    args = ap.parse_args()

    paths = [Path(p) for p in args.paths]

    deltas_all: List[float] = []
    deltas_protect_critical: List[float] = []
    deltas_protect_als: List[float] = []

    total_records = 0
    used_records = 0

    for rec in iter_records(paths):
        total_records += 1

        # Identify nearest
        nearest, nearest_eta = pick_nearest_candidate(rec)
        if nearest is None or nearest_eta is None:
            continue

        # Find best alternative under our heuristics
        alt, alt_eta, is_crit_case, is_als_case = find_best_alternative(
            rec, nearest, nearest_eta
        )

        if alt is None or alt_eta is None:
            continue

        delta = alt_eta - nearest_eta
        if delta < 0:
            # Shouldn't happen, but ignore pathological cases.
            continue

        used_records += 1
        deltas_all.append(delta)
        if is_crit_case:
            deltas_protect_critical.append(delta)
        if is_als_case:
            deltas_protect_als.append(delta)

    print("\n=== ΔETA Analysis ===")
    print(f"Total log records read: {total_records}")
    print(f"Records with usable nearest+alternative: {used_records}")

    def print_stats(label: str, vals: List[float]) -> None:
        print(f"\n--- {label} ---")
        if not vals:
            print("No samples.")
            return
        stats = summarize(vals)
        print(f"Count:   {int(stats['count'])}")
        print(f"Mean:    {stats['mean']:.2f} minutes")
        print(f"Median:  {stats['median']:.2f} minutes")
        print(f"P75:     {stats['p75']:.2f} minutes")
        print(f"P90:     {stats['p90']:.2f} minutes")

    print_stats("All relevant ΔETA (critical + ALS-protection)", deltas_all)
    print_stats("ΔETA for protecting critical municipalities", deltas_protect_critical)
    print_stats("ΔETA for protecting ALS", deltas_protect_als)

    # Suggest K based on union of all relevant deltas
    if deltas_all:
        stats_all = summarize(deltas_all)
        k_median = stats_all["median"]
        k_p75 = stats_all["p75"]

        print("\n=== Suggested K (minutes) ===")
        print(f"- Median-based K ≈ {k_median:.2f}")
        print(f"- 75th percentile K ≈ {k_p75:.2f}")
        print(
            "You can start with K around the 75th percentile "
            "and tighten/loosen it after observing policy impact."
        )
    else:
        print("\nNo ΔETA samples found that match the critical/ALS heuristics.")


if __name__ == "__main__":
    main()