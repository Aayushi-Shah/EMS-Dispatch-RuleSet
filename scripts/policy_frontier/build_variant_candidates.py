# Utility to enumerate policy variants (baseline P1–P5, top-down P5 toggles, bottom-up feature flags).
# Does not execute simulations; generates a CSV of unique variant definitions and suggested CLI.
from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class RawVariant:
    family: str              # "baseline", "top_down", "bottom_up"
    policy: str              # policy function/class name as used by the simulator
    kwargs: Dict[str, Any]   # raw kwargs dict
    complexity: int
    slug: str                # short name fragment, e.g. "p1", "cov_fair"
    note: str = ""           # e.g. "baseline", "top_down", "bottom_up"

    def command(self, kwargs_json: str) -> str:
        if kwargs_json and kwargs_json != "{}":
            # kwargs_json is valid JSON; Python side will ast.literal_eval/json.loads it
            return f'--policy {self.policy} --policy-kwargs "{kwargs_json}"'
        return f"--policy {self.policy}"


def canonical_kwargs(kwargs: Dict[str, Any]) -> str:
    """
    Return canonical JSON string with sorted keys for kwargs.
    This becomes the value in the 'kwargs' column and feeds variant_key.
    """
    if kwargs is None:
        kwargs = {}
    canon = {k: kwargs[k] for k in sorted(kwargs.keys())}
    return json.dumps(canon, separators=(",", ":"), sort_keys=True)


def make_variant_key(policy: str, kwargs_json: str) -> str:
    """
    Strict identity key for a policy configuration.
    """
    return f"{policy}::{kwargs_json}"


# Short names for slugs (to avoid collisions and keep IDs interpretable)
TOP_DOWN_TOGGLE_ABBR = {
    "use_coverage": "cov",
    "use_fairness": "fair",
    "use_zone_weights": "zone",
    "use_last_unit_guardrail": "last",
    "use_fairness_weight_boost": "fairw",
}

BOTTOM_UP_FLAG_ABBR = {
    "use_severity": "sev",
    "use_als_guard": "alsguard",
    "use_coverage_unit": "covu",
    "use_coverage_call": "covc",
    "use_guardrail": "guard",
    "use_urban_boost": "urban",
    "use_fairness": "fair",
    "use_fairness_weight": "fairw",
    "use_zone_weights": "zone",
}


def baseline_variants() -> List[RawVariant]:
    """
    Baseline P1–P5 ladder.

    These are taken directly from your previous variant_candidates.csv:
      baseline_p1: policy nearest_eta,     kwargs {}, complexity 1
      baseline_p2: policy p2_als_bls,      kwargs {}, complexity 2
      baseline_p3: policy p3_coverage,     kwargs {}, complexity 3
      baseline_p4: policy p4_fairness,     kwargs {}, complexity 3
      baseline_p5: policy p5_hybrid,       kwargs {}, complexity 4
    """
    BASELINE_DEFS = [
        {"slug": "p1", "policy": "nearest_eta",   "kwargs": {}, "complexity": 1},
        {"slug": "p2", "policy": "p2_als_bls",    "kwargs": {}, "complexity": 2},
        {"slug": "p3", "policy": "p3_coverage",   "kwargs": {}, "complexity": 3},
        {"slug": "p4", "policy": "p4_fairness",   "kwargs": {}, "complexity": 3},
        {"slug": "p5", "policy": "p5_hybrid",     "kwargs": {}, "complexity": 4},
    ]

    variants: List[RawVariant] = []
    for bd in BASELINE_DEFS:
        variants.append(
            RawVariant(
                family="baseline",
                policy=bd["policy"],
                kwargs=bd["kwargs"],
                complexity=bd["complexity"],
                slug=bd["slug"],
                note="baseline",
            )
        )
    return variants


def top_down_variants() -> List[RawVariant]:
    """
    Top-down variants: P5 hybrid with boolean toggles.
    Each combo becomes one RawVariant with a stable slug.
    """
    toggle_keys = [
        "use_coverage",
        "use_fairness",
        "use_zone_weights",
        "use_last_unit_guardrail",
        "use_fairness_weight_boost",
    ]
    variants: List[RawVariant] = []

    for bits in itertools.product([False, True], repeat=len(toggle_keys)):
        kwargs = {k: bool(v) for k, v in zip(toggle_keys, bits) if v}
        # Build slug from abbreviations of enabled toggles, sorted for stability
        if kwargs:
            abbrs = [TOP_DOWN_TOGGLE_ABBR[k] for k in sorted(kwargs.keys())]
            slug = "_".join(abbrs)
        else:
            slug = "none"
        complexity = 1 + sum(bits)  # base + number of toggles on

        variants.append(
            RawVariant(
                family="top_down",
                policy="p5_hybrid",
                kwargs=kwargs,
                complexity=complexity,
                slug=slug,
                note="top_down",
            )
        )
    return variants


def bottom_up_variants(max_variants: int | None = None) -> List[RawVariant]:
    """
    Bottom-up variants: all non-empty combinations of feature flags.
    Each combo becomes one RawVariant with a stable slug.
    """
    flags = [
        "use_severity",
        "use_als_guard",
        "use_coverage_unit",
        "use_coverage_call",
        "use_guardrail",
        "use_urban_boost",
        "use_fairness",
        "use_fairness_weight",
        "use_zone_weights",
    ]
    variants: List[RawVariant] = []

    # Generate all non-empty combinations (exhaustive)
    for r in range(1, len(flags) + 1):
        for combo in itertools.combinations(flags, r):
            kwargs = {f: True for f in combo}
            # Stable slug: sorted abbreviations of flags in the combo
            abbrs = [BOTTOM_UP_FLAG_ABBR[f] for f in sorted(combo)]
            slug = "_".join(abbrs)
            complexity = 1 + len(combo)  # base + number of flags

            variants.append(
                RawVariant(
                    family="bottom_up",
                    policy="feature_flag",
                    kwargs=kwargs,
                    complexity=complexity,
                    slug=slug,
                    note="bottom_up",
                )
            )

            if max_variants is not None and len(variants) >= max_variants:
                return variants

    return variants


def main():
    # 1) Generate conceptual variants (baselines first so they win dedup if collisions exist)
    raw_variants: List[RawVariant] = (
        baseline_variants() + top_down_variants() + bottom_up_variants()
    )

    # 2) Canonicalize kwargs, compute variant_key, deduplicate on strict identity
    rows: List[Dict[str, Any]] = []
    for rv in raw_variants:
        kwargs_json = canonical_kwargs(rv.kwargs)
        variant_key = make_variant_key(rv.policy, kwargs_json)

        rows.append(
            {
                "family": rv.family,
                "policy": rv.policy,
                "kwargs": kwargs_json,          # canonical JSON, still called 'kwargs' for compatibility
                "complexity": rv.complexity,
                "slug": rv.slug,
                "variant_key": variant_key,
                "note": rv.note,
                "command": rv.command(kwargs_json),
            }
        )

    from collections import OrderedDict

    unique_by_key: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    for row in rows:
        key = row["variant_key"]
        if key not in unique_by_key:
            unique_by_key[key] = row

    unique_rows = list(unique_by_key.values())

    # 3) Assign final stable variant_id based on family + slug
    for row in unique_rows:
        fam = row["family"]
        slug = row["slug"]
        if fam == "baseline":
            variant_id = f"baseline_{slug}"      # baseline_p1 ... baseline_p5
        elif fam == "top_down":
            variant_id = f"p5_td_{slug}"
        elif fam == "bottom_up":
            variant_id = f"ff_{slug}"
        else:
            variant_id = f"{fam}_{slug}"
        row["variant_id"] = variant_id

    # 4) Write canonical variant_candidates.csv
    out_path = Path("reference/variant_candidates.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "variant_id",
        "family",
        "slug",
        "policy",
        "kwargs",
        "variant_key",
        "complexity",
        "note",
        "command",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in unique_rows:
            writer.writerow(row)

    print(f"Wrote {len(unique_rows)} unique variants to {out_path}")


if __name__ == "__main__":
    main()