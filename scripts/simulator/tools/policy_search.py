# Utility to enumerate policy variants (top-down from P5 toggles and bottom-up feature flags).
# Does not execute simulations; generates a CSV of variant definitions and suggested CLI.
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class Variant:
    variant_id: str
    policy: str
    kwargs: Dict[str, object]
    complexity: int
    note: str = ""

    def command(self) -> str:
        if self.kwargs:
            return f"--policy {self.policy} --policy-kwargs \"{self.kwargs}\""
        return f"--policy {self.policy}"


def top_down_variants() -> List[Variant]:
    # toggles in hybrid
    toggle_keys = ["use_coverage", "use_fairness", "use_zone_weights", "use_last_unit_guardrail", "use_fairness_weight_boost"]
    variants: List[Variant] = []
    for bits in itertools.product([False, True], repeat=len(toggle_keys)):
        kwargs = dict(zip(toggle_keys, bits))
        name = "_".join([k.split("_")[1] for k, v in kwargs.items() if v]) or "none"
        variant_id = f"p5_td_{name}"
        complexity = 1 + sum(bits)  # base + toggles on
        variants.append(Variant(variant_id, "p5_hybrid", kwargs, complexity, note="top_down"))
    return variants


def bottom_up_variants(max_variants: int | None = None) -> List[Variant]:
    # feature flags in FeatureFlagPolicy
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
    variants: List[Variant] = []
    # generate all non-empty combinations (exhaustive)
    for r in range(1, len(flags) + 1):
        for combo in itertools.combinations(flags, r):
            kwargs = {f: True for f in combo}
            variant_id = f"ff_{'_'.join([f.split('_')[1] for f in combo])}"
            complexity = 1 + len(combo)
            variants.append(Variant(variant_id, "feature_flag", kwargs, complexity, note="bottom_up"))
            if max_variants is not None and len(variants) >= max_variants:
                return variants
    return variants


def main():
    variants = top_down_variants() + bottom_up_variants()
    out_path = Path("reports/variant_candidates.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant_id", "policy", "kwargs", "complexity", "note", "command"])
        writer.writeheader()
        for v in variants:
            row = asdict(v)
            row["kwargs"] = row["kwargs"] or {}
            row["command"] = v.command()
            writer.writerow(row)
    print(f"Wrote {len(variants)} variants to {out_path}")


if __name__ == "__main__":
    main()
