from __future__ import annotations

import ast
import json
import hashlib
from pathlib import Path
from typing import Any, Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANT_CSV = PROJECT_ROOT / "reference" / "variant_candidates.csv"
REF_DIR = PROJECT_ROOT / "reference"
REF_DIR.mkdir(parents=True, exist_ok=True)

OUT_INDEX_JSON = REF_DIR / "variants_index.json"
OUT_INDEX_CSV = REF_DIR / "variants_index.csv"
OUT_FROZEN_CSV = REF_DIR / "variant_candidates_frozen.csv"


def parse_kwargs(raw: str) -> Dict[str, Any]:
    """
    Safely parse kwargs from the CSV (Python literal style).
    Returns a dict. Empty / invalid → {}.
    """
    if raw is None or str(raw).strip() == "":
        return {}
    try:
        value = ast.literal_eval(str(raw))
    except Exception:
        return {}
    if isinstance(value, dict):
        return value
    return {}


def canonical_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sort kwargs keys for stable serialization.
    """
    return dict(sorted(kwargs.items(), key=lambda kv: kv[0]))


def canonical_key(policy: str, kwargs: Dict[str, Any]) -> str:
    """
    Build a canonical identity for a variant: policy + canonical kwargs.
    """
    canon = canonical_kwargs(kwargs)
    canon_json = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return f"{policy}::{canon_json}"


def freeze_variants(variant_csv: Path = DEFAULT_VARIANT_CSV) -> None:
    if not variant_csv.exists():
        raise FileNotFoundError(f"variant_candidates.csv not found at {variant_csv}")

    print(f"[1/4] Loading variant candidates from {variant_csv} ...")
    df = pd.read_csv(variant_csv)

    required_cols = {"variant_id", "policy", "kwargs"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Clean up whitespace in critical fields
    df["variant_id"] = df["variant_id"].astype(str).str.strip()
    df["policy"] = df["policy"].astype(str).str.strip()

    # Containers
    frozen_records = []          # final unique variants
    seen_canon: Dict[str, str] = {}         # canonical_key -> frozen_id
    id_counts: Dict[str, int] = {}          # base_id -> count for suffixing
    frozen_ids: list[str] = []   # aligned with original df rows

    print("[2/4] Resolving duplicates and freezing variant IDs ...")

    for _, row in df.iterrows():
        base_id = row["variant_id"]
        policy = row["policy"]
        kwargs_raw = row["kwargs"]

        kw = parse_kwargs(kwargs_raw)
        ckey = canonical_key(policy, kw)

        # If we've already seen this exact (policy, kwargs), reuse its frozen id
        if ckey in seen_canon:
            frozen_id = seen_canon[ckey]
            frozen_ids.append(frozen_id)
            continue

        # New canonical variant
        # Ensure the base_id is unique across *different* canonical variants
        if base_id not in id_counts:
            # first time we see this base id
            id_counts[base_id] = 1
            frozen_id = base_id
        else:
            # already used for a different canonical variant; assign suffix
            id_counts[base_id] += 1
            frozen_id = f"{base_id}_{id_counts[base_id]}"

        seen_canon[ckey] = frozen_id
        frozen_ids.append(frozen_id)

        # Compute a small stable hash for reference (optional)
        short_hash = hashlib.sha1(ckey.encode("utf-8")).hexdigest()[:8]

        rec = {
            "frozen_variant_id": frozen_id,
            "canonical_key": ckey,
            "policy": policy,
            "kwargs": kw,
            "kwargs_json": json.dumps(canonical_kwargs(kw), sort_keys=True),
            "complexity": row.get("complexity", None),
            "note": row.get("note", None),
            "command": row.get("command", None),
            "hash8": short_hash,
        }
        frozen_records.append(rec)

    # Align frozen IDs back to original df
    df["frozen_variant_id"] = frozen_ids

    print(f"[3/4] Writing frozen candidates to {OUT_FROZEN_CSV} ...")
    df.to_csv(OUT_FROZEN_CSV, index=False)

    # Unique frozen variants (one row per canonical variant)
    frozen_df = pd.DataFrame(frozen_records).sort_values("frozen_variant_id")

    print(f"[3/4] Writing variants index (CSV) to {OUT_INDEX_CSV} ...")
    frozen_df.to_csv(OUT_INDEX_CSV, index=False)

    # JSON index: map frozen_id -> metadata
    index_dict = {
        rec["frozen_variant_id"]: {
            "policy": rec["policy"],
            "kwargs": rec["kwargs"],
            "complexity": rec["complexity"],
            "note": rec["note"],
            "command": rec["command"],
            "canonical_key": rec["canonical_key"],
            "hash8": rec["hash8"],
        }
        for rec in frozen_records
    }

    print(f"[4/4] Writing variants index (JSON) to {OUT_INDEX_JSON} ...")
    with OUT_INDEX_JSON.open("w", encoding="utf-8") as f:
        json.dump(index_dict, f, indent=2, sort_keys=True)

    print("Done. Frozen variant index created.")
    print(f"  - Frozen candidates: {OUT_FROZEN_CSV}")
    print(f"  - Variants index CSV: {OUT_INDEX_CSV}")
    print(f"  - Variants index JSON: {OUT_INDEX_JSON}")


if __name__ == "__main__":
    freeze_variants()
