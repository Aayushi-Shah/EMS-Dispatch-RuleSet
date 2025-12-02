"""
Build variant_metrics.csv from:
  - variant_candidates.csv (variant_id, policy, kwargs, complexity, ...)
  - existing run folders under reports/runs/<run_id>/

For each variant:
  1) Find the run whose meta.json has matching policy + policy_kwargs.
  2) Read summary.csv for that run.
  3) Extract basic KPIs and write a row to variant_metrics.csv.

Usage:
  python scripts/analysis/build_variant_metrics.py \
    --variants data/variant_candidates.csv \
    --runs-root reports/runs \
    --out data/variant_metrics.csv
"""

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def parse_kwargs_cell(cell: Any) -> Dict[str, Any]:
    """
    variant_candidates.kwargs is usually a string like:
      "{'use_severity': True, 'use_als_guard': True}"
    Robustly parse to a dict. Empty or NaN -> {}.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}
    if isinstance(cell, dict):
        return cell
    s = str(cell).strip()
    if not s:
        return {}
    try:
        # ast.literal_eval is safer than eval
        val = ast.literal_eval(s)
        if isinstance(val, dict):
            return val
        return {}
    except Exception:
        return {}


def normalize_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize policy kwargs for stable comparison:
      - sort keys
      - convert True/False/None consistently
      - convert any lists/tuples into tuples
    """
    out = {}
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, (list, tuple)):
            v = tuple(v)
        out[k] = v
    return out


def kwargs_to_key(d: Dict[str, Any]) -> str:
    """
    Turn kwargs dict into a stable string key for matching.
    """
    d_norm = normalize_kwargs(d)
    return json.dumps(d_norm, sort_keys=True, separators=(",", ":"))


def load_variant_table(variants_path: str) -> pd.DataFrame:
    df = pd.read_csv(variants_path)
    if "variant_id" not in df.columns or "policy" not in df.columns:
        raise ValueError("variant_candidates.csv must have at least 'variant_id' and 'policy' columns")

    # Parse and normalize kwargs into a canonical key
    df["kwargs_dict"] = df["kwargs"].apply(parse_kwargs_cell) if "kwargs" in df.columns else [{}] * len(df)
    df["kwargs_norm"] = df["kwargs_dict"].apply(normalize_kwargs)
    df["kwargs_key"] = df["kwargs_dict"].apply(kwargs_to_key)
    df["policy_norm"] = df["policy"].astype(str).str.strip().str.lower()
    df["variant_key"] = (
        df["variant_id"].astype(str).str.strip()
        + "::"
        + df["policy_norm"]
        + "::"
        + df["kwargs_key"]
    )
    return df


def index_runs(runs_root: str) -> pd.DataFrame:
    """
    Scan reports/runs/*/meta.json and build a table:
      run_id, policy, policy_kwargs_key, meta_path, summary_path, variant_id
    """
    rows = []
    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(f"runs_root {runs_root} does not exist")

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        summary_path = run_dir / "summary.csv"
        if not meta_path.exists() or not summary_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue

        cfg = meta.get("config") or {}

        policy_name = (
            meta.get("policy_name")
            or meta.get("policy")
            or cfg.get("POLICY_NAME")
            or cfg.get("policy")
            or ""
        )
        kw = meta.get("policy_kwargs") or meta.get("kwargs") or cfg.get("POLICY_KWARGS") or {}

        # Some runners store kwargs as string; parse if needed
        if isinstance(kw, str):
            kw = parse_kwargs_cell(kw)

        kw_norm = normalize_kwargs(kw)
        kw_key = kwargs_to_key(kw_norm)
        variant_id = meta.get("variant_id") or cfg.get("VARIANT_ID") or run_dir.name.split("_")[0]

        rows.append(
            {
                "run_id": run_dir.name,
                "variant_id": variant_id,
                "policy": policy_name,
                "kwargs_norm": kw_norm,
                "policy_kwargs_key": kw_key,
                "meta_path": str(meta_path),
                "summary_path": str(summary_path),
            }
        )

    if not rows:
        raise RuntimeError(f"No usable runs found under {runs_root}")

    return pd.DataFrame(rows)


def match_variant_to_run(
    variants_df: pd.DataFrame,
    runs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join variants to runs on (policy, kwargs_key).
    If multiple runs match a variant, keep the one with the latest run_id
    (or you can change to best KPI later).
    """
    v = variants_df.copy()
    r = runs_df.copy()

    # Normalize policy strings
    if "policy_norm" not in v.columns:
        v["policy_norm"] = v["policy"].astype(str).str.strip().str.lower()
    r["policy_norm"] = r["policy"].astype(str).str.strip().str.lower()
    if "kwargs_norm" not in v.columns:
        v["kwargs_norm"] = v["kwargs_dict"].apply(normalize_kwargs)
    if "variant_key" not in v.columns:
        v["variant_key"] = (
            v["variant_id"].astype(str).str.strip()
            + "::"
            + v["policy_norm"]
            + "::"
            + v["kwargs_key"]
        )

    # Primary match: policy + kwargs
    merged = v.merge(
        r,
        left_on=["policy_norm", "kwargs_key"],
        right_on=["policy_norm", "policy_kwargs_key"],
        how="left",
        suffixes=("", "_run"),
    )

    # Fallback 1: subset kwargs match (candidate kwargs must be contained in run kwargs)
    unmatched_mask = merged["run_id"].isna()
    if unmatched_mask.any():
        runs_by_policy = {}
        for _, rrow in r.iterrows():
            runs_by_policy.setdefault(rrow["policy_norm"], []).append(rrow)

        for idx, vrow in merged.loc[unmatched_mask].iterrows():
            cand_kw = vrow.get("kwargs_norm") or {}
            policy_key = vrow.get("policy_norm")
            candidates = runs_by_policy.get(policy_key, [])
            matches = []
            for rrow in candidates:
                r_kw = rrow.get("kwargs_norm") or {}
                if all(k in r_kw and r_kw[k] == v for k, v in cand_kw.items()):
                    matches.append(rrow)
            if not matches:
                continue
            # pick latest run_id lexicographically
            best = sorted(matches, key=lambda rr: rr["run_id"])[-1]
            merged.loc[idx, ["run_id", "policy", "policy_kwargs_key", "meta_path", "summary_path"]] = [
                best["run_id"],
                best["policy"],
                best["policy_kwargs_key"],
                best["meta_path"],
                best["summary_path"],
            ]

    # Fallback 2: match by variant_id if still unmatched
    unmatched_mask = merged["run_id"].isna()
    if unmatched_mask.any():
        subset = merged.loc[unmatched_mask].copy()
        subset["variant_id_norm"] = subset["variant_id"].astype(str).str.strip()

        r_variant = r.copy()
        r_variant["variant_id_norm"] = r_variant["variant_id"].astype(str).str.strip()
        r_variant.sort_values("run_id", inplace=True)
        r_variant = r_variant.drop_duplicates(subset=["variant_id_norm"], keep="last")

        fallback = subset.merge(
            r_variant,
            on="variant_id_norm",
            how="left",
            suffixes=("", "_fallback"),
        )
        for merged_idx, row in zip(subset.index, fallback.itertuples(index=False)):
            if pd.isna(row.run_id_fallback):
                continue
            merged.loc[merged_idx, ["run_id", "policy", "policy_kwargs_key", "meta_path", "summary_path"]] = [
                row.run_id_fallback,
                row.policy_fallback,
                row.policy_kwargs_key_fallback,
                row.meta_path_fallback,
                row.summary_path_fallback,
            ]

    # If some variants have multiple runs, keep last by run_id (lexicographically) per variant_key
    merged.sort_values(["variant_key", "run_id"], inplace=True)
    merged = merged.drop_duplicates(subset=["variant_key"], keep="last")

    # Sanity: warn about unmatched variants
    unmatched = merged[merged["run_id"].isna()]
    if len(unmatched) > 0:
        print("WARNING: the following variants had no matching run:")
        print(unmatched[["variant_id", "policy", "kwargs"]])

    return merged


def extract_kpis_for_run(summary_path: str) -> Dict[str, Any]:
    """
    Extract KPIs from a run's summary.csv.

    We don't assume exact columns, but we preferentially pick:
      - total_calls
      - w_avg_p50_resp_min
      - w_avg_p90_resp_min
      - w_avg_avg_resp_min
      - plus any other numeric columns (prefixed with 'kpi_')
    """
    df = pd.read_csv(summary_path)
    if df.empty:
        return {}

    row = df.iloc[0]

    kpis: Dict[str, Any] = {}

    # Common expected fields
    for col in [
        "total_calls",
        "w_avg_p50_resp_min",
        "w_avg_p90_resp_min",
        "w_avg_avg_resp_min",
        "segments",
        "units",
    ]:
        if col in row.index:
            kpis[col] = row[col]

    # Add any other numeric columns as kpi_<name>
    for col in row.index:
        if col in kpis:
            continue
        val = row[col]
        if isinstance(val, (int, float)) and not pd.isna(val):
            kpis[f"kpi_{col}"] = val

    return kpis


def build_variant_metrics(
    variants_path: str,
    runs_root: str,
    out_path: str,
) -> None:
    print(f"[1/4] Loading variants from {variants_path} ...")
    variants_df = load_variant_table(variants_path)
    print(f"  variants: {len(variants_df)}")

    print(f"[2/4] Indexing runs under {runs_root} ...")
    runs_df = index_runs(runs_root)
    print(f"  runs indexed: {len(runs_df)}")

    print("[3/4] Matching variants to runs ...")
    matched = match_variant_to_run(variants_df, runs_df)
    matched_with_run = matched[~matched["run_id"].isna()]
    print(f"  matched variants: {len(matched_with_run)} / {len(variants_df)}")

    # Build metrics rows
    records = []
    print("[4/4] Extracting KPIs from summary.csv per matched variant ...")
    for _, row in matched_with_run.iterrows():
        variant_id = row["variant_id"]
        run_id = row["run_id"]
        summary_path = row["summary_path"]

        kpis = extract_kpis_for_run(summary_path)
        rec = {
            "variant_id": variant_id,
            "run_id": run_id,
            "policy": row["policy"],
            "kwargs": row["kwargs"],
            "complexity": row.get("complexity", None),
            "note": row.get("note", None),
        }
        rec.update(kpis)
        records.append(rec)

    metrics_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    metrics_df.to_csv(out_path, index=False)
    print(f"\n→ Wrote {len(metrics_df)} rows to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="Path to variant_candidates.csv")
    ap.add_argument("--runs-root", default="reports/runs", help="Root directory with run folders")
    ap.add_argument("--out", default="data/variant_metrics.csv", help="Output CSV path")
    args = ap.parse_args()

    build_variant_metrics(args.variants, args.runs_root, args.out)


if __name__ == "__main__":
    main()
    
