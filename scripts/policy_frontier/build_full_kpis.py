#!/usr/bin/env python3
"""
Compute richer KPIs per variant by matching variants to runs, then reading
decisions.csv (and summary.csv when available).

Inputs:
  - reference/variant_candidates.csv (variant metadata, kwargs, complexity, note)
  - reports/runs/variant_runs/<run_id>/decisions.csv + meta.json (+ summary.csv)

Output:
  - reference/variant_kpis_full.csv

Usage:
  python scripts/policy_frontier/build_full_kpis.py \
    --variants reference/variant_candidates.csv \
    --runs-root reports/runs/variant_runs \
    --out reference/variant_kpis_full.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

VARIANTS_DEFAULT = Path("reference") / "variant_candidates.csv"
RUNS_ROOT_DEFAULT = Path("reports") / "runs" / "variant_runs"
OUT_DEFAULT = Path("reference") / "variant_kpis_full.csv"

# Thresholds / constants
RISK_HIGH_THRESHOLD = 0.75
COVERAGE_LOSS_THRESHOLD = 0.5  # for coverage_loss_above_thresh_rate


# -----------------------------
# Helpers
# -----------------------------
def parse_kwargs_cell(cell: Any) -> Dict[str, Any]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}
    if isinstance(cell, dict):
        return cell
    s = str(cell).strip()
    if not s:
        return {}
    try:
        val = ast.literal_eval(s)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def normalize_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, (list, tuple)):
            v = tuple(v)
        out[k] = v
    return out


def kwargs_to_key(d: Dict[str, Any]) -> str:
    return json.dumps(normalize_kwargs(d), sort_keys=True, separators=(",", ":"))


def safe_debug(df: pd.DataFrame) -> pd.Series:
    """
    Parse the 'debug' column into dicts, or {} if unavailable/unparseable.
    """
    if "debug" not in df.columns:
        return pd.Series([{}] * len(df))

    def parse(cell: Any) -> Dict[str, Any]:
        if isinstance(cell, dict):
            return cell
        s = str(cell).strip()
        if not s:
            return {}
        try:
            val = ast.literal_eval(s)
            return val if isinstance(val, dict) else {}
        except Exception:
            return {}

    return df["debug"].apply(parse)


def percentile(series: pd.Series, q: float) -> float:
    """
    Compute q-th percentile for a numeric Series, returning NaN if empty.
    """
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


# -----------------------------
# Data loading / matching (old, robust logic)
# -----------------------------
def load_variant_table(variants_path: Path) -> pd.DataFrame:
    df = pd.read_csv(variants_path)
    if "variant_id" not in df.columns or "policy" not in df.columns:
        raise ValueError("variant_candidates.csv must have at least 'variant_id' and 'policy' columns")

    # Parse kwargs if present
    if "kwargs" in df.columns:
        df["kwargs_dict"] = df["kwargs"].apply(parse_kwargs_cell)
    else:
        df["kwargs_dict"] = [{}] * len(df)

    df["kwargs_norm"] = df["kwargs_dict"].apply(normalize_kwargs)
    df["kwargs_key"] = df["kwargs_norm"].apply(kwargs_to_key)

    df["policy_norm"] = df["policy"].astype(str).str.strip().str.lower()
    df["variant_key"] = (
        df["variant_id"].astype(str).str.strip()
        + "::"
        + df["policy_norm"]
        + "::"
        + df["kwargs_key"]
    )
    return df


def index_runs(runs_root: Path) -> pd.DataFrame:
    """
    Walk runs_root and index all usable runs with their policy + kwargs.
    This is the same logic you had before, which we know works.
    """
    rows = []
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root {runs_root} does not exist")

    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue

        decisions_path = run_dir / "decisions.csv"
        summary_path = run_dir / "summary.csv"
        meta_path = run_dir / "meta.json"
        if not decisions_path.exists() or not meta_path.exists():
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
        if isinstance(kw, str):
            try:
                kw = ast.literal_eval(kw)
            except Exception:
                kw = {}
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
                "summary_path": str(summary_path) if summary_path.exists() else "",
                "decisions_path": str(decisions_path),
            }
        )

    if not rows:
        raise RuntimeError(f"No usable runs found under {runs_root}")

    return pd.DataFrame(rows)


def match_variant_to_run(variants_df: pd.DataFrame, runs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Original robust matching:
      1) match on policy_norm + kwargs_key
      2) fallback: candidate kwargs ⊆ run kwargs
      3) fallback: match on variant_id
    """
    v = variants_df.copy()
    r = runs_df.copy()

    if "policy_norm" not in v.columns:
        v["policy_norm"] = v["policy"].astype(str).str.strip().str.lower()
    r["policy_norm"] = r["policy"].astype(str).str.strip().str.lower()

    if "kwargs_norm" not in v.columns:
        v["kwargs_dict"] = v["kwargs"].apply(parse_kwargs_cell)
        v["kwargs_norm"] = v["kwargs_dict"].apply(normalize_kwargs)

    if "kwargs_key" not in v.columns:
        v["kwargs_key"] = v["kwargs_norm"].apply(kwargs_to_key)

    if "variant_key" not in v.columns:
        v["variant_key"] = (
            v["variant_id"].astype(str).str.strip()
            + "::"
            + v["policy_norm"]
            + "::"
            + v["kwargs_key"]
        )

    # Primary match: policy + kwargs_key
    merged = v.merge(
        r,
        left_on=["policy_norm", "kwargs_key"],
        right_on=["policy_norm", "policy_kwargs_key"],
        how="left",
        suffixes=("", "_run"),
    )

    # Fallback 1: subset kwargs match (candidate kwargs ⊆ run kwargs)
    unmatched_mask = merged["run_id"].isna()
    if unmatched_mask.any():
        runs_by_policy: Dict[str, List[pd.Series]] = {}
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
            # Take the latest run_id for stability
            best = sorted(matches, key=lambda rr: rr["run_id"])[-1]
            merged.loc[idx, ["run_id", "policy", "policy_kwargs_key", "meta_path", "summary_path", "decisions_path"]] = [
                best["run_id"],
                best["policy"],
                best["policy_kwargs_key"],
                best["meta_path"],
                best["summary_path"],
                best["decisions_path"],
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
            merged.loc[merged_idx, ["run_id", "policy", "policy_kwargs_key", "meta_path", "summary_path", "decisions_path"]] = [
                row.run_id_fallback,
                row.policy_fallback,
                row.policy_kwargs_key_fallback,
                row.meta_path_fallback,
                row.summary_path_fallback,
                row.decisions_path_fallback,
            ]

    merged.sort_values(["variant_key", "run_id"], inplace=True)
    merged = merged.drop_duplicates(subset=["variant_key"], keep="last")

    unmatched = merged[merged["run_id"].isna()]
    if len(unmatched) > 0:
        print("WARNING: the following variants had no matching run:")
        print(unmatched[["variant_id", "policy", "kwargs"]])

    return merged


# -----------------------------
# KPI computation per run (v2)
# -----------------------------
def compute_run_kpis(decisions_path: Path, summary_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Compute v2 KPIs (28 metrics) from one decisions.csv (+ summary.csv if present).
    Returns dict of KPI fields; raises if decisions.csv is unusable.
    """
    df = pd.read_csv(decisions_path)
    if df.empty or "resp_min" not in df.columns:
        raise ValueError(f"{decisions_path} missing resp_min or empty")

    dbg = safe_debug(df)

    # Extract fields from debug dict
    call_area = dbg.apply(lambda d: d.get("call_area"))
    coverage_loss_raw = dbg.apply(lambda d: d.get("coverage_loss"))
    risk_raw = dbg.apply(lambda d: d.get("risk"))
    utype_raw = dbg.apply(lambda d: d.get("utype"))
    u_area = dbg.apply(lambda d: d.get("u_area"))

    # Numeric conversions
    resp = pd.to_numeric(df["resp_min"], errors="coerce")
    cov_loss = pd.to_numeric(coverage_loss_raw, errors="coerce")
    risk = pd.to_numeric(risk_raw, errors="coerce")

    utype = utype_raw.astype(str).str.upper()

    n_calls = int(len(df))

    # Risk slices
    high_risk_mask = risk >= RISK_HIGH_THRESHOLD
    low_risk_mask = (risk < RISK_HIGH_THRESHOLD) & (~risk.isna())
    n_calls_high_risk = int(high_risk_mask.sum())
    n_calls_low_risk = int(low_risk_mask.sum())

    # Response time KPIs
    mean_resp_time = float(resp.mean())
    p50_resp_time = percentile(resp, 50.0)
    p90_resp_time = percentile(resp, 90.0)

    if n_calls_high_risk > 0:
        resp_high = resp[high_risk_mask]
        mean_resp_time_high_risk = float(resp_high.mean())
        p90_resp_time_high_risk = percentile(resp_high, 90.0)
    else:
        mean_resp_time_high_risk = float("nan")
        p90_resp_time_high_risk = float("nan")

    # ALS/BLS behavior
    als_mask = utype == "ALS"
    bls_mask = utype == "BLS"
    als_or_bls_mask = utype.isin(["ALS", "BLS"])

    # High-risk ALS mismatch (underuse)
    denom_highrisk_als_bls_mask = high_risk_mask & als_or_bls_mask
    n_calls_highrisk_als_bls = int(denom_highrisk_als_bls_mask.sum())
    if n_calls_highrisk_als_bls > 0:
        mismatches = int((high_risk_mask & bls_mask & als_or_bls_mask).sum())
        als_mismatch_rate = mismatches / n_calls_highrisk_als_bls
    else:
        als_mismatch_rate = float("nan")

    # Low-risk ALS overuse
    denom_lowrisk_als_bls_mask = low_risk_mask & als_or_bls_mask
    n_calls_lowrisk_als_bls = int(denom_lowrisk_als_bls_mask.sum())
    if n_calls_lowrisk_als_bls > 0:
        overuses = int((low_risk_mask & als_mask & als_or_bls_mask).sum())
        als_overuse_rate = overuses / n_calls_lowrisk_als_bls
    else:
        als_overuse_rate = float("nan")

    # ALS shares
    als_count = int(als_mask.sum())
    if n_calls > 0:
        als_share_of_calls = als_count / n_calls
    else:
        als_share_of_calls = float("nan")

    if n_calls_high_risk > 0:
        als_share_of_calls_high_risk = int(als_mask[high_risk_mask].sum()) / n_calls_high_risk
    else:
        als_share_of_calls_high_risk = float("nan")

    if n_calls_low_risk > 0:
        als_share_of_calls_low_risk = int(als_mask[low_risk_mask].sum()) / n_calls_low_risk
    else:
        als_share_of_calls_low_risk = float("nan")

    # Coverage KPIs
    coverage_loss_mean = float(cov_loss.mean())
    coverage_loss_p90 = percentile(cov_loss, 90.0)

    if n_calls_high_risk > 0:
        cov_high = cov_loss[high_risk_mask]
        coverage_loss_high_risk_mean = float(cov_high.mean())
        coverage_loss_high_risk_p90 = percentile(cov_high, 90.0)
    else:
        coverage_loss_high_risk_mean = float("nan")
        coverage_loss_high_risk_p90 = float("nan")

    # Fraction of calls with coverage_loss above threshold
    if n_calls > 0:
        above_thresh = int((cov_loss > COVERAGE_LOSS_THRESHOLD).sum())
        coverage_loss_above_thresh_rate = above_thresh / n_calls
    else:
        coverage_loss_above_thresh_rate = float("nan")

    # Fairness KPIs (urban vs rural)
    urban_mask = call_area == "urban"
    rural_mask = call_area == "rural"
    n_calls_urban = int(urban_mask.sum())
    n_calls_rural = int(rural_mask.sum())

    # Overall fairness gaps (means / p90)
    if n_calls_urban > 0 and n_calls_rural > 0:
        mean_urban = float(resp[urban_mask].mean())
        mean_rural = float(resp[rural_mask].mean())
        fairness_gap_mean = abs(mean_urban - mean_rural)

        p90_urban = percentile(resp[urban_mask], 90.0)
        p90_rural = percentile(resp[rural_mask], 90.0)
        fairness_gap_p90 = abs(p90_urban - p90_rural)
    else:
        fairness_gap_mean = float("nan")
        fairness_gap_p90 = float("nan")

    # High-risk fairness gaps
    urb_hr_mask = urban_mask & high_risk_mask
    rur_hr_mask = rural_mask & high_risk_mask
    n_urban_hr = int(urb_hr_mask.sum())
    n_rural_hr = int(rur_hr_mask.sum())

    if n_urban_hr > 0 and n_rural_hr > 0:
        mean_urban_hr = float(resp[urb_hr_mask].mean())
        mean_rural_hr = float(resp[rur_hr_mask].mean())
        fairness_gap_high_risk_mean = abs(mean_urban_hr - mean_rural_hr)

        p90_urban_hr = percentile(resp[urb_hr_mask], 90.0)
        p90_rural_hr = percentile(resp[rur_hr_mask], 90.0)
        fairness_gap_high_risk_p90 = abs(p90_urban_hr - p90_rural_hr)
    else:
        fairness_gap_high_risk_mean = float("nan")
        fairness_gap_high_risk_p90 = float("nan")

    # Area mismatch KPIs
    known_u = u_area.isin(["urban", "rural"])
    known_c = call_area.isin(["urban", "rural"])
    both_known = known_u & known_c
    n_calls_area_info = int(both_known.sum())
    if n_calls_area_info > 0:
        mismatched = int((both_known & (u_area != call_area)).sum())
        unit_area_mismatch_rate = mismatched / n_calls_area_info
    else:
        unit_area_mismatch_rate = float("nan")

    # Build KPI dict (28 metrics: 20 values + 8 counts)
    kpis: Dict[str, Any] = {
        # Response time
        "mean_resp_time": mean_resp_time,
        "p50_resp_time": p50_resp_time,
        "p90_resp_time": p90_resp_time,
        "mean_resp_time_high_risk": mean_resp_time_high_risk,
        "p90_resp_time_high_risk": p90_resp_time_high_risk,
        # ALS/BLS
        "als_mismatch_rate": als_mismatch_rate,
        "als_overuse_rate": als_overuse_rate,
        "als_share_of_calls": als_share_of_calls,
        "als_share_of_calls_high_risk": als_share_of_calls_high_risk,
        "als_share_of_calls_low_risk": als_share_of_calls_low_risk,
        # Coverage
        "coverage_loss_mean": coverage_loss_mean,
        "coverage_loss_p90": coverage_loss_p90,
        "coverage_loss_high_risk_mean": coverage_loss_high_risk_mean,
        "coverage_loss_high_risk_p90": coverage_loss_high_risk_p90,
        "coverage_loss_above_thresh_rate": coverage_loss_above_thresh_rate,
        # Fairness
        "fairness_gap_mean": fairness_gap_mean,
        "fairness_gap_p90": fairness_gap_p90,
        "fairness_gap_high_risk_mean": fairness_gap_high_risk_mean,
        "fairness_gap_high_risk_p90": fairness_gap_high_risk_p90,
        # Area mismatch
        "unit_area_mismatch_rate": unit_area_mismatch_rate,
        # Denominators / counts
        "n_calls": n_calls,
        "n_calls_high_risk": n_calls_high_risk,
        "n_calls_low_risk": n_calls_low_risk,
        "n_calls_urban": n_calls_urban,
        "n_calls_rural": n_calls_rural,
        "n_calls_area_info": n_calls_area_info,
        "n_calls_highrisk_als_bls": n_calls_highrisk_als_bls,
        "n_calls_lowrisk_als_bls": n_calls_lowrisk_als_bls,
    }

    # Optional: pull in numeric summary.csv columns as extra KPIs
    if summary_path and summary_path.exists():
        try:
            s_df = pd.read_csv(summary_path)
            if not s_df.empty:
                row = s_df.iloc[0]
                for col, val in row.items():
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        kpis.setdefault(col, float(val))
        except Exception:
            pass

    return kpis


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=str(VARIANTS_DEFAULT), help="Path to variant_candidates.csv")
    ap.add_argument(
        "--runs-root",
        default=str(RUNS_ROOT_DEFAULT),
        help="Root directory containing per-run folders (default: reports/runs/variant_runs)",
    )
    ap.add_argument(
        "--out",
        default=str(OUT_DEFAULT),
        help="Output CSV path (default: reference/variant_kpis_full.csv)",
    )
    args = ap.parse_args()

    variants_df = load_variant_table(Path(args.variants))
    runs_df = index_runs(Path(args.runs_root))

    matched = match_variant_to_run(variants_df, runs_df)
    matched_with_run = matched[~matched["run_id"].isna()].copy()

    if matched_with_run.empty:
        raise RuntimeError("No matched variants with runs after matching. Check variant_candidates vs runs meta.")

    # Compute KPIs per run
    records: List[Dict[str, Any]] = []

    print("[4/4] Extracting v2 KPIs per matched run ...")
    for _, row in matched_with_run.iterrows():
        variant_id = row["variant_id"]
        variant_key = row.get("variant_key", "")
        run_id = row["run_id"]
        decisions_path = Path(row["decisions_path"])
        summary_path = Path(row["summary_path"]) if row.get("summary_path") else None

        try:
            kpis = compute_run_kpis(decisions_path, summary_path)
        except Exception as e:
            print(f"WARNING: skipping run {run_id} for variant {variant_id} due to error: {e}")
            continue

        rec: Dict[str, Any] = {
            "variant_id": variant_id,
            "variant_key": variant_key,
            "run_id": run_id,
        }
        # propagate some variant metadata
        rec["policy"] = row.get("policy", "")
        rec["kwargs"] = row.get("kwargs", "")
        rec["complexity"] = row.get("complexity", None)
        rec["family"] = row.get("family", None)
        rec["note"] = row.get("note", None)

        rec.update(kpis)
        records.append(rec)

    if not records:
        raise RuntimeError("No usable runs after KPI computation.")

    df_runs = pd.DataFrame(records)

    # Aggregate per variant_id across runs
    numeric_cols = [
        c
        for c in df_runs.columns
        if c
        not in ("variant_id", "variant_key", "run_id", "policy", "kwargs", "complexity", "family", "note")
    ]

    count_cols = [
        "n_calls",
        "n_calls_high_risk",
        "n_calls_low_risk",
        "n_calls_urban",
        "n_calls_rural",
        "n_calls_area_info",
        "n_calls_highrisk_als_bls",
        "n_calls_lowrisk_als_bls",
    ]

    agg_spec: Dict[str, Any] = {}
    for col in numeric_cols:
        if col in count_cols:
            agg_spec[col] = "sum"
        else:
            agg_spec[col] = "mean"

    grouped = df_runs.groupby("variant_id").agg(agg_spec)
    grouped["n_runs"] = df_runs.groupby("variant_id")["run_id"].nunique()
    grouped.reset_index(inplace=True)

    # Restore variant metadata (complexity, family, note, policy, kwargs, variant_key)
    meta_cols = ["complexity", "family", "note", "policy", "kwargs", "variant_key"]
    meta_first = (
        df_runs[["variant_id"] + meta_cols]
        .drop_duplicates(subset=["variant_id"])
        .set_index("variant_id")
    )
    grouped = grouped.set_index("variant_id").join(meta_first, how="left")
    grouped.reset_index(inplace=True)

    # Reorder columns: variant_id, complexity, n_runs, then KPIs, then metadata.
    kpi_cols_order = [
        "mean_resp_time",
        "p50_resp_time",
        "p90_resp_time",
        "mean_resp_time_high_risk",
        "p90_resp_time_high_risk",
        "als_mismatch_rate",
        "als_overuse_rate",
        "als_share_of_calls",
        "als_share_of_calls_high_risk",
        "als_share_of_calls_low_risk",
        "coverage_loss_mean",
        "coverage_loss_p90",
        "coverage_loss_high_risk_mean",
        "coverage_loss_high_risk_p90",
        "coverage_loss_above_thresh_rate",
        "fairness_gap_mean",
        "fairness_gap_p90",
        "fairness_gap_high_risk_mean",
        "fairness_gap_high_risk_p90",
        "unit_area_mismatch_rate",
        "n_calls",
        "n_calls_high_risk",
        "n_calls_low_risk",
        "n_calls_urban",
        "n_calls_rural",
        "n_calls_area_info",
        "n_calls_highrisk_als_bls",
        "n_calls_lowrisk_als_bls",
    ]

    cols = ["variant_id", "complexity", "n_runs"]
    cols += [c for c in kpi_cols_order if c in grouped.columns]
    cols += ["policy", "kwargs", "family", "note", "variant_key"]

    cols = [c for c in cols if c in grouped.columns]
    df_out = grouped[cols]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(args.out, index=False)
    print(f"\n→ Wrote {len(df_out)} rows to {args.out}")


if __name__ == "__main__":
    main()