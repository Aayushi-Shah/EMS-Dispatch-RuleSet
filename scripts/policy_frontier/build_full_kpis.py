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
    --variants reference/variant_candidates_frozen.csv \
    --runs-root reports/runs/variant_runs \
    --out reference/variant_kpis_full.csv
"""

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


def safe_literal_eval(cell: Any) -> Dict[str, Any]:
    """
    Parse the 'debug' cell if it's a dict-like string, else return {}.
    """
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


def percentile(series: pd.Series, q: float) -> float:
    """
    Compute q-th percentile for a numeric Series, returning NaN if empty.
    """
    arr = series.dropna().to_numpy()
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


# -----------------------------
# Data loading / matching
# -----------------------------
def load_variant_table(variants_path: Path) -> pd.DataFrame:
    df = pd.read_csv(variants_path)
    if "variant_id" not in df.columns or "policy" not in df.columns:
        raise ValueError("variant_candidates.csv must have at least 'variant_id' and 'policy' columns")

    df["kwargs_dict"] = df["kwargs"].apply(parse_kwargs_cell) if "kwargs" in df.columns else [{}] * len(df)
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
    v = variants_df.copy()
    r = runs_df.copy()

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
# KPI computation
# -----------------------------
def compute_full_kpis(decisions_path: Path, summary_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Compute KPIs from decisions.csv (+ summary.csv if present).
    Returns dict of KPI fields, or None if decisions.csv is unusable.
    """
    try:
        df = pd.read_csv(decisions_path)
    except Exception:
        return None

    if df.empty or "resp_min" not in df.columns:
        return None

    dbg = df["debug"].apply(safe_literal_eval) if "debug" in df.columns else pd.Series([{}] * len(df))

    call_area = dbg.apply(lambda d: d.get("call_area"))
    coverage_loss_raw = dbg.apply(lambda d: d.get("coverage_loss"))
    risk = dbg.apply(lambda d: d.get("risk"))
    utype = dbg.apply(lambda d: d.get("utype"))
    u_area = dbg.apply(lambda d: d.get("u_area"))

    resp = df["resp_min"].astype(float)

    # Core time metrics
    mean_resp_time = float(resp.mean())
    p50_resp_time = percentile(resp, 50.0)
    p90_resp_time = percentile(resp, 90.0)

    # Coverage metrics
    cov = pd.to_numeric(coverage_loss_raw, errors="coerce").fillna(0.0)
    coverage_loss_mean = float(cov.mean())
    coverage_loss_p90 = percentile(cov, 90.0)

    # Fairness: urban vs rural mean response gap
    urban_mask = call_area == "urban"
    rural_mask = call_area == "rural"
    if urban_mask.any() and rural_mask.any():
        mean_urban = float(resp[urban_mask].mean())
        mean_rural = float(resp[rural_mask].mean())
        fairness_gap = abs(mean_urban - mean_rural)
    else:
        fairness_gap = float("nan")

    # ALS / high-risk mismatch
    risk_vals = pd.to_numeric(risk, errors="coerce")
    utype_str = utype.astype(str).str.upper()

    high_risk_mask = risk_vals >= 0.75
    bls_mask = utype_str == "BLS"

    denom_mask = high_risk_mask & utype_str.isin(["ALS", "BLS"])
    denom = int(denom_mask.sum())
    if denom > 0:
        mismatches = int((high_risk_mask & bls_mask).sum())
        als_mismatch_rate = mismatches / denom
    else:
        als_mismatch_rate = float("nan")

    # Unit area vs call area mismatch
    known_u = u_area.isin(["urban", "rural"])
    known_c = call_area.isin(["urban", "rural"])
    both_known = known_u & known_c

    denom_area = int(both_known.sum())
    if denom_area > 0:
        area_mismatch = int((both_known & (u_area != call_area)).sum())
        unit_area_mismatch_rate = area_mismatch / denom_area
    else:
        unit_area_mismatch_rate = float("nan")

    kpis = {
        "mean_resp_time": mean_resp_time,
        "p50_resp_time": p50_resp_time,
        "p90_resp_time": p90_resp_time,
        "coverage_loss_mean": coverage_loss_mean,
        "coverage_loss_p90": coverage_loss_p90,
        "fairness_gap": fairness_gap,
        "als_mismatch_rate": als_mismatch_rate,
        "unit_area_mismatch_rate": unit_area_mismatch_rate,
    }

    # Pull in numeric summary.csv columns as extra KPIs if present
    if summary_path and Path(summary_path).exists():
        try:
            s_df = pd.read_csv(summary_path)
            if not s_df.empty:
                row = s_df.iloc[0]
                for col, val in row.items():
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        kpis.setdefault(col, val)
        except Exception:
            pass

    return kpis


# -----------------------------
# Main
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
    matched_with_run = matched[~matched["run_id"].isna()]

    records: List[Dict[str, Any]] = []

    print("[4/4] Extracting full KPIs per matched variant ...")
    for _, row in matched_with_run.iterrows():
        # Use frozen_variant_id if present, otherwise fall back to original
        output_id = row.get("frozen_variant_id", row["variant_id"])
        base_id = row["variant_id"]  # original ID used in runs/meta
        run_id = row["run_id"]
        decisions_path = row["decisions_path"]
        summary_path = row.get("summary_path", None)

        kpis = compute_full_kpis(Path(decisions_path), Path(summary_path) if summary_path else None)
        if kpis is None:
            continue

        rec = {
            "variant_id": output_id,          # <- what downstream sees
            "base_variant_id": base_id,       # <- original for debugging
            "run_id": run_id,
            "policy": row["policy"],
            "kwargs": row.get("kwargs", ""),
            "complexity": row.get("complexity", None),
            "note": row.get("note", None),
        }

        rec.update(kpis)
        records.append(rec)

    df_out = pd.DataFrame(records)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(args.out, index=False)
    print(f"\n→ Wrote {len(df_out)} rows to {args.out}")


if __name__ == "__main__":
    main()