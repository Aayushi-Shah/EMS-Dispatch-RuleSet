from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def load_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    df = pd.read_csv(summary_path)
    # Expect one row; take weighted avg columns if present
    row = df.iloc[0].to_dict()
    return row


def aggregate_decisions(run_dir: Path) -> Dict[str, Any]:
    decisions_path = run_dir / "decisions.csv"
    if not decisions_path.exists():
        return {}

    cov_losses = []
    fair_penalties = []
    mismatches = 0
    total = 0
    with decisions_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            total += 1
            dbg_raw = r.get("debug")
            if not dbg_raw:
                continue
            try:
                dbg = ast.literal_eval(dbg_raw)
            except Exception:
                continue
            cov = dbg.get("coverage_loss")
            if cov is not None:
                cov_losses.append(float(cov))
            fair = dbg.get("fairness_penalty")
            if fair is not None:
                fair_penalties.append(float(fair))
            u_area = dbg.get("u_area")
            c_area = dbg.get("call_area")
            if u_area and c_area and u_area != c_area:
                mismatches += 1

    return {
        "avg_coverage_loss": sum(cov_losses) / len(cov_losses) if cov_losses else None,
        "avg_fairness_penalty": sum(fair_penalties) / len(fair_penalties) if fair_penalties else None,
        "area_mismatch_pct": (mismatches / total * 100.0) if total else None,
    }


def summarize_runs(runs_dir: Path, candidates_csv: Path, out_csv: Path) -> None:
    runs_dir = runs_dir.expanduser()
    # Build lookup maps from candidates CSV
    cand_rows = list(csv.DictReader(candidates_csv.open()))
    candidates = {row["variant_id"]: row for row in cand_rows}

    def _norm_kwargs(raw: str | dict | None) -> str:
        """Return a deterministic JSON string for kwargs to enable matching."""
        if raw is None:
            return "{}"
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw or raw == "{}":
                return "{}"
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                return "{}"
            raw = parsed
        if not isinstance(raw, dict):
            return "{}"
        return json.dumps(raw, sort_keys=True)

    cand_by_policy_kwargs = {
        (row["policy"], _norm_kwargs(row.get("kwargs"))): row for row in cand_rows
    }
    rows: List[Dict[str, Any]] = []

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
        run_id = meta.get("run_id") or run_dir.name
        cfg = meta.get("config") or {}
        policy_from_meta = cfg.get("POLICY_NAME")
        kwargs_from_meta = cfg.get("POLICY_KWARGS") or {}

        # Pull variant hints from meta or run_id prefix
        variant_hint = meta.get("variant_id") or cfg.get("VARIANT_ID") or run_id.split("_")[0]
        cand = candidates.get(variant_hint, {})
        if not cand:
            key = (policy_from_meta, _norm_kwargs(kwargs_from_meta))
            cand = cand_by_policy_kwargs.get(key, {})
            # If no candidate match, skip this run (likely old/manual run)
            if not cand:
                continue
            variant_hint = cand.get("variant_id") or variant_hint

        complexity = cand.get("complexity") or cfg.get("VARIANT_COMPLEXITY")
        policy = cand.get("policy") or cfg.get("VARIANT_POLICY") or policy_from_meta

        summary = load_summary(run_dir)
        agg = aggregate_decisions(run_dir)
        row: Dict[str, Any] = {
            "run_id": run_id,
            "variant_id": variant_hint,
            "policy": policy,
            "complexity": complexity,
        }
        row.update(summary)
        row.update(agg)
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote summary for {len(rows)} runs to {out_csv}")


if __name__ == "__main__":
    summarize_runs(Path("reports/runs"), Path("reports/variant_candidates.csv"), Path("reports/variant_metrics.csv"))
