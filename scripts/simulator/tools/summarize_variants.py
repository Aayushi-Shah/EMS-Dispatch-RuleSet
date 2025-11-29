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
    candidates = {row["variant_id"]: row for row in csv.DictReader(candidates_csv.open())}
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
        # Expect variant_id prefix in run_id like "<variant>_<timestamp>"
        variant_id = run_id.split("_")[0]
        cand = candidates.get(variant_id, {})
        complexity = cand.get("complexity")
        policy = cand.get("policy")

        summary = load_summary(run_dir)
        agg = aggregate_decisions(run_dir)
        row: Dict[str, Any] = {
            "run_id": run_id,
            "variant_id": variant_id,
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
