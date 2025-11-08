from datetime import datetime, timezone
import json
import pandas as pd
from . import config

def new_run_id():
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def write_artifacts(run_id: str, per_shift_df: pd.DataFrame, summary_df: pd.DataFrame, meta: dict):
    run_dir = (config.RUNS_DIR / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    per_shift_csv = run_dir / "per_shift.csv"
    summary_csv   = run_dir / "summary.csv"
    meta_json     = run_dir / "meta.json"

    per_shift_df.to_csv(per_shift_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    meta_json.write_text(json.dumps(meta, indent=2))

    # append to runlog
    config.RUNLOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    log_row = {
        "run_id": run_id,
        "calls": int(meta["counts"]["calls"]),
        "units": int(meta["counts"]["units"]),
        "segments": int(meta["counts"]["segments"]),
        "w_avg_p50_resp_min": float(summary_df["w_avg_p50_resp_min"].iloc[0]) if "w_avg_p50_resp_min" in summary_df else None,
        "w_avg_p90_resp_min": float(summary_df["w_avg_p90_resp_min"].iloc[0]) if "w_avg_p90_resp_min" in summary_df else None,
        "w_avg_avg_resp_min": float(summary_df["w_avg_avg_resp_min"].iloc[0]) if "w_avg_avg_resp_min" in summary_df else None,
    }
    try:
        existing = pd.read_csv(config.RUNLOG_CSV)
        out = pd.concat([existing, pd.DataFrame([log_row])], ignore_index=True)
    except Exception:
        out = pd.DataFrame([log_row])
    out.to_csv(config.RUNLOG_CSV, index=False)

    return {
        "per_shift_csv": str(per_shift_csv),
        "summary_csv":   str(summary_csv),
        "meta_json":     str(meta_json),
        "runlog_csv":    str(config.RUNLOG_CSV),
    }