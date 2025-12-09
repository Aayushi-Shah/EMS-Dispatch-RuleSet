# experiments/rulelist_policies/analysis/build_critical_zones.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config


def load_calls(path: Path | None) -> pd.DataFrame:
    """
    Load tagged LEMSA calls from a parquet file (default = config.CALLS_PARQUET).
    """
    if path is None:
        calls_path = Path(config.CALLS_PARQUET)
    else:
        calls_path = Path(path)

    if not calls_path.exists():
        raise FileNotFoundError(f"Calls file not found: {calls_path}")

    df = pd.read_parquet(calls_path)
    return df


def compute_zone_stats(
    df: pd.DataFrame,
    zone_col: str = "municipality",
    sev_col: str = "severity_bucket",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Compute per-zone priority-weighted stats and return the top-N zones.

    Columns expected:
      - zone_col: e.g., 'municipality'
      - sev_col : e.g., 'severity_bucket' with values like 'high', 'medium', 'low'
    """

    if zone_col not in df.columns:
        raise RuntimeError(f"Zone column '{zone_col}' not in calls dataframe.")

    if sev_col not in df.columns:
        raise RuntimeError(f"Severity column '{sev_col}' not in calls dataframe.")

    # Normalise fields
    df = df.copy()
    df[zone_col] = df[zone_col].astype(str).str.strip()
    df[zone_col] = df[zone_col].where(df[zone_col] != "", other="UNKNOWN")

    sev = (
        df[sev_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Priority weights: tune if you want, but this matches your plan:
    #   high = 3, medium = 2, others = 1
    weight_map = {
        "high": 3.0,
        "medium": 2.0,
        "low": 1.0,
    }
    df["_sev_norm"] = sev
    df["_pw"] = df["_sev_norm"].map(weight_map).fillna(1.0)

    # Aggregate per zone
    grp = df.groupby(zone_col, dropna=False)

    agg = grp.agg(
        total_calls=("incidentTime", "count"),  # any non-null column is fine
        pw_calls=("_pw", "sum"),
        high_calls=("_sev_norm", lambda s: (s == "high").sum()),
        medium_calls=("_sev_norm", lambda s: (s == "medium").sum()),
        low_calls=("_sev_norm", lambda s: (s == "low").sum()),
    ).reset_index(names=[zone_col])

    # Derive fractions
    agg["frac_high"] = agg["high_calls"] / agg["total_calls"].clip(lower=1)
    agg["frac_medium"] = agg["medium_calls"] / agg["total_calls"].clip(lower=1)

    # Sort by priority-weighted calls descending
    agg = agg.sort_values("pw_calls", ascending=False)

    # Pick top-N
    top_n = int(top_n)
    if top_n > 0:
        agg_top = agg.head(top_n).reset_index(drop=True)
    else:
        agg_top = agg.reset_index(drop=True)

    return agg_top


def main():
    parser = argparse.ArgumentParser(
        description="Build priority-weighted critical demand zones from historical CAD."
    )
    parser.add_argument(
        "--calls",
        type=str,
        default=None,
        help="Path to tagged calls parquet (default: config.CALLS_PARQUET)",
    )
    parser.add_argument(
        "--zone-col",
        type=str,
        default="municipality",
        help="Column to use as zone identifier (default: municipality)",
    )
    parser.add_argument(
        "--sev-col",
        type=str,
        default="severity_bucket",
        help="Column with severity buckets (default: severity_bucket)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of critical zones to keep (default: 5)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="tools/critical_zones/critical_zones_municipality.csv",
        help="Output CSV path for critical zone table.",
    )

    args = parser.parse_args()

    df = load_calls(Path(args.calls) if args.calls else None)
    crit = compute_zone_stats(
        df,
        zone_col=args.zone_col,
        sev_col=args.sev_col,
        top_n=args.top_n,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crit.to_csv(out_path, index=False)

    print(f"Critical zones written to: {out_path}")
    print(crit)


if __name__ == "__main__":
    main()
