#!/usr/bin/env python3
import re
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.ems_config import settings


UNIT_TOKEN = re.compile(
    r"\b(?P<type>AMBULANCE|AMB|MICU|MEDIC|ALS|BLS)?\s*(?P<st>\d{2})-(?P<num>\d+)\b",
    re.IGNORECASE
)

def norm_unit(s: str) -> str:
    m = UNIT_TOKEN.search(s or "")
    if not m: return ""
    t = (m.group("type") or "").upper()
    if t == "AMBULANCE": t = "AMB"
    return f"{t} {m.group('st')}-{m.group('num')}".strip()

def main():
    Path(settings.UNITS_VS_CAD_REPORT).parent.mkdir(parents=True, exist_ok=True)

    # Load CAD calls
    df = pd.read_parquet(str(settings.CALLS_FOR_VERIFICATION))
    df["unitsString"] = df.get("unitsString","").astype(str)
    # Extract all unit tokens seen in CAD
    seen = []
    for s in df["unitsString"]:
        for tok in re.split(r"[;,/]| {2,}", s):
            tok = tok.strip()
            if not tok: continue
            u = norm_unit(tok)
            if u:
                seen.append(u)
    seen_df = pd.Series(seen, name="unit_designator").value_counts().rename_axis("unit_designator").reset_index(name="calls")

    # Load stations & manual units
    st = pd.read_csv(str(settings.STATIONS_CSV))
    mu = pd.read_csv(str(settings.MANUAL_UNITS_CSV))
    mu["unit_designator"] = mu["unit_designator"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True)
    mu["station_number"] = mu["station_number"].astype(str).str.upper()

    # Join station name for clarity
    mu = mu.merge(st[["station_number","station_name"]], on="station_number", how="left")

    # Compare coverage
    merged = mu.merge(seen_df, on="unit_designator", how="left")
    merged["calls"] = merged["calls"].fillna(0).astype(int)

    # Units in CAD but not in manual
    cad_only = seen_df.merge(mu[["unit_designator"]], on="unit_designator", how="left", indicator=True)
    cad_only = cad_only[cad_only["_merge"] == "left_only"].drop(columns=["_merge"])

    # Write report
    lines = []
    lines.append("LEMSA Units – Manual vs CAD Coverage\n")
    lines.append("Manual units with CAD call counts:")
    for _, r in merged.sort_values("unit_designator").iterrows():
        lines.append(f"  {r['unit_designator']:<12}  station={r['station_number']:<5}  calls={r['calls']:<6}  {r.get('station_name','')}")
    lines.append("")
    lines.append("Units seen in CAD but missing from manual list:")
    if len(cad_only) == 0:
        lines.append("  (none)")
    else:
        for _, r in cad_only.sort_values("calls", ascending=False).iterrows():
            lines.append(f"  {r['unit_designator']:<12}  calls={r['calls']:<6}")
    Path(settings.UNITS_VS_CAD_REPORT).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n🧾 Report: {settings.UNITS_VS_CAD_REPORT}")

if __name__ == "__main__":
    main()
