#!/usr/bin/env python3
"""
discover_lemsa_units.py
Discovers units from CAD data by extracting unit designators from LEMSA calls.
Separately tracks:
  - LEMSA units (06-XX, 56-XX stations)
  - Mutual aid units (all other stations responding in LEMSA coverage)
Compares against manual list to identify potentially missing units.
"""
import re
from pathlib import Path
import pandas as pd

# Input/output paths
LEMSA_CALLS = Path("data/processed/medical_calls_lemsa.parquet")
MANUAL_UNITS = Path("reference/lemsa_units_manual.csv")
OUTPUT_LEMSA = Path("reference/lemsa_units_candidates.csv")
OUTPUT_MUTUAL_AID = Path("reference/lemsa_mutual_aid_units.csv")
OUTPUT_REPORT = Path("reports/lemsa_unit_discovery.txt")

# LEMSA station numbers (from reference/lemsa_stations.csv)
LEMSA_STATIONS = {"06", "6", "56"}  # Allow both "06" and "6" format

# Unit regex pattern
UNIT_TOKEN = re.compile(
    r"\b(AMBULANCE|AMB|MICU|MEDIC|ALS|BLS)?\s*0*(\d{1,2})-(\d+)\b",
    re.IGNORECASE
)

def normalize_unit(unit_str: str) -> str:
    """Normalize unit designator to standard format: TYPE STATION-NUMBER"""
    m = UNIT_TOKEN.search(str(unit_str))
    if not m:
        return ""
    
    unit_type = m.group(1) or ""
    station = m.group(2)
    number = m.group(3)
    
    # Normalize unit type
    unit_type = unit_type.upper().strip()
    if unit_type == "AMBULANCE":
        unit_type = "AMB"
    elif unit_type in ("ALS", "BLS"):
        unit_type = ""  # Type-only without unit number, skip
    
    # Normalize station (handle "06" vs "6")
    station_num = int(station)
    if station_num < 10:
        station_str = f"0{station_num}"
    else:
        station_str = str(station_num)
    
    # Build normalized unit
    if unit_type:
        return f"{unit_type} {station_str}-{number}"
    return f"{station_str}-{number}"

def is_lemsa_unit(unit_str: str) -> bool:
    """Check if unit belongs to LEMSA station (06-XX or 56-XX)"""
    m = UNIT_TOKEN.search(str(unit_str))
    if not m:
        return False
    station = m.group(2)
    return station in LEMSA_STATIONS or int(station) in [6, 56]

def extract_units_from_string(s: str) -> tuple[set, set]:
    """
    Extract all unit designators from a unitsString.
    Returns: (lemsa_units, mutual_aid_units)
    """
    if pd.isna(s) or not s:
        return set(), set()
    
    lemsa_units = set()
    mutual_aid_units = set()
    
    # Split on common delimiters
    for token in re.split(r"[;,/]| {2,}", str(s)):
        token = token.strip()
        if not token:
            continue
        
        unit = normalize_unit(token)
        if not unit:
            continue
        
        # Check if LEMSA or mutual aid
        if is_lemsa_unit(token):
            lemsa_units.add(unit)
        else:
            # It's a valid unit but not LEMSA = mutual aid
            mutual_aid_units.add(unit)
    
    return lemsa_units, mutual_aid_units

def main():
    print("=" * 60)
    print("LEMSA Unit Discovery")
    print("=" * 60)
    
    # Load LEMSA calls
    if not LEMSA_CALLS.exists():
        raise FileNotFoundError(f"LEMSA calls file not found: {LEMSA_CALLS}")
    
    print(f"\nLoading LEMSA calls from {LEMSA_CALLS}...")
    df = pd.read_parquet(LEMSA_CALLS)
    print(f"  Total calls: {len(df):,}")
    
    if "unitsString" not in df.columns:
        raise ValueError("LEMSA calls must have 'unitsString' column")
    
    # Extract all units (LEMSA and mutual aid)
    print("\nExtracting unit designators from calls...")
    all_lemsa_units = []
    all_mutual_aid_units = []
    
    for idx, units_str in df["unitsString"].items():
        try:
            if pd.isna(units_str):
                units_str_val = ""
            else:
                units_str_val = str(units_str)
        except (TypeError, ValueError):
            units_str_val = ""
        lemsa_units, mutual_aid_units = extract_units_from_string(units_str_val)
        all_lemsa_units.extend(lemsa_units)
        all_mutual_aid_units.extend(mutual_aid_units)
    
    # Count LEMSA units
    lemsa_df = pd.Series(all_lemsa_units, name="unit_designator").value_counts().reset_index()
    lemsa_df.columns = ["unit_designator", "calls"]
    lemsa_df = lemsa_df.sort_values("calls", ascending=False)
    
    # Count mutual aid units
    mutual_aid_df = pd.Series(all_mutual_aid_units, name="unit_designator").value_counts().reset_index()
    mutual_aid_df.columns = ["unit_designator", "calls"]
    mutual_aid_df = mutual_aid_df.sort_values("calls", ascending=False)
    
    print(f"  Found {len(lemsa_df)} unique LEMSA units")
    print(f"  Found {len(mutual_aid_df)} unique mutual aid units")
    
    # Save LEMSA candidates
    OUTPUT_LEMSA.parent.mkdir(parents=True, exist_ok=True)
    lemsa_df.to_csv(OUTPUT_LEMSA, index=False)
    print(f"\n✅ Saved LEMSA units: {OUTPUT_LEMSA}")
    
    # Save mutual aid units
    OUTPUT_MUTUAL_AID.parent.mkdir(parents=True, exist_ok=True)
    mutual_aid_df.to_csv(OUTPUT_MUTUAL_AID, index=False)
    print(f"✅ Saved mutual aid units: {OUTPUT_MUTUAL_AID}")
    
    # Compare LEMSA units with manual list
    if MANUAL_UNITS.exists():
        print(f"\nComparing LEMSA units with manual list: {MANUAL_UNITS}")
        manual_df = pd.read_csv(MANUAL_UNITS)
        manual_units = set(manual_df["unit_designator"].str.upper().str.strip())
        discovered_lemsa_units = set(lemsa_df["unit_designator"].str.upper().str.strip())
        
        # Find missing units (in discovered, not in manual)
        missing_units = discovered_lemsa_units - manual_units
        missing_df = lemsa_df[lemsa_df["unit_designator"].str.upper().str.strip().isin(missing_units)].copy()
        
        # Find units in manual but not discovered (low activity?)
        manual_only = manual_units - discovered_lemsa_units
        manual_only_df = manual_df[manual_df["unit_designator"].str.upper().str.strip().isin(manual_only)].copy()
        
        # Generate report
        OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("LEMSA Unit Discovery Report")
        lines.append("=" * 60)
        lines.append(f"\nManual units: {len(manual_units)}")
        lines.append(f"Discovered LEMSA units (from CAD): {len(discovered_lemsa_units)}")
        lines.append(f"Overlap (in both): {len(manual_units & discovered_lemsa_units)}")
        
        lines.append(f"\n{'=' * 60}")
        lines.append("POTENTIALLY MISSING LEMSA UNITS (in CAD, not in manual)")
        lines.append(f"Count: {len(missing_units)}")
        lines.append("=" * 60)
        if len(missing_df) > 0:
            lines.append("\nUnit Designator          | Calls")
            lines.append("-" * 60)
            for _, row in missing_df.iterrows():
                lines.append(f"{row['unit_designator']:<24} | {row['calls']:>6,}")
        else:
            lines.append("\n(none)")
        
        lines.append(f"\n{'=' * 60}")
        lines.append("MANUAL UNITS NOT SEEN IN CAD (low activity or retired?)")
        lines.append(f"Count: {len(manual_only)}")
        lines.append("=" * 60)
        if len(manual_only_df) > 0:
            lines.append("\nUnit Designator          | Notes")
            lines.append("-" * 60)
            for _, row in manual_only_df.iterrows():
                notes = row.get("notes", "")
                lines.append(f"{row['unit_designator']:<24} | {notes}")
        else:
            lines.append("\n(none)")
        
        # Add mutual aid summary
        lines.append(f"\n{'=' * 60}")
        lines.append("MUTUAL AID UNITS (responding in LEMSA coverage)")
        lines.append(f"Total unique mutual aid units: {len(mutual_aid_df)}")
        lines.append(f"Total mutual aid responses: {mutual_aid_df['calls'].sum():,}")
        lines.append("=" * 60)
        if len(mutual_aid_df) > 0:
            lines.append("\nTop mutual aid units:")
            lines.append("\nUnit Designator          | Calls | Station")
            lines.append("-" * 60)
            for _, row in mutual_aid_df.head(20).iterrows():
                # Extract station number
                unit_str = str(row['unit_designator'])
                station_match = re.search(r'(\d{1,2})-\d+', unit_str)
                station = station_match.group(1) if station_match else "?"
                lines.append(f"{unit_str:<24} | {row['calls']:>6,} | {station}")
        
        # Write report
        OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n✅ Discovery report: {OUTPUT_REPORT}")
        
        # Print summary
        print("\n📊 Summary:")
        print(f"   Manual units: {len(manual_units)}")
        print(f"   Discovered LEMSA units: {len(discovered_lemsa_units)}")
        print(f"   ✅ In both: {len(manual_units & discovered_lemsa_units)}")
        print(f"   ⚠️  Potentially missing (in CAD, not manual): {len(missing_units)}")
        print(f"   ⚠️  Manual units not seen in CAD: {len(manual_only)}")
        print(f"\n   🤝 Mutual aid units: {len(mutual_aid_df)}")
        print(f"   🤝 Total mutual aid responses: {mutual_aid_df['calls'].sum():,}")
        
        if len(missing_df) > 0:
            print(f"\n🔍 Top missing LEMSA units to review:")
            for _, row in missing_df.head(10).iterrows():
                print(f"   {row['unit_designator']:<15} ({row['calls']:>5,} calls)")
        
        if len(mutual_aid_df) > 0:
            print(f"\n🤝 Top mutual aid units:")
            for _, row in mutual_aid_df.head(10).iterrows():
                print(f"   {row['unit_designator']:<15} ({row['calls']:>5,} calls)")
    else:
        print(f"\n⚠️  Manual units file not found: {MANUAL_UNITS}")
        print("   Skipping comparison. Run after creating manual list.")
        print(f"\n   🤝 Mutual aid units: {len(mutual_aid_df)}")
        print(f"   🤝 Total mutual aid responses: {mutual_aid_df['calls'].sum():,}")

if __name__ == "__main__":
    main()
