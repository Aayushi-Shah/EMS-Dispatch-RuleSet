#!/usr/bin/env python3
"""
LEMSA stations + units from lemsa.com locations page, merged with County PDF.

Outputs:
  reference/lemsa_stations.csv
  reference/lemsa_units_from_site.csv

Usage:
  python scripts/lemsa_stations_and_units_from_site.py \
    --lemsa-locations-url "https://www.lemsa.com/service-areas-and-locations" \
    --county-pdf "https://www.lancastercountypa.gov/DocumentCenter/View/18081/EMS-Stations-by-Municipality" \
    --out-stations reference/lemsa_stations.csv \
    --out-units reference/lemsa_units_from_site.csv \
    --geocode
"""

from __future__ import annotations
import argparse, io, re, time, json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UA = "LancasterEMSResearch/1.0 (+contact: your-email@example.com)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"])
)))
SESSION.mount("http://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"])
)))

# ---------- Patterns ----------
STATION_HDR = re.compile(r"(?P<name>.+?)\s*\((?P<num>\d{2}-\d+)\)\s*$")
ADDR_PAT    = re.compile(r"\d{2,6}\s+[A-Za-z0-9.\-'\s]+(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Pike|Lane|Ln\.?|Drive|Dr\.?|Way|Blvd\.?|Court|Ct\.?|Circle|Cir\.?)\b", re.I)
CITYZIP_PAT = re.compile(r"\b([A-Za-z][A-Za-z .'\-]+),\s*PA\s*\d{5}\b", re.I)
PHONE_PAT   = re.compile(r"(?:p\.\s*)?(\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4})", re.I)
UNIT_LINE   = re.compile(r"\b(?P<type>AMBULANCE|AMB|MICU|MEDIC|ALS|BLS)\s+(?P<num>\d{2}-\d+)\b", re.I)

PDF_ROW_PAT = re.compile(
    r"(LEMSA\s*[-–]\s*[^\n]*?)\s+(Station\s+(?P<num>(06|56)-\d+))\s+(?P<addr>\d+[^,\n]+,\s*[A-Za-z .'-]+,\s*PA(?:\s*\d{5})?)",
    re.IGNORECASE
)

def norm(s: Optional[str]) -> str:
    import re
    return re.sub(r"\s+", " ", (s or "").strip())

def fetch_text(url: str) -> str:
    r = SESSION.get(url, timeout=40)
    r.raise_for_status()
    return r.text

def fetch_bytes(url: str) -> bytes:
    r = SESSION.get(url, timeout=60)
    r.raise_for_status()
    return r.content

# ---------- LEMSA locations page parsing ----------
def extract_stations_and_units_from_locations(url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the locations page as text blocks. We look for station headers like:
      "Buck Station (56-4)"
    followed by lines with address, city/zip, phone, and optional unit lines:
      "Medic 56-4", "Ambulance 56-9", etc.
    """
    html = fetch_text(url)
    soup = BeautifulSoup(html, "html.parser")

    # Build a linear list of meaningful text lines preserving order
    texts: List[str] = []
    for el in soup.select("h1,h2,h3,h4,strong,b,p,li,div"):
        txt = norm(el.get_text(" "))
        if txt:
            texts.append(txt)

    stations: List[Dict] = []
    units: List[Dict] = []

    i = 0
    while i < len(texts):
        line = texts[i]
        m_hdr = STATION_HDR.search(line)
        if not m_hdr:
            i += 1
            continue

        station_name = norm(m_hdr.group("name"))
        station_num  = m_hdr.group("num").upper()

        # Look ahead a few lines for address/city/phone & unit lines
        address, cityzip, phone = "", "", ""
        j = i + 1
        seen_unit_lines = []
        while j < len(texts) and j <= i + 8:  # limit local context
            t = texts[j]

            if not address:
                m1 = ADDR_PAT.search(t)
                if m1: address = norm(m1.group(0))
            if not cityzip:
                m2 = CITYZIP_PAT.search(t)
                if m2: cityzip = norm(m2.group(0))
            if not phone:
                m3 = PHONE_PAT.search(t)
                if m3: phone = m3.group(1)

            for um in UNIT_LINE.finditer(t):
                utype = um.group("type").upper()
                if utype == "AMBULANCE":
                    utype = "AMB"
                u_num = um.group("num").upper()
                # keep only unit numbers that match this station prefix
                if u_num.split("-")[0] == station_num.split("-")[0]:
                    seen_unit_lines.append((utype, u_num, t))

            j += 1

        stations.append({
            "station_id": f"LEMSA-{station_num}",
            "station_number": station_num,
            "station_name": station_name,
            "role": "Station",
            "address": ", ".join([x for x in [address, cityzip] if x]),
            "phone": phone,
            "latitude": "", "longitude": "",
            "source_url": url
        })

        for utype, u_num, raw in seen_unit_lines:
            units.append({
                "agency": "Lancaster EMS (LEMSA)",
                "station_number": station_num,
                "unit_designator": f"{utype} {u_num}",
                "unit_type": utype,
                "detail_source": url,
                "raw_context": raw
            })

        i = j

    return pd.DataFrame(stations), pd.DataFrame(units)

# ---------- County PDF parsing ----------
def parse_county_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            text = pdf_bytes.decode("latin-1", errors="ignore")

    rows = []
    for m in PDF_ROW_PAT.finditer(text):
        full_name = norm(m.group(1))
        station_no = norm(m.group("num")).upper()
        addr = norm(m.group("addr"))
        rows.append({
            "station_id": f"LEMSA-{station_no}",
            "station_number": station_no,
            "station_name": full_name,
            "role": "Station",
            "address": addr,
            "phone": "",
            "latitude": "", "longitude": "",
            "source_url": "County PDF"
        })
    return pd.DataFrame(rows)

# ---------- Geocode ----------
def geocode_series(addresses: pd.Series) -> Tuple[list, list]:
    lats, lons = [], []
    for addr in addresses.fillna(""):
        a = addr.strip()
        if not a:
            lats.append(""); lons.append(""); continue
        try:
            r = SESSION.get(
                "https://nominatim.openstreetmap.org/search",
                params={"format":"json","limit":1,"q":f"{a}, Lancaster County, PA"},
                timeout=30, headers={"User-Agent": UA}
            )
            r.raise_for_status()
            js = r.json()
            if js:
                lats.append(js[0]["lat"]); lons.append(js[0]["lon"])
            else:
                lats.append(""); lons.append("")
        except Exception:
            lats.append(""); lons.append("")
        time.sleep(1.0)
    return lats, lons

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lemsa-locations-url", required=True)
    ap.add_argument("--county-pdf", required=True)
    ap.add_argument("--out-stations", default="reference/lemsa_stations.csv")
    ap.add_argument("--out-units", default="reference/lemsa_units_from_site.csv")
    ap.add_argument("--geocode", action="store_true", default=True)
    args = ap.parse_args()

    Path(args.out_stations).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_units).parent.mkdir(parents=True, exist_ok=True)

    print("🌐 Parsing LEMSA locations page for stations + units…")
    df_loc_st, df_loc_units = extract_stations_and_units_from_locations(args.lemsa_locations_url)
    print(f"   → stations found on page: {len(df_loc_st)}")
    print(f"   → unit lines found on page: {len(df_loc_units)}")

    print("📄 Parsing County PDF for station supplement…")
    pdf_bytes = fetch_bytes(args.county_pdf)
    df_pdf = parse_county_pdf(pdf_bytes)
    print(f"   → stations found in PDF: {len(df_pdf)}")

    # Merge stations: prefer LEMSA page rows; fill missing address/phone from either source
    st = pd.concat([df_loc_st, df_pdf], ignore_index=True)
    # Dedup by station_number; keep the row with longer address/has phone
    st = (st.sort_values(by=["address"], key=lambda s: s.str.len().fillna(0))
            .drop_duplicates(subset=["station_number"], keep="last"))
    st = st.sort_values("station_number")

    # Geocode if needed
    if args.geocode and len(st):
        need = st[(st["latitude"]=="") | (st["longitude"]=="")]
        if len(need):
            print(f"🧭 Geocoding {len(need)} stations…")
            lats, lons = geocode_series(need["address"])
            st.loc[need.index, "latitude"] = lats
            st.loc[need.index, "longitude"] = lons

    # Save stations
    st.to_csv(args.out_stations, index=False)
    print(f"✅ Saved stations → {args.out_stations} (rows: {len(st)})")

    # Clean units: dedup, attach station_id coords if available
    if not df_loc_units.empty:
        df_loc_units = df_loc_units.drop_duplicates(subset=["unit_designator","station_number"])
        df_loc_units = df_loc_units.merge(
            st[["station_number","station_id","station_name","latitude","longitude"]],
            on="station_number", how="left"
        )
        df_loc_units = df_loc_units.sort_values(["station_number","unit_designator"])
    df_loc_units.to_csv(args.out_units, index=False)
    print(f"✅ Saved units → {args.out_units} (rows: {len(df_loc_units)})")

    # Audit snippet
    if len(df_loc_units):
        top = (df_loc_units.groupby("station_number")["unit_designator"]
               .count().sort_values(ascending=False).head(10))
        print("\nTop stations by units listed on site:")
        for k,v in top.items():
            print(f"  {k}: {v} units")

    print("\nSamples:")
    if len(st): print(st.head(8).to_string(index=False))
    if len(df_loc_units): print(df_loc_units.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
