# config/delta_eta_logging.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

LOG_DIR = Path("results/delta_eta")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_path(scenario: str) -> Path:
    return LOG_DIR / f"{scenario}.jsonl"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Try attribute access first, then mapping-style access (dict-like).
    Never throw; always fall back to default.
    """
    # Attribute access
    try:
        if hasattr(obj, key):
            val = getattr(obj, key)
            if val is not None:
                return val
    except Exception:
        pass

    # Mapping (dict / pandas row / etc.)
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        if hasattr(obj, "get"):
            return obj.get(key, default)
    except Exception:
        pass

    return default

def log_delta_eta_choice(
    scenario: str,
    call: Any,
    nearest_unit: Any,
    candidate_units: Sequence[Any],
    eta_by_unit: dict[Any, float],
    unit_id_field: str = "unit_designator",
) -> None:
    """
    Log enough structure so we can compute ΔETA offline.

    We DO NOT decide the alternative here; we just dump:
    - who was nearest,
    - what all ETAs were,
    - and tags we care about (critical muni, ALS/BLS, etc.).
    """

    try:
        # ---- Call fields ----
        # ID
        call_id = (
            _get(call, "incidentID")
            or _get(call, "call_id")
        )

        # Tags we care about
        call_zone = _get(call, "zone")
        call_severity = _get(call, "severity_bucket")
        call_pref = _get(call, "preferred_unit_type")

        # Use standardized municipality if present, fall back to raw
        call_muni = _get(call, "municipality_std") or _get(call, "municipality")

        call_crit_raw = _get(call, "is_critical_municipality", False)
        call_crit = bool(call_crit_raw) if call_crit_raw is not None else False

        # ---- Nearest unit ----
        nearest_uid = _get(nearest_unit, unit_id_field)

        # ---- Candidate payload ----
        cand_payload = []
        for u in candidate_units:
            uid = _get(u, unit_id_field)
            if uid is None:
                continue

            unit_muni = _get(u, "municipality_std") or _get(u, "municipality")
            unit_crit_raw = _get(u, "is_critical_municipality", False)
            unit_crit = bool(unit_crit_raw) if unit_crit_raw is not None else False

            cand_payload.append(
                {
                    "unit_id": uid,
                    "utype": _get(u, "utype"),
                    "unit_zone": _get(u, "unit_zone"),
                    "unit_area": _get(u, "unit_area"),
                    "unit_municipality": unit_muni,
                    "unit_is_critical_municipality": unit_crit,
                    "eta": eta_by_unit.get(uid),
                }
            )

        record = {
            "scenario": scenario,
            "call_id": call_id,
            "call_zone": call_zone,
            "call_severity": call_severity,
            "call_preferred_unit_type": call_pref,
            "call_municipality": call_muni,
            "call_is_critical_municipality": call_crit,
            "nearest_unit_id": nearest_uid,
            "candidates": cand_payload,
        }

        path = _log_path(scenario)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        # Never crash the sim because logging broke
        return