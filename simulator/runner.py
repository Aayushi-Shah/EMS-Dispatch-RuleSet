# simulator/runner.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Any, Callable

import numpy as np
import pandas as pd
import copy 

from simulator.io import load_calls, load_units
from simulator.des import DES, Unit
import config
from simulator import traffic

from policies.base import DispatchPolicy
from policies.nearestETA import NearestETAPolicy
from policies.rulelist_policies import (
    NearestETAR1Policy,
    NearestETAR2Policy,
    NearestETAR1R2Policy,
)

# -----------------------
# SimState adapter for policies
# -----------------------

class SimStateAdapter:
    """
    Lightweight adapter passed into DispatchPolicy.choose_unit(...).

    Adds urban/rural-aware scene speed.
    """

    def __init__(self, units: List[Unit]):
        self.units: List[Unit] = units
        self.now_min: float = 0.0  # updated by the wrapper before each policy call

    def _get_call_area(self, call: dict) -> str:
        area = (call.get("call_area")
                or call.get("urban_rural")
                or "").lower()
        if area in ("urban", "rural"):
            return area
        return "unknown"

    def _scene_speed_mph(self, call: dict) -> float:
        area = self._get_call_area(call)
        if area == "urban":
            return getattr(config, "SCENE_SPEED_MPH_URBAN", config.SCENE_SPEED_MPH)
        if area == "rural":
            return getattr(config, "SCENE_SPEED_MPH_RURAL", config.SCENE_SPEED_MPH)
        return getattr(config, "SCENE_SPEED_MPH_DEFAULT", config.SCENE_SPEED_MPH)

    def eta_to_scene(self, unit: Unit, call: dict) -> float:
        """
        Compute ETA from the unit's current location to the call location,
        using urban/rural scene speeds.
        """
        call_abs = float(call.get("_abs_epoch", 0.0))
        call_tmin = float(call.get("tmin", 0.0))
        abs_epoch_now = call_abs + (self.now_min - call_tmin) * 60.0

        speed = self._scene_speed_mph(call)

        return traffic.travel_minutes(
            unit.lon,
            unit.lat,
            float(call["lon"]),
            float(call["lat"]),
            speed,
            abs_epoch_now,
        )


def expand_fleet(units: list[Unit], fleet_factor: int = 1) -> list[Unit]:
    """
    Approximate an "infinite" fleet by cloning each physical unit.
    Each clone must behave like an independent, fully idle unit.
    """
    if fleet_factor <= 1:
        return units

    expanded: list[Unit] = []
    for u in units:
        expanded.append(u)
        for i in range(fleet_factor - 1):
            clone = copy.copy(u)
            base_name = getattr(u, "name", f"unit_{id(u)}")
            clone.name = f"{base_name}#x{i+1}"
            # Ensure clones start idle/dispatchable and do not share state.
            clone.can_dispatch = True
            clone.busy_until = 0.0
            clone.on_call_id = None
            expanded.append(clone)

    print(f"[expand_fleet] base_units={len(units)}, factor={fleet_factor}, total={len(expanded)}")
    return expanded


# -----------------------
# Supply stress
# -----------------------

def apply_supply_stress(
    units: list[Unit],
    als_frac: float = 1.0,
    bls_frac: float = 1.0,
    seed: int | None = None,
) -> list[Unit]:
    """
    Downsample ALS/BLS units to simulate reduced fleet.

    - als_frac = 1.0, bls_frac = 1.0 → no change.
    - als_frac = 0.7 → keep ~70% of ALS units.
    - bls_frac = 0.5 → keep ~50% of BLS units.

    Sampling is random but controlled by 'seed' for reproducibility.
    """
    if seed is not None:
        rng = np.random.default_rng(seed + 10_000)
    else:
        rng = np.random.default_rng()

    als_units = [u for u in units if (u.utype or "").upper() == "ALS"]
    bls_units = [u for u in units if (u.utype or "").upper() == "BLS"]
    other_units = [u for u in units if (u.utype or "").upper() not in ("ALS", "BLS")]

    def sample_group(group: list[Unit], frac: float) -> list[Unit]:
        if not group or frac >= 0.999:
            return group
        k = max(1, int(round(len(group) * frac)))
        idx = rng.choice(len(group), size=k, replace=False)
        return [group[i] for i in idx]

    als_keep = sample_group(als_units, als_frac)
    bls_keep = sample_group(bls_units, bls_frac)

    return als_keep + bls_keep + other_units


# -----------------------
# Demand stress: top-N busiest 3-hour windows ONLY
# -----------------------

def find_top_nonoverlapping_peak_windows(
    calls: list[dict],
    window_minutes: int = 180,
    n_windows: int = 4,
) -> list[tuple[float, float, int]]:
    """
    Find the top N non-overlapping busiest windows of length `window_minutes`
    based on call['tmin'].

    Returns a list of tuples:
        (start_t, end_t, count)
    in original tmin space, sorted by start_t.
    """
    if not calls or n_windows <= 0 or window_minutes <= 0:
        return []

    calls_sorted = sorted(calls, key=lambda c: float(c.get("tmin", 0.0)))
    tmins = np.array([float(c.get("tmin", 0.0)) for c in calls_sorted], dtype=float)
    n = len(tmins)
    if n == 0:
        return []

    L = float(window_minutes)

    # Sliding window counts for each potential start index
    j = 0
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        start = tmins[i]
        while j < n and tmins[j] < start + L:
            j += 1
        counts[i] = j - i

    # Candidates: (start_idx, start_t, count)
    candidates = [
        (i, tmins[i], int(counts[i]))
        for i in range(n)
        if counts[i] > 0
    ]
    # Sort busiest first
    candidates.sort(key=lambda x: x[2], reverse=True)

    chosen: list[tuple[float, float, int, int]] = []  # (start_t, end_t, start_idx, count)
    for start_idx, start_t, cnt in candidates:
        if len(chosen) >= n_windows:
            break
        end_t = start_t + L

        # Enforce non-overlap
        overlaps = any(
            not (end_t <= s2 or start_t >= e2)
            for (s2, e2, _, _) in chosen
        )
        if overlaps:
            continue

        chosen.append((start_t, end_t, start_idx, cnt))

    chosen.sort(key=lambda x: x[0])
    return [(s, e, cnt) for (s, e, _i, cnt) in chosen]


def apply_multiwindow_peak_demand_stress(
    calls: list[dict],
    demand_factor: float = 1.0,
    window_minutes: int = 180,
    n_windows: int = 4,
    seed: int | None = None,
) -> list[dict]:
    """
    Peak-only demand stress, exactly as requested:

      1. Find N busiest 3-hour windows over the whole dataset.
      2. For each such window:
         - Keep all original calls in that window.
         - If demand_factor > 1.0, add extra calls by cloning calls from
           OUTSIDE that window, jittering tmin within the window.
      3. Strip ALL calls outside those N windows.
      4. We average metrics over all decisions in those N windows → stable
         peak-only response.

    This function ALWAYS restricts the dataset to those N windows
    (even when demand_factor == 1.0).
    """
    if not calls or window_minutes <= 0 or n_windows <= 0:
        return calls

    if seed is not None:
        rng = np.random.default_rng(seed + 30_000)
    else:
        rng = np.random.default_rng()

    # 1) Find top-N busiest non-overlapping windows
    windows = find_top_nonoverlapping_peak_windows(
        calls, window_minutes=window_minutes, n_windows=n_windows
    )
    if not windows:
        return calls

    print(
        f"[multiwindow peak] demand_factor={demand_factor}, "
        f"window={window_minutes} min, n_windows={len(windows)}"
    )
    for idx, (s, e, cnt) in enumerate(windows):
        print(f"  window {idx}: [{s:.1f}, {e:.1f}) min, calls={cnt}")

    # Pre-sort original calls
    all_calls = sorted(calls, key=lambda c: float(c.get("tmin", 0.0)))

    def in_any_window(t: float) -> bool:
        for (s, e, _cnt) in windows:
            if s <= t < e:
                return True
        return False

    # Base peak-only set: all original calls that fall in ANY of the windows
    base_peak_calls: list[dict] = []
    base_peak_counts = [0] * len(windows)
    for c in all_calls:
        t = float(c.get("tmin", 0.0))
        for w_idx, (s, e, _cnt) in enumerate(windows):
            if s <= t < e:
                base_peak_calls.append(c)
                base_peak_counts[w_idx] += 1
                break

    print(
        "[multiwindow peak] base_peak_calls="
        f"{len(base_peak_calls)}, per-window={base_peak_counts}"
    )

    # If no stress, just return peak-only slice
    if demand_factor <= 1.0:
        peak_only_sorted = sorted(base_peak_calls, key=lambda c: float(c.get("tmin", 0.0)))
        print(
            f"[multiwindow peak] demand_factor<=1.0 → "
            f"return peak-only calls={len(peak_only_sorted)}"
        )
        return peak_only_sorted

    # 2) For each window, add stressed calls.
    # NOTE: previous behavior cloned calls from OUTSIDE the window which
    # could change the composition of peak calls. To preserve the original
    # peak composition we now clone FROM the calls already inside the
    # window (i.e., duplicate peak calls with jitter), keeping zone/
    # severity distributions consistent as load increases.
    clones: list[dict] = []

    for w_idx, (start_t, end_t, _cnt_window) in enumerate(windows):
        # Partition original calls relative to THIS window
        in_window: list[dict] = []
        out_window: list[dict] = []
        for c in all_calls:
            t = float(c.get("tmin", 0.0))
            if start_t <= t < end_t:
                in_window.append(c)
            else:
                out_window.append(c)

        # We only need calls inside the window to clone from; skip empty windows
        if not in_window:
            continue

        extra_factor = demand_factor - 1.0
        extra_target = int(round(extra_factor * len(in_window)))
        if extra_target <= 0:
            continue

        # Sample sources from the peak (in_window) to preserve composition.
        # We sample with replacement and jitter times within the same window.
        idxs = rng.choice(len(in_window), size=extra_target, replace=True)

        for k, idx in enumerate(idxs):
            src = in_window[idx]
            clone = dict(src)

            old_tmin = float(src.get("tmin", 0.0))
            new_tmin = float(rng.uniform(start_t, end_t))
            clone["tmin"] = new_tmin

            # Update tod_min if present (minutes since midnight)
            if "tod_min" in clone:
                day_min = 24.0 * 60.0
                clone["tod_min"] = int(new_tmin % day_min)

            # Adjust abs_epoch if present
            if "_abs_epoch" in clone:
                delta_min = new_tmin - old_tmin
                clone["_abs_epoch"] = float(clone["_abs_epoch"]) + delta_min * 60.0

            # New ID and tag for traceability
            clone["id"] = f"{src.get('id', 'call')}#mw{w_idx}_stress{k}"
            clone["_multiwindow_peak_idx"] = w_idx

            clones.append(clone)

    # 3) Combine: only calls that fall inside ANY of the windows
    stressed_all = base_peak_calls + clones
    stressed_peak_only = [
        c for c in stressed_all
        if in_any_window(float(c.get("tmin", 0.0)))
    ]
    stressed_peak_only.sort(key=lambda c: float(c.get("tmin", 0.0)))

    print(
        f"[multiwindow peak] original_peak_calls={len(base_peak_calls)}, "
        f"extra_calls={len(clones)}, "
        f"final_peak_only={len(stressed_peak_only)}"
    )
    return stressed_peak_only


def apply_global_demand_stress(
    calls: list[dict],
    demand_factor: float = 1.0,
    seed: int | None = None,
) -> list[dict]:
    """
    Apply demand stress across the entire call set (no windowing).

    - If `demand_factor` <= 1.0 returns the original calls unchanged.
    - If `demand_factor` > 1.0, adds extra calls by cloning existing
      calls sampled from the full dataset, and places clones with
      `tmin` uniformly across the full time span. This preserves
      overall composition (zones/severity) while increasing volume
      uniformly across the day.
    """
    if not calls:
        return calls

    if demand_factor <= 1.0:
        return sorted(calls, key=lambda c: float(c.get("tmin", 0.0)))

    if seed is not None:
        rng = np.random.default_rng(seed + 40_000)
    else:
        rng = np.random.default_rng()

    all_calls = sorted(calls, key=lambda c: float(c.get("tmin", 0.0)))
    tmins = np.array([float(c.get("tmin", 0.0)) for c in all_calls], dtype=float)
    tmin_min = float(tmins.min())
    tmin_max = float(tmins.max())

    extra_factor = demand_factor - 1.0
    extra_target = int(round(extra_factor * len(all_calls)))
    clones: list[dict] = []
    if extra_target <= 0:
        return all_calls

    # Sample sources from the full dataset with replacement
    idxs = rng.choice(len(all_calls), size=extra_target, replace=True)

    for k, idx in enumerate(idxs):
        src = all_calls[idx]
        clone = dict(src)

        # Place clone uniformly across the full time span
        new_tmin = float(rng.uniform(tmin_min, tmin_max))
        clone["tmin"] = new_tmin

        if "tod_min" in clone:
            day_min = 24.0 * 60.0
            clone["tod_min"] = int(new_tmin % day_min)

        if "_abs_epoch" in clone:
            old_tmin = float(src.get("tmin", 0.0))
            delta_min = new_tmin - old_tmin
            clone["_abs_epoch"] = float(clone["_abs_epoch"]) + delta_min * 60.0

        clone["id"] = f"{src.get('id', 'call')}#global_stress_{k}"
        clones.append(clone)

    stressed = all_calls + clones
    stressed.sort(key=lambda c: float(c.get("tmin", 0.0)))
    print(f"[global demand] demand_factor={demand_factor}, original_calls={len(all_calls)}, extra_calls={len(clones)}, total={len(stressed)}")
    return stressed


# -----------------------
# Policy factory
# -----------------------

def make_policy(name: str) -> DispatchPolicy:
    """Return a DispatchPolicy instance by name."""
    normalized = name.lower()

    policy_map: dict[str, type[DispatchPolicy]] = {
        "nearest_eta": NearestETAPolicy,
        "nearest_eta_r1": NearestETAR1Policy,
        "nearest_eta_r2": NearestETAR2Policy,
        "nearest_eta_r1_r2": NearestETAR1R2Policy,
    }

    cls = policy_map.get(normalized)
    if cls is None:
        raise ValueError(
            "Unknown policy: {}. Available policies: {}".format(
                name, ", ".join(sorted(policy_map))
            )
        )
    policy = cls()
    setattr(policy, "name", getattr(policy, "name", normalized))
    return policy


# -----------------------
# Adapter from DispatchPolicy to DES.select_unit_fn
# -----------------------

def make_select_unit_fn(policy: DispatchPolicy, units: List[Unit]) -> Callable:
    """
    Adapter from experimental DispatchPolicy interface to DES.select_unit_fn signature.

    DES expects:
        select_unit_fn(units, now_min, call) -> (unit | None, resp_minutes, debug)

    Policy expects:
        policy.choose_unit(call, candidates, sim_state) -> candidate

    We:
      - Filter by availability (can_dispatch + busy_until <= now_min).
      - Provide SimStateAdapter with .now_min updated each call.
      - Let the policy choose a unit.
      - Compute resp_minutes once (eta to scene) via sim_state.eta_to_scene.
    """
    sim_state = SimStateAdapter(units)

    def select_unit_fn(
        all_units: List[Unit],
        now_min: float,
        call: dict,
    ) -> tuple[Unit | None, float, dict]:
        sim_state.now_min = now_min

        candidates = [
            u for u in all_units
            if u.can_dispatch and u.busy_until <= now_min
        ]
        if not candidates:
            return None, float("inf"), {"reason": "no_available_units"}

        chosen = policy.choose_unit(call, candidates, sim_state)
        if chosen is None:
            return None, float("inf"), {"reason": "policy_returned_none"}

        resp_minutes = sim_state.eta_to_scene(chosen, call)

        debug = {
            "policy_name": getattr(policy, "name", type(policy).__name__),
            "n_candidates": len(candidates),
        }
        return chosen, float(resp_minutes), debug

    return select_unit_fn


# ----------------
# Runner / harness
# ----------------

def run_simulation(
    policy_name: str,
    out_dir: Path | None = None,
    seed: int | None = None,
    als_frac: float = 1.0,
    bls_frac: float = 1.0,
    demand_factor: float = 1.0,
    scenario_name: str = "unknown",
    fleet_factor: int = 1,
) -> dict:
    """
    One-pass simulation using the given policy.

    Demand:
      - Always restricted to N busiest 3-hour windows.
      - demand_factor controls stress (1.0 = no extra calls, still peak-only).

    Supply:
      - ALS/BLS downsampling via als_frac / bls_frac (unchanged).
    """
    if seed is not None:
        np.random.seed(seed)

    calls = load_calls()
    units = load_units()

    # Demand stress applied across the full dataset (no windowing):
    # when demand_factor > 1.0 we clone calls from the full call set.
    calls = apply_global_demand_stress(
        calls,
        demand_factor=demand_factor,
        seed=seed,
    )

    units = apply_supply_stress(units, als_frac=als_frac, bls_frac=bls_frac, seed=seed)
    units = expand_fleet(units, fleet_factor=fleet_factor)

    policy = make_policy(policy_name)
    setattr(policy, "scenario_name", scenario_name)
    select_unit_fn = make_select_unit_fn(policy, units)
    sim = DES(select_unit_fn=select_unit_fn)

    for u in units:
        sim.add_unit(u)

    for call in calls:
        if "tmin" not in call or "_abs_epoch" not in call:
            raise RuntimeError("Calls must have 'tmin' and '_abs_epoch' fields.")
        sim.schedule(call["tmin"], "call", **call)

    while sim.advance():
        pass

    metrics = sim.metrics

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        decisions_df = pd.DataFrame(metrics["decisions"])
        decisions_path = out_dir / f"decisions_{policy_name}.parquet"
        decisions_df.to_parquet(decisions_path, index=False)

        summary = {
            "policy": policy_name,
            "n_calls": metrics["n_calls"],
            "missed_calls": metrics["missed_calls"],
            "n_decisions": len(metrics["decisions"]),
        }
        pd.DataFrame([summary]).to_csv(
            out_dir / f"summary_{policy_name}.csv", index=False
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Experimental rule-list EMS runner")
    parser.add_argument(
        "--policy",
        type=str,
        default="nearest_eta",
        help="Policy name (nearest_eta for baseline; later ALS/coverage variants)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for decisions parquet + summary CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (for on-scene/turnaround draws, transport decisions)",
    )
    parser.add_argument(
        "--als-frac",
        type=float,
        default=1.0,
        help="Fraction of ALS units to keep (1.0 = no ALS stress)",
    )
    parser.add_argument(
        "--bls-frac",
        type=float,
        default=1.0,
        help="Fraction of BLS units to keep (1.0 = no BLS stress)",
    )
    parser.add_argument(
        "--demand-factor",
        type=float,
        default=1.0,
        help="Call volume multiplier in top-N busiest 3-hour windows "
             "(1.0 = no extra calls, still peak-only).",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default="unknown",
        help="Scenario identifier for logging (e.g. S0_baseline_peaks_only)",
    )
    parser.add_argument(
        "--fleet-factor",
        type=int,
        default=1,
        help="Replication factor for 'infinite' fleet baseline (1 = normal fleet).",
    )
    args = parser.parse_args()

    metrics = run_simulation(
        policy_name=args.policy,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        seed=args.seed,
        als_frac=args.als_frac,
        bls_frac=args.bls_frac,
        demand_factor=args.demand_factor,
        scenario_name=args.scenario_name,
        fleet_factor=args.fleet_factor,
    )

    print(
        f"Done. policy={args.policy}, "
        f"als_frac={args.als_frac}, bls_frac={args.bls_frac}, "
        f"demand_factor={args.demand_factor}, "
        f"demand_factor={args.demand_factor}, fleet_factor={args.fleet_factor}, "
        f"n_calls={metrics['n_calls']}, "
        f"missed_calls={metrics['missed_calls']}, "
        f"n_decisions={len(metrics['decisions'])}"
    )


if __name__ == "__main__":
    main()
