# scripts/simulator/policy_base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from .des import Unit
from . import traffic, config

@dataclass
class PickResult:
    unit: Optional[Unit]
    resp_minutes: float
    meta: Dict[str, Any]

class Policy:
    """Interface. Implement select()."""
    policy_id: str = "base"
    display_name: str = "Base Policy"

    def select(self, units: List[Unit], t_min: float, call: dict) -> Tuple[Optional[Unit], float]:
        """Return (unit or None, response_minutes)."""
        raise NotImplementedError

    # Helpers every policy can use
    @staticmethod
    def eta_minutes(u: Unit, call: dict, now_abs_epoch: float) -> float:
        """Dispatch delay + heuristic road time from unit->incident."""
        mi = traffic.travel_minutes(
            u.lon, u.lat,
            call["lon"], call["lat"],
            config.SCENE_SPEED_MPH,
            now_abs_epoch
        )
        return config.DISPATCH_DELAY_MIN + float(mi)

    @staticmethod
    def now_abs(call: dict, t_min: float) -> float:
        """Absolute epoch seconds at the time of decision."""
        return float(call["_abs_epoch_start"]) + 60.0 * float(t_min)