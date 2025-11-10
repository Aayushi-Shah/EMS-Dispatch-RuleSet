from __future__ import annotations
import numpy as np
from . import config

def _hour_of_epoch(abs_epoch: float) -> int:
    return int((abs_epoch % (24 * 3600)) // 3600)

def turnaround_minutes(
    abs_epoch_now: float,
    base_min: float = None,
    scale: float = None,
    hourly_map: dict[int, float] = None,
    alpha: float = None,
    ed_load_ratio: float = 0.0,
) -> float:
    """
    Heuristic ED turnaround:
      draw = base + Gamma(k=2, theta=scale/2)
      draw *= hourly_map[hour] * (1 + alpha * ed_load_ratio)
    """
    b = config.TURNAROUND_MIN if base_min is None else base_min
    s = config.TURNAROUND_SCALE if scale is None else scale
    hm = config.TA_HOURLY if hourly_map is None else hourly_map
    a = config.TA_ALPHA if alpha is None else alpha

    hour = _hour_of_epoch(abs_epoch_now)
    hmult = hm.get(hour, 1.0)
    ed_mult = (1.0 + max(0.0, a) * max(0.0, ed_load_ratio))

    gamma_draw = np.random.gamma(2.0, s / 2.0)
    minutes = (b + gamma_draw) * hmult * ed_mult
    return float(minutes)