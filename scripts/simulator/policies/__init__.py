# scripts/simulator/policies/__init__.py
from __future__ import annotations

from typing import Any, Callable, Dict, Type

from scripts.simulator.policies.common import (  # noqa: F401
    BasePolicy,
    CallDict,
    PolicyResult,
    UnitLike,
    classify_urban_rural,
    classify_urban_rural_cached,
)
from scripts.simulator.policies.nearest_eta import NearestETA
from scripts.simulator.policies.als_bls_severity import ALSBLSSeverityPolicy
from scripts.simulator.policies.coverage import CoveragePreservingETA
from scripts.simulator.policies.fairness import FairnessFirstPolicy
from scripts.simulator.policies.hybrid import HybridSeverityCoverageFairness
from scripts.simulator.policies.feature_flag import FeatureFlagPolicy

PolicyClass = Type[BasePolicy]

POLICY_REGISTRY: Dict[str, PolicyClass] = {
    "nearest_eta": NearestETA,
    "p2_als_bls": ALSBLSSeverityPolicy,
    "p3_coverage": CoveragePreservingETA,
    "coverage_preserving": CoveragePreservingETA,
    "p4_fairness": FairnessFirstPolicy,
    "fairness_first": FairnessFirstPolicy,
    "p5_hybrid": HybridSeverityCoverageFairness,
    "hybrid_all": HybridSeverityCoverageFairness,
    "feature_flag": FeatureFlagPolicy,
}

__all__ = [
    "BasePolicy",
    "CallDict",
    "PolicyResult",
    "UnitLike",
    "classify_urban_rural",
    "classify_urban_rural_cached",
    "NearestETA",
    "ALSBLSSeverityPolicy",
    "CoveragePreservingETA",
    "FairnessFirstPolicy",
    "HybridSeverityCoverageFairness",
    "FeatureFlagPolicy",
    "select_policy",
    "POLICY_REGISTRY",
]


def select_policy(name: str, **kwargs: Any):
    key = (name or "").strip().lower()

    cls = POLICY_REGISTRY.get(key)
    if cls is not None:
        return cls(**kwargs)

    raise ValueError(f"Unknown policy name: {name!r}")
