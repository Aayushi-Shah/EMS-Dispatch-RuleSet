# scripts/simulator/policies/__init__.py
from __future__ import annotations

from typing import Any, Dict, Type

from scripts.policies.common import (  # noqa: F401
    BasePolicy,
    CallDict,
    PolicyResult,
    UnitLike,
    classify_urban_rural,
    classify_urban_rural_cached,
)
from scripts.policies.P1_nearest_eta import NearestETAPolicy
from scripts.policies.P2_als_bls_severity import ALSBLSSeverityPolicy
from scripts.policies.P3_coverage import CoveragePreservingPolicy
from scripts.policies.P4_fairness import UrbanRuralFairnessPolicy
from scripts.policies.P5_hybrid import HybridPolicy
from scripts.policies.feature_flag import FeatureFlagPolicy

PolicyClass = Type[BasePolicy]

POLICY_REGISTRY: Dict[str, PolicyClass] = {
    "nearest_eta": NearestETAPolicy,
    "p2_als_bls": ALSBLSSeverityPolicy,
    "p3_coverage": CoveragePreservingPolicy,
    "p4_fairness": UrbanRuralFairnessPolicy,
    "p5_hybrid": HybridPolicy,
    "feature_flag": FeatureFlagPolicy,
}

__all__ = [
    "BasePolicy",
    "CallDict",
    "PolicyResult",
    "UnitLike",
    "classify_urban_rural",
    "classify_urban_rural_cached",
    "ALSBLSSeverityPolicy",
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
