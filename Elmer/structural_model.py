"""Solver-independent model API implemented by the current procedural mesh code."""

from .regions.combined_structure import (
    CombinedStructure,
    EnvironmentalLoads,
    FoundationSupport,
    MaterialProperties,
    audit_combined_structure,
    build_combined_structure,
    build_named_groups,
)
from .regions.model_groups import (
    FACE_GROUP_IDS,
    GROUP_ENTITIES,
    REGION_GROUP_IDS,
    REQUIRED_GROUPS,
)

__all__ = [
    "CombinedStructure",
    "EnvironmentalLoads",
    "FACE_GROUP_IDS",
    "FoundationSupport",
    "GROUP_ENTITIES",
    "MaterialProperties",
    "REGION_GROUP_IDS",
    "REQUIRED_GROUPS",
    "audit_combined_structure",
    "build_combined_structure",
    "build_named_groups",
]