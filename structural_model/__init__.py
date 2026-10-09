"""Shared dam model API for solver-specific adapters."""

from Elmer.structural_model import (
    CombinedStructure,
    EnvironmentalLoads,
    FACE_GROUP_IDS,
    FoundationSupport,
    GROUP_ENTITIES,
    MaterialProperties,
    REGION_GROUP_IDS,
    REQUIRED_GROUPS,
    audit_combined_structure,
    build_combined_structure,
    build_named_groups,
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