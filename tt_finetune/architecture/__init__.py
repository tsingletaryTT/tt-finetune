"""Architecture-general discovery and capability APIs."""

from .capabilities import (
    ArchitectureCapabilityReport,
    UnsupportedCapability,
    architecture_capabilities,
)
from .descriptor import (
    CanonicalArchitectureDescriptor,
    MoEDescriptor,
    ProjectionBiases,
    RopeDescriptor,
    describe_config,
)
from .mapping import (
    CanonicalExpertLinearRole,
    CanonicalLayerMap,
    CanonicalLinearRole,
    CanonicalMoELayerMap,
    CanonicalModelMap,
    canonical_lora_target_paths,
    describe_model,
    grouped_expert_lora_target_geometry,
    grouped_lora_target_geometry,
    map_model,
)

__all__ = [
    "ArchitectureCapabilityReport",
    "CanonicalArchitectureDescriptor",
    "CanonicalExpertLinearRole",
    "CanonicalLayerMap",
    "CanonicalLinearRole",
    "CanonicalMoELayerMap",
    "CanonicalModelMap",
    "MoEDescriptor",
    "ProjectionBiases",
    "RopeDescriptor",
    "UnsupportedCapability",
    "architecture_capabilities",
    "canonical_lora_target_paths",
    "describe_config",
    "describe_model",
    "grouped_expert_lora_target_geometry",
    "grouped_lora_target_geometry",
    "map_model",
]
