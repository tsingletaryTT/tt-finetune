"""Model-level integrations for native Blackhole training kernels."""

from .grouped_peft import (
    GroupedLoRAInstallReport,
    NativeGroupedKVCoordinator,
    NativeGroupedLlamaMLP,
    install_native_grouped_lora_blocks,
)
from .scalable_projection import (
    ScalableProjectionInstallReport,
    install_scalable_native_projection_backward,
)
from .streamed_attention import (
    ATTENTION_BACKEND as STREAMED_ATTENTION_BACKEND,
    StreamedAttentionInstallReport,
    install_attention_for_plan,
    install_streamed_attention,
    streamed_attention_forward,
)
from .resident_attention import (
    ResidentAttentionInstallReport,
    install_resident_attention_backward,
)
from .architecture_runtime import (
    ArchitectureInstallReport,
    plan_and_install_architecture,
)
from .sparse_moe import (
    DeviceSparseMoELoRA,
    GenericDeviceSparseMoELoRA,
    SparseMoEInstallReport,
    deterministic_topk,
    install_sparse_moe,
    sparse_moe_regularization_loss,
)

__all__ = [
    "GroupedLoRAInstallReport",
    "NativeGroupedKVCoordinator",
    "NativeGroupedLlamaMLP",
    "ResidentAttentionInstallReport",
    "install_native_grouped_lora_blocks",
    "ScalableProjectionInstallReport",
    "install_scalable_native_projection_backward",
    "STREAMED_ATTENTION_BACKEND",
    "StreamedAttentionInstallReport",
    "install_attention_for_plan",
    "install_streamed_attention",
    "streamed_attention_forward",
    "install_resident_attention_backward",
    "ArchitectureInstallReport",
    "plan_and_install_architecture",
    "DeviceSparseMoELoRA",
    "GenericDeviceSparseMoELoRA",
    "SparseMoEInstallReport",
    "deterministic_topk",
    "install_sparse_moe",
    "sparse_moe_regularization_loss",
]
