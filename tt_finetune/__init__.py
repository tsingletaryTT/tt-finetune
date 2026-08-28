"""Notebook-first fine-tuning on Tenstorrent Blackhole."""

from ._runtime_bootstrap import activate_bundled_runtime

activate_bundled_runtime()

from .config import (
    AdapterConfig,
    CompileBucket,
    DataConfig,
    IA3Config,
    LoRAConfig,
    PEFTAdapterConfig,
    SFTConfig,
    TrainingArguments,
)
from .capabilities import SupportResult
from .adapters import AdapterInfo, inspect_adapter, merge_adapter
from .data import DatasetSource
from .environment import RuntimeEnvironment
from .model import (
    AdapterPlan,
    BlackholeModel,
    FastLanguageModel,
    FastTTModel,
    TokenizerHandle,
)
from .errors import (
    TTCompileError,
    TTDependencyError,
    TTDeviceError,
    TTFinetuneError,
    TTInvalidLoRATargetError,
    TTOutOfMemoryError,
    TTRuntimeError,
    TTUnsupportedGeometryError,
    TTUnsupportedModelError,
)
from .foundation import build_token_foundation
from .multi_adapter import (
    AdapterStage,
    MultiAdapterSFTTrainer,
    MultiAdapterTrainResult,
    TokenWarmupConfig,
)
from .trainer import SFTTrainer, TTTrainer

__all__ = [
    "BlackholeModel",
    "AdapterInfo",
    "AdapterConfig",
    "AdapterPlan",
    "AdapterStage",
    "build_token_foundation",
    "CompileBucket",
    "DataConfig",
    "DatasetSource",
    "FastLanguageModel",
    "FastTTModel",
    "IA3Config",
    "LoRAConfig",
    "PEFTAdapterConfig",
    "MultiAdapterSFTTrainer",
    "MultiAdapterTrainResult",
    "SFTConfig",
    "SFTTrainer",
    "TTTrainer",
    "RuntimeEnvironment",
    "TokenizerHandle",
    "TokenWarmupConfig",
    "TrainingArguments",
    "SupportResult",
    "inspect_adapter",
    "merge_adapter",
    "TTCompileError",
    "TTDependencyError",
    "TTDeviceError",
    "TTFinetuneError",
    "TTInvalidLoRATargetError",
    "TTOutOfMemoryError",
    "TTRuntimeError",
    "TTUnsupportedGeometryError",
    "TTUnsupportedModelError",
]

__version__ = "0.2.1rc1"
