# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import warnings
import json
import math
from dataclasses import replace
from fnmatch import fnmatch
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import IA3Config, LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

from blacksmith.tools.templates.configs import TrainingConfig


_P150A_ALLOCATABLE_DRAM_BYTES = 34_217_131_008
_P150A_MEMORY_SAFETY_FRACTION = 0.88
_FROZEN_WEIGHT_EXECUTABLE_COPIES = 2.0
_RUNTIME_MEMORY_RESERVE_BYTES = 768 * 1024**2
_TT_TILE_ELEMENTS = 32 * 32
_TT_TILE_BYTES = {
    "bf16": 2048,
    "bfp_bf8": 1088,
    "bfp_bf4": 576,
}


def _tiled_weight_bytes(parameter, dtype: str) -> int:
    if parameter.ndim != 2:
        return parameter.numel() * parameter.element_size()
    rows, columns = map(int, parameter.shape)
    tiles = math.ceil(rows / 32) * math.ceil(columns / 32)
    return tiles * _TT_TILE_BYTES[dtype]


def _quantizable_frozen_weights(model) -> dict[str, torch.nn.Parameter]:
    """Find frozen matmul/linear RHS tensors without architecture names."""

    modules = dict(model.named_modules())
    output = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad or not name.endswith(".weight"):
            continue
        module_path = name.rsplit(".", 1)[0]
        module = modules.get(module_path)
        if module is None:
            continue
        # Transformers Conv1D is intentionally identified by its public class
        # semantic so this loader does not import a model-family implementation.
        if isinstance(module, torch.nn.Linear) or type(module).__name__ == "Conv1D":
            output[name] = parameter
    return output


def _load_override_config(value) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    text = str(value)
    if text in {"bfp_bf4", "bfp_bf8", "bf16"}:
        return {"default": text}
    with Path(text).open(encoding="utf-8") as source:
        return json.load(source)


def _plan_frozen_weight_dtype(model, requested: str) -> dict:
    if requested not in {"auto", "bf16", "bf8_frozen", "bf4_frozen"}:
        raise ValueError(f"unsupported memory_mode: {requested}")
    weights = _quantizable_frozen_weights(model)
    bf16_bytes = sum(_tiled_weight_bytes(value, "bf16") for value in weights.values())
    other_frozen_bytes = sum(
        parameter.numel() * parameter.element_size()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and name not in weights
    )
    trainable_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    budget = int(
        _P150A_ALLOCATABLE_DRAM_BYTES * _P150A_MEMORY_SAFETY_FRACTION
    )

    # Frozen weights are consumed by independently compiled forward and dX
    # graphs in the generic TT-XLA path. Reduced program inputs currently
    # coexist with the lazy BF16 XLA parameter backing, so a low-memory peak is
    # the BF16 source plus two cached physical BFP copies. This was measured at
    # the Qwen-7B allocator boundary; omitting the source copy made predicted-
    # fit BF8 jobs fail around 30 GiB. It is a capacity-derived model, not a
    # checkpoint identity route.
    def estimate(dtype: str) -> int:
        weight_bytes = sum(
            _tiled_weight_bytes(value, dtype) for value in weights.values()
        )
        return int(
            weight_bytes * _FROZEN_WEIGHT_EXECUTABLE_COPIES
            + (bf16_bytes if dtype != "bf16" else 0)
            + other_frozen_bytes
            + trainable_bytes * 4
            + _RUNTIME_MEMORY_RESERVE_BYTES
        )

    estimates = {
        dtype: estimate(dtype)
        for dtype in ("bf16", "bfp_bf8", "bfp_bf4")
    }
    if requested == "bf16":
        selected = "bf16"
        reason = "explicit BF16 memory contract"
    elif requested == "bf8_frozen":
        selected = "bfp_bf8"
        reason = "explicit BF8 frozen-weight memory contract"
    elif requested == "bf4_frozen":
        selected = "bfp_bf4"
        reason = "explicit BF4 frozen-weight memory contract"
    elif estimates["bf16"] <= budget:
        selected = "bf16"
        reason = "estimated BF16 execution state fits the guarded P150a budget"
    elif estimates["bfp_bf8"] <= budget:
        selected = "bfp_bf8"
        reason = "BF16 estimate exceeds the guarded budget; BF8 estimate fits"
    else:
        # Prefer a deterministic mixed native representation over reducing
        # every frozen weight to BF4. Start from BF8 and lower the largest
        # ordinary frozen matmul tensors only until the guarded budget fits.
        # The output embedding/head is retained at BF8 where it can be
        # identified semantically; logits are particularly sensitive and it
        # is unnecessary to spend that quality budget first.
        output_module = (
            model.get_output_embeddings()
            if hasattr(model, "get_output_embeddings")
            else None
        )
        modules = dict(model.named_modules())
        candidates = []
        for name, parameter in weights.items():
            module = modules.get(name.rsplit(".", 1)[0])
            if module is output_module:
                continue
            saving = (
                _tiled_weight_bytes(parameter, "bfp_bf8")
                - _tiled_weight_bytes(parameter, "bfp_bf4")
            ) * _FROZEN_WEIGHT_EXECUTABLE_COPIES
            candidates.append((int(saving), name))
        mixed_dtypes = {name: "bfp_bf8" for name in weights}
        mixed_estimate = estimates["bfp_bf8"]
        for saving, name in sorted(candidates, key=lambda item: (-item[0], item[1])):
            if mixed_estimate <= budget:
                break
            mixed_dtypes[name] = "bfp_bf4"
            mixed_estimate -= saving
        if mixed_estimate <= budget:
            selected = "mixed_bfp_bf8_bf4"
            estimates[selected] = int(mixed_estimate)
            reason = (
                "BF16 and full BF8 exceed the guarded budget; a "
                "geometry-driven BF8/BF4 frozen-weight mix fits"
            )
        else:
            selected = "bfp_bf4"
            reason = (
                "BF16, BF8, and the guarded BF8/BF4 mix exceed the budget; "
                "using the explicit lowest-memory native BF4 contract"
            )
            mixed_dtypes = {}
    return {
        "requested_mode": requested,
        "selected_weight_dtype": selected,
        "selection_reason": reason,
        "quantizable_tensor_count": len(weights),
        "quantizable_parameter_count": sum(value.numel() for value in weights.values()),
        "logical_bf16_weight_bytes": bf16_bytes,
        "other_frozen_logical_bytes": other_frozen_bytes,
        "trainable_logical_bytes": trainable_bytes,
        "guarded_dram_budget_bytes": budget,
        "allocatable_dram_bytes": _P150A_ALLOCATABLE_DRAM_BYTES,
        "estimated_execution_bytes": estimates,
        "executable_frozen_weight_copy_factor": _FROZEN_WEIGHT_EXECUTABLE_COPIES,
        "runtime_reserve_bytes": _RUNTIME_MEMORY_RESERVE_BYTES,
        "bf16_source_retained_in_reduced_modes": selected != "bf16",
        "automatic_weight_dtypes": (
            mixed_dtypes
            if selected == "mixed_bfp_bf8_bf4"
            else {name: selected for name in weights}
            if selected != "bf16"
            else {}
        ),
        "weights": weights,
    }


def _resolved_weight_overrides(model, automatic: dict, user_value) -> dict[str, str]:
    resolved = {}
    selected = automatic["selected_weight_dtype"]
    # User memory-policy overrides refine the same frozen matmul-weight set
    # admitted by the automatic planner.  In particular, ``default`` must not
    # mean every ``*.weight`` in the PEFT-wrapped model: that would annotate
    # LoRA parameters too and freeze their XLA storage through parametrization.
    eligible_names = set(automatic["weights"])
    if selected != "bf16":
        resolved.update(automatic.get("automatic_weight_dtypes", {}))
    user = _load_override_config(user_value)
    if user:
        default = user.pop("default", None)
        if default is not None:
            for name in eligible_names:
                resolved[name] = default
        for pattern, dtype in user.items():
            matches = [name for name in eligible_names if fnmatch(name, pattern)]
            if not matches:
                warnings.warn(
                    f"Weight dtype override pattern {pattern!r} matched no "
                    "eligible frozen matmul weights"
                )
            for name in matches:
                resolved[name] = dtype
    return resolved


def _is_trainable_param(model: torch.nn.Module, param_path: str) -> bool:
    """Look up a parameter by its pre-parametrize dotted path and return whether it's trainable.

    After register_parametrization the original lives at
    `<path-without-.weight>.parametrizations.<weight-name>.original`, so we try
    that location first and fall back to the original path.
    """
    module_path, param_name = param_path.rsplit(".", 1)
    try:
        module = model.get_submodule(module_path)
    except AttributeError:
        return False
    param = None
    if hasattr(module, "parametrizations") and param_name in getattr(module.parametrizations, "_modules", {}):
        param = getattr(module.parametrizations[param_name], "original", None)
    if param is None:
        param = getattr(module, param_name, None)
    return isinstance(param, torch.nn.Parameter) and param.requires_grad


def _xla_compiler_options(compile_options: dict) -> dict:
    """Translate TT backend options to the PJRT/TT-MLIR option surface."""

    output = {"enable_const_eval": compile_options["enable_const_eval"]}
    if "experimental_weight_dtype" in compile_options:
        output["experimental_weight_dtype"] = compile_options[
            "experimental_weight_dtype"
        ]
    if compile_options.get(
        "experimental_enable_dram_space_saving_optimization", False
    ):
        # PJRT's public spelling uses hyphens (the Python backend option
        # retains its established underscore spelling).
        output[
            "experimental-enable-dram-space-saving-optimization"
        ] = True
    return output


def get_model(config: TrainingConfig, device: torch.device, compile_model: bool = True):
    # This will be replaced with forge models loader, we should add adapter functions to modify the model as needed

    # Load a model
    load_kwargs = {
        "use_cache": False,
        "low_cpu_mem_usage": True,
        "revision": config.model_revision,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    if device.type == "cuda":
        # Stream weights directly to GPU to avoid CPU RAM spikes during load.
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)

    if config.use_tt:
        from tt_finetune.runtime.training_dropout import (
            config_has_training_dropout,
        )

        if (
            config_has_training_dropout(model.config)
            and hasattr(model, "set_attn_implementation")
        ):
            # Opaque SDPA dropout consumes backend RNG internally.  Eager
            # attention exposes the same semantic dropout through F.dropout,
            # where the production loop supplies deterministic device-data
            # masks keyed by micro-step.
            model.set_attn_implementation("eager")

    ia3_fp32_state = None

    # Apply training specific modifications
    # Apply LoRA if rank is specified
    if config.training_model_type in {"lora", "peft"}:
        initial_adapter_path = getattr(config, "initial_adapter_path", None)
        if initial_adapter_path:
            adapter_name = getattr(
                config, "initial_adapter_name", "default"
            )
            if adapter_name != "default":
                raise NotImplementedError(
                    "continued training materializes the selected adapter as "
                    "the default PEFT adapter; select a named adapter in the "
                    "public adapter lifecycle before compiling"
                )
            model = PeftModel.from_pretrained(
                model,
                initial_adapter_path,
                adapter_name=adapter_name,
                is_trainable=True,
            )
            model._tt_initial_adapter_report = {
                "path": str(initial_adapter_path),
                "name": adapter_name,
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "loaded": True,
            }
        else:
            model = _apply_peft_adapter(model, config)
        # Architecture admission must observe the configured execution dtype,
        # not the checkpoint's host-storage dtype.  Large BF16 jobs were
        # otherwise rejected as FP32 before frozen-weight memory planning had
        # a chance to select its memory representation.  The later .to(dtype)
        # remains as an idempotent safeguard for non-LoRA training modes.
        if config.adapter_type == "ia3":
            ia3_fp32_state = {
                name: parameter.detach().cpu().float().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and ".ia3_l." in name
            }
        model.to(config.torch_dtype())
        if config.adapter_type == "ia3":
            # IA3 vectors initialize at one.  BF16's spacing there is 1/128,
            # so ordinary 1e-4-scale AdamW updates can round away forever.
            # PEFT explicitly casts the scaled activation/result back to the
            # incoming dtype, making FP32 adapter vectors a local optimizer
            # precision policy rather than a base-model precision change.
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and ".ia3_l." in name:
                    parameter.data = ia3_fp32_state[name].to(
                        device=parameter.device,
                        dtype=torch.float32,
                    )
        lora_backend = getattr(config, "lora_backend", "auto")
        if lora_backend == "auto" and config.adapter_type == "lora":
            from tt_finetune.integration import plan_and_install_architecture

            architecture_report = plan_and_install_architecture(
                model,
                rank=int(config.lora_r),
                batch_size=int(config.batch_size),
                sequence_length=int(config.max_length),
                requested_attention_backend=str(getattr(
                    config, "tt_attention_backend", "auto"
                )),
                requested_moe_backend=str(getattr(
                    config, "tt_moe_backend", "auto"
                )),
                attention_query_chunk_tokens=int(getattr(
                    config, "tt_attention_query_chunk_tokens", 128
                )),
            )
            from tt_finetune.runtime import public_resident_admission

            resident_admission = public_resident_admission(
                config, architecture_report.plan
            )
            architecture_report = replace(
                architecture_report,
                executor_backend=resident_admission.backend,
                executor_fallback_reason=(
                    None
                    if resident_admission.admitted
                    else resident_admission.reason
                ),
            )
            model._tt_architecture_report = architecture_report
            model._tt_public_resident_admission = resident_admission
            if initial_adapter_path:
                from tt_finetune.integration.sparse_adapter_io import (
                    load_sparse_moe_adapter_extension,
                )

                sparse_adapter_report = load_sparse_moe_adapter_extension(
                    model, initial_adapter_path
                )
                model._tt_sparse_adapter_report = sparse_adapter_report
                model._tt_initial_adapter_report["sparse_moe_extension"] = (
                    sparse_adapter_report.to_dict()
                )
        elif lora_backend == "native_experimental":
            if config.adapter_type != "lora":
                raise ValueError(
                    "native_experimental is a LoRA-only backend"
                )
            from tt_finetune.integration import (
                install_native_grouped_lora_blocks,
            )

            resident_mlp = bool(
                getattr(config, "lora_resident_full_mlp_backward", False)
            )
            resident_attention = bool(
                getattr(
                    config,
                    "lora_resident_output_attention_backward",
                    False,
                )
            )
            native_report = install_native_grouped_lora_blocks(
                model,
                backend="native_experimental",
                # The accepted Blackhole path reads the ordinary logical
                # PEFT tensors directly.  It avoids the padded rank-64
                # packing dispatch and uses the exact native dX/dA tail.
                native_forward=True,
                native_backward=True,
                full_native_tail=True,
                grouped_native_tail=True,
                native_packing=False,
                direct_logical=True,
                resident_mlp_forward=resident_mlp,
                resident_full_mlp_backward=resident_mlp,
                install_attention=not resident_attention,
            )
            model._tt_native_lora_install_report = native_report
            if resident_attention:
                from tt_finetune.integration import (
                    install_resident_attention_backward,
                )

                attention_report = install_resident_attention_backward(
                    model,
                    sequence_length=int(config.max_length),
                    native_backward=True,
                    full_block_backward=True,
                    native_projection_backward=True,
                    native_output_attention_backward=True,
                )
                model._tt_resident_attention_install_report = (
                    attention_report
                )
                if attention_report.installed_count == 0:
                    warnings.warn(
                        "resident output+attention requested, but no exact "
                        "module shape was installable; using generic PEFT "
                        "attention backward"
                    )
                elif attention_report.fallbacks:
                    warnings.warn(
                        "some resident output+attention modules fell back "
                        "to the previous exact attention path: "
                        f"{attention_report.fallbacks}"
                    )
            if native_report.installed_count == 0:
                warnings.warn(
                    "native_experimental LoRA requested, but no exact "
                    "grouped transformer blocks were installable"
                )
            else:
                warnings.warn(
                    "native_experimental grouped LoRA backward is currently "
                    "validated bit-exact only for 128 flattened tokens. "
                    "Unsupported token-tile grids fail closed; use "
                    "lora_backend='generic' for unrestricted shapes."
                )
        elif lora_backend not in {"auto", "generic"}:
            raise ValueError(
                "lora_backend must be auto, generic, or "
                "native_experimental"
            )
    elif config.training_model_type == "adapters":
        _apply_adapters(model, config)
    elif config.training_model_type in {
        "token_rows",
        "token_rows_output_tied",
    }:
        _apply_token_rows(model, config)
        if config.training_model_type == "token_rows_output_tied":
            model.get_input_embeddings().new_delta.requires_grad_(False)
    else:
        warnings.warn(
            f"Unknown training_model_type '{config.training_model_type}'; "
            "falling back to full fine-tuning (all parameters trainable)."
        )
        for param in model.parameters():
            param.requires_grad = True

    if config.use_tt and (
        config_has_training_dropout(model.config)
        or float(getattr(config, "lora_dropout", 0.0)) > 0.0
    ):
        from tt_finetune.runtime.training_dropout import (
            install_explicit_training_dropout,
        )

        install_explicit_training_dropout(
            model,
            batch_size=int(config.batch_size),
            sequence_length=int(config.max_length),
        )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.to(config.torch_dtype())
    if config.adapter_type == "ia3" and ia3_fp32_state is not None:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and ".ia3_l." in name:
                parameter.data = ia3_fp32_state[name].to(
                    device=parameter.device,
                    dtype=torch.float32,
                )
    resident_public_execution = bool(
        getattr(
            getattr(model, "_tt_public_resident_admission", None),
            "admitted",
            False,
        )
    )
    if config.use_tt and not resident_public_execution:
        model.to(device)

    # The memory planner chooses reduced precision only for frozen
    # linear/matmul weights.
    # LoRA and every other trainable tensor retain the configured training
    # dtype. Per-tensor annotations must be registered before torch.compile so
    # the custom_call is visible to TT-MLIR.
    memory_plan = None
    resolved_overrides = {}
    if config.use_tt and not resident_public_execution:
        memory_plan = _plan_frozen_weight_dtype(
            model, getattr(config, "memory_mode", "auto")
        )
        resolved_overrides = _resolved_weight_overrides(
            model,
            memory_plan,
            getattr(config, "weight_dtype_overrides", None),
        )
    if config.use_tt and not resident_public_execution and resolved_overrides:
        from tt_torch import apply_weight_dtype_overrides

        applied = apply_weight_dtype_overrides(model, resolved_overrides)

        # register_parametrization does `set_(original, original)` internally,
        # which freezes XLA storage. If the target is trainable, the optimizer
        # later fails with "cannot mutate tensors with frozen storage" on the
        # first in-place update. Fail loudly here with an actionable message.
        trainable_hits = [name for name, _ in applied if _is_trainable_param(model, name)]
        if trainable_hits:
            raise RuntimeError(
                "weight_dtype_overrides matched trainable parameters, which is "
                "unsupported during training on XLA (torch parametrize freezes "
                "their storage and optimizer.step() will fail). Restrict the "
                "config to frozen weights only.\nOffending parameters:\n  - " + "\n  - ".join(trainable_hits)
            )
        memory_plan["applied_tensor_count"] = len(applied)
        memory_plan["applied_dtype_counts"] = {
            dtype: sum(1 for _, value in applied if value == dtype)
            for dtype in sorted({value for _, value in applied})
        }
    if memory_plan is not None:
        memory_plan.pop("weights", None)
        memory_plan.pop("automatic_weight_dtypes", None)
        memory_plan.setdefault("applied_tensor_count", 0)
        memory_plan.setdefault("applied_dtype_counts", {})
        memory_plan["numerical_contract"] = (
            "bf16_exact"
            if memory_plan["selected_weight_dtype"] == "bf16"
            else "deterministic_native_mixed_blockfloat_frozen_weights"
            if memory_plan["selected_weight_dtype"]
            == "mixed_bfp_bf8_bf4"
            else "deterministic_native_blockfloat_frozen_weights"
        )
        model._tt_memory_mode_report = memory_plan

    if config.use_tt and compile_model and not resident_public_execution:
        import torch_xla

        reduced_frozen_weights = bool(
            memory_plan is not None
            and memory_plan["selected_weight_dtype"] != "bf16"
        )
        # Keep frozen parameter transfers lazy until the compiled program can
        # request their final physical layouts. A global step marker in batch
        # preparation would otherwise materialize every parameter first in the
        # default tiled BF16 layout, forcing persistent BFP/row-major program
        # inputs to coexist with a full duplicate model.
        config.tt_selective_input_materialization = reduced_frozen_weights
        compile_options = {
            "tt_enable_torch_fx_fusion_pass": (
                config.tt_enable_torch_fx_fusion_pass
            ),
            "tt_legacy_compile": True,
            "tt_use_aot_autograd": False,
            # Const-eval retains its converted outputs for the executable
            # lifetime. A training step is split into several independently
            # compiled programs, so retaining multi-gigabyte frozen weights in
            # each program defeats the single-card block-float memory mode.
            # Reduced frozen weights instead enter every program as persistent
            # BFP PJRT inputs: ensure_layout packs each backing tensor once and
            # subsequent programs reuse that tensor. Ordinary BF16 execution
            # keeps the established const-eval behavior.
            "enable_const_eval": (
                config.enable_const_eval and not reduced_frozen_weights
            ),
        }
        if config.experimental_weight_dtype:
            compile_options["experimental_weight_dtype"] = config.experimental_weight_dtype
        if getattr(config, "memory_mode", "auto") != "bf16":
            # Enables TTNN lifetime reuse for large generic graphs. This pass
            # does not change arithmetic; it only shortens DRAM buffer lives.
            compile_options[
                "experimental_enable_dram_space_saving_optimization"
            ] = True

        # ``torch.compile(..., options=...)`` configures the Python TT backend
        # (legacy/AOT selection and FX passes), but TT-MLIR compilation happens
        # later through torch-xla's lazy graph executor.  Compiler options that
        # control TT-MLIR must therefore also be installed on torch-xla.  In
        # particular, merely placing ``enable_const_eval=False`` in the Dynamo
        # options dict left const-eval enabled and retained one converted copy
        # of every multi-gigabyte frozen-weight pack per executable.
        xla_compile_options = _xla_compiler_options(compile_options)
        torch_xla.set_custom_compile_options(xla_compile_options)
        if memory_plan is not None:
            memory_plan["xla_compile_options"] = dict(xla_compile_options)
        model = torch.compile(model, backend="tt", options=compile_options)

    if resident_public_execution:
        # The worker consumes the host PEFT module exactly once to construct
        # the packed resident ABI.  Moving or compiling it here would create a
        # duplicate generic graph and make exact checkpoint restoration occur
        # after packing instead of before it.
        model._tt_execution_model_state = "host_pending_resident_pack"

    return model


def _apply_peft_adapter(model, config: TrainingConfig):
    if config.adapter_type == "ia3":
        target_modules = config.lora_target_modules
        feedforward_modules = config.ia3_feedforward_modules
        return get_peft_model(
            model,
            IA3Config(
                target_modules=target_modules,
                feedforward_modules=feedforward_modules,
                task_type=config.lora_task_type,
                modules_to_save=config.lora_modules_to_save,
                init_ia3_weights=config.init_ia3_weights,
            ),
        )
    if config.adapter_type != "lora":
        raise ValueError(
            f"unsupported PEFT adapter_type={config.adapter_type!r}"
        )

    # When unfreeze_embeddings is enabled, use modules_to_save to also train
    # the embedding layer alongside LoRA adapters. This is needed for models
    # like Falcon3 that have limited language coverage - unfreezing embeddings
    # allows the model to adapt token representations for unseen languages.
    modules_to_save = config.lora_modules_to_save
    if getattr(config, "unfreeze_embeddings", False):
        modules_to_save = sorted(set((modules_to_save or []) + ["embed_tokens"]))

    target_modules = (
        "all-linear"
        if config.lora_target_modules == ["all-linear"]
        else config.lora_target_modules
    )
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        task_type=config.lora_task_type,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        modules_to_save=modules_to_save,
        use_rslora=config.lora_use_rslora,
        use_dora=config.lora_use_dora,
        layers_to_transform=config.lora_layers_to_transform,
        layers_pattern=config.lora_layers_pattern,
        init_lora_weights=config.lora_init_weights,
    )

    return get_peft_model(model, lora_config)


def _apply_lora(model, config: TrainingConfig):
    """Backward-compatible private alias."""

    return _apply_peft_adapter(model, config)


class TrainableTailEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, base_vocab_size: int):
        super().__init__()
        self.base_weight = nn.Parameter(
            embedding.weight.detach(),
            requires_grad=False,
        )
        self.new_delta = nn.Parameter(
            torch.zeros_like(embedding.weight[base_vocab_size:])
        )
        self.base_vocab_size = base_vocab_size
        self.padding_idx = embedding.padding_idx

    @property
    def weight(self):
        return self.base_weight

    def forward(self, input_ids):
        base_embeddings = F.embedding(
            input_ids,
            self.base_weight,
            padding_idx=self.padding_idx,
        )
        relative_ids = (input_ids - self.base_vocab_size).clamp(
            min=0,
            max=self.new_delta.shape[0] - 1,
        )
        delta_embeddings = F.embedding(relative_ids, self.new_delta)
        new_token_mask = (input_ids >= self.base_vocab_size).unsqueeze(-1)
        return base_embeddings + delta_embeddings * new_token_mask


class TrainableTailLMHead(nn.Module):
    def __init__(self, linear: nn.Linear, base_vocab_size: int):
        super().__init__()
        self.base_weight = nn.Parameter(
            linear.weight.detach(),
            requires_grad=False,
        )
        new_token_count = linear.weight.shape[0] - base_vocab_size
        self.new_delta = nn.Parameter(
            torch.zeros_like(linear.weight[base_vocab_size:])
        )
        selector = torch.zeros(
            new_token_count,
            linear.weight.shape[0],
            dtype=linear.weight.dtype,
        )
        selector[:, base_vocab_size:] = torch.eye(
            new_token_count,
            dtype=linear.weight.dtype,
        )
        self.register_buffer("tail_selector", selector, persistent=False)

    @property
    def weight(self):
        return self.base_weight

    def forward(self, hidden_states):
        base_logits = F.linear(hidden_states, self.base_weight)
        delta_logits = F.linear(hidden_states, self.new_delta)
        return base_logits + torch.matmul(delta_logits, self.tail_selector)


def _apply_token_rows(model, config: TrainingConfig):
    if not config.token_contract_path:
        raise ValueError("token_contract_path is required for training_model_type=token_rows")
    contract_path = Path(config.token_contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    base_vocab_size = int(contract["base_vocab_size"])
    foundation_vocab_size = int(contract["foundation_vocab_size"])
    if model.config.vocab_size != foundation_vocab_size:
        raise ValueError(
            f"Model vocab_size={model.config.vocab_size} does not match "
            f"token contract foundation_vocab_size={foundation_vocab_size}"
        )
    if base_vocab_size >= foundation_vocab_size:
        raise ValueError("Token contract does not contain appended trainable rows")

    for parameter in model.parameters():
        parameter.requires_grad = False
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if not isinstance(input_layer, nn.Embedding) or not isinstance(output_layer, nn.Linear):
        raise TypeError(
            "token_rows currently requires nn.Embedding input and nn.Linear output modules"
        )
    model.set_input_embeddings(TrainableTailEmbedding(input_layer, base_vocab_size))
    model.set_output_embeddings(TrainableTailLMHead(output_layer, base_vocab_size))
    return model


def _apply_adapters(model, config: TrainingConfig):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Apply adapters
    if len(config.adapter_layers) == 0:
        adapter_layers = list(range(len(model.model.layers)))
    else:
        adapter_layers = config.adapter_layers

    for block_idx in adapter_layers:
        #### Insert first adapter
        original_layer_output = model.model.layers[block_idx].self_attn.o_proj
        adapted_layer = make_adapted_layer(original_layer_output, config)
        model.model.layers[block_idx].self_attn.o_proj = adapted_layer

        #### Insert second adapter
        original_layer_output = model.model.layers[block_idx].mlp.down_proj
        adapted_layer = make_adapted_layer(original_layer_output, config)
        model.model.layers[block_idx].mlp.down_proj = adapted_layer

    return model


def make_adapted_layer(linear, config: TrainingConfig):
    class ResidualAdapter(nn.Module):
        def __init__(self, linear, bottleneck_dim):
            super().__init__()
            self.linear = linear
            d = linear.out_features

            self.adapter = nn.Sequential(
                nn.Linear(d, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, d),
            )

            # Start as identity
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

        def forward(self, x):
            y = self.linear(x)
            return y + self.adapter(y)

    return ResidualAdapter(linear, config.adapter_bottleneck_dim)
