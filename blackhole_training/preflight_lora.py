#!/usr/bin/env python3
"""Validate a generic PEFT YAML without allocating full model weights."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from peft import IA3Config, LoraConfig, PeftConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.qwen.configs import TrainingConfig
from tt_finetune.architecture import (
    describe_model,
    grouped_lora_target_geometry,
)
from tt_finetune.runtime import LoRATargetGeometry, plan_execution
from tt_finetune.targeting import resolve_canonical_role_targets


AUTO_TARGET_POLICIES = {
    "auto",
    "auto_attention",
    "auto-attention",
    "auto_qv",
    "auto-qv",
    "auto_all_linear",
    "auto-all-linear",
}


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def dataset_report(dataset):
    lengths = dataset.lengths
    return {
        "source_rows": dataset.loaded_rows,
        "accepted_rows": len(dataset),
        "dropped_over_max_length": dataset.dropped_rows,
        "min_tokens": min(lengths),
        "p50_tokens": percentile(lengths, 0.50),
        "p95_tokens": percentile(lengths, 0.95),
        "max_tokens": max(lengths),
        "coverage_rate": len(dataset) / dataset.loaded_rows,
    }


def is_attention_module(name):
    """Return whether a qualified module name is inside an attention block."""
    return any(
        marker in part.lower()
        for part in name.split(".")
        for marker in ("attn", "attention")
    )


def resolve_targets(requested, linear_modules, weight_map=None):
    if weight_map is not None:
        canonical = resolve_canonical_role_targets(
            requested,
            weight_map=weight_map,
            eligible_modules=linear_modules,
        )
        if canonical is not None:
            return list(canonical.paths), canonical.policy
    if isinstance(requested, str):
        requested = [requested]
    if len(requested) != 1 or requested[0] not in AUTO_TARGET_POLICIES:
        return requested, None

    policy = requested[0].replace("-", "_")
    if policy == "auto_all_linear":
        return ["all-linear"], policy
    if policy == "auto":
        policy = "auto_qv"
    attention_modules = {
        name: module
        for name, module in linear_modules.items()
        if is_attention_module(name)
    }
    if not attention_modules:
        raise ValueError(
            "Automatic LoRA target selection found no nn.Linear modules under "
            "an attention/attn path. Use explicit lora_target_modules."
        )

    attention_leaves = sorted(
        {name.rsplit(".", 1)[-1] for name in attention_modules}
    )
    if policy == "auto_qv":
        preferred_pairs = (
            ("q_proj", "v_proj"),
            ("query", "value"),
            ("q", "v"),
        )
        selected = next(
            ([left, right] for left, right in preferred_pairs if {left, right} <= set(attention_leaves)),
            None,
        )
        if selected is None:
            fused = [
                name
                for name in attention_leaves
                if name.lower() in {
                    "query_key_value", "qkv", "qkv_proj", "wqkv", "c_attn"
                }
            ]
            if len(fused) == 1:
                selected = fused
        if not selected:
            raise ValueError(
                f"auto_qv could not identify separate q/v or one fused QKV module. "
                f"Attention linear leaf names: {attention_leaves}"
            )
    else:
        selected = attention_leaves

    collisions = {
        leaf: [
            name
            for name in linear_modules
            if name.rsplit(".", 1)[-1] == leaf and not is_attention_module(name)
        ]
        for leaf in selected
    }
    collisions = {leaf: names for leaf, names in collisions.items() if names}
    if collisions:
        raise ValueError(
            "Automatic target leaf names also occur outside attention, so PEFT "
            f"would over-match them: {collisions}. Use explicit targets."
        )
    return selected, policy


def select_all_linear_modules(model, linear_modules):
    """Mirror PEFT's all-linear policy, excluding the model output layer."""
    output_layer = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings")
        else None
    )
    return {
        name: module
        for name, module in linear_modules.items()
        if module is not output_layer
        and name.rsplit(".", 1)[-1] != "lm_head"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--resolved-config",
        type=Path,
        help="Write a training-ready YAML with automatic targets expanded.",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name = raw.get("model_name")
    hf_config = AutoConfig.from_pretrained(
        model_name,
        revision=raw.get("model_revision"),
        trust_remote_code=raw.get("trust_remote_code", False),
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            hf_config,
            trust_remote_code=raw.get("trust_remote_code", False),
        )
    architecture, weight_map = describe_model(model)
    adapter_type = str(raw.get("adapter_type", "lora")).lower()
    if adapter_type not in {"lora", "ia3"}:
        raise ValueError("adapter_type must be lora or ia3")
    raw["adapter_type"] = adapter_type

    linear_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, Conv1D))
    }
    leaf_names = sorted({name.rsplit(".", 1)[-1] for name in linear_modules})
    requested_targets = raw.get("lora_target_modules", ["all-linear"])
    targets, target_policy = resolve_targets(
        requested_targets, linear_modules, weight_map
    )
    feedforward_targets: list[str] = []
    if adapter_type == "ia3":
        requested_feedforward = raw.get(
            "ia3_feedforward_modules", "ia3_feedforward"
        )
        feedforward_targets, _ = resolve_targets(
            requested_feedforward, linear_modules, weight_map
        )
    requested_layers_raw = raw.get("lora_layers_to_transform")
    canonical_layer_by_path = {
        item.path: item.layer_index for item in weight_map.lora_targets
    }
    if requested_layers_raw is not None:
        requested_layers_early = set(requested_layers_raw)
        invalid_layers = sorted(
            index
            for index in requested_layers_early
            if index < 0 or index >= architecture.layer_count
        )
        if invalid_layers:
            raise ValueError(
                "Requested LoRA layers are outside the canonical decoder "
                f"range 0..{architecture.layer_count - 1}: {invalid_layers}"
            )
        # Canonical semantic policies resolve to exact qualified paths. Filter
        # them before adapter import validation and serialization so the PEFT
        # metadata describes the adapter that is actually trained, rather than
        # every layer followed by a second hidden filter.
        if targets and all(target in canonical_layer_by_path for target in targets):
            targets = [
                target
                for target in targets
                if canonical_layer_by_path[target] in requested_layers_early
            ]
            if not targets:
                raise ValueError(
                    "The requested canonical roles do not exist in the selected "
                    f"decoder layers: {sorted(requested_layers_early)}"
                )
        if feedforward_targets and all(
            target in canonical_layer_by_path
            for target in feedforward_targets
        ):
            feedforward_targets = [
                target
                for target in feedforward_targets
                if canonical_layer_by_path[target] in requested_layers_early
            ]
    raw["lora_target_modules"] = targets
    if adapter_type == "ia3":
        missing_feedforward = sorted(
            set(feedforward_targets) - set(targets)
        )
        if missing_feedforward:
            raise ValueError(
                "IA3 feedforward modules must be a subset of target_modules: "
                f"{missing_feedforward}"
            )
        if not feedforward_targets:
            raise ValueError(
                "IA3 requires at least one canonical feedforward projection"
            )
        raw["ia3_feedforward_modules"] = feedforward_targets
    config = TrainingConfig.model_validate(raw)
    if config.dataset_id != "jsonl_causal":
        raise ValueError("Generic preflight expects dataset_id: jsonl_causal")

    initial_adapter = None
    if config.initial_adapter_path:
        if config.initial_adapter_name != "default":
            raise NotImplementedError(
                "Continued training supports adapter_name='default'"
            )
        loaded = PeftConfig.from_pretrained(config.initial_adapter_path)
        loaded_type = str(
            getattr(loaded, "peft_type", "LORA")
        ).split(".")[-1].lower()
        if loaded_type != adapter_type:
            raise ValueError(
                "initial_adapter_path PEFT type does not match the requested "
                f"adapter_type: {loaded_type!r} != {adapter_type!r}"
            )
        if adapter_type == "ia3":
            expected = {
                "target_modules": sorted(targets),
                "feedforward_modules": sorted(feedforward_targets),
            }
            actual = {
                "target_modules": sorted(loaded.target_modules or []),
                "feedforward_modules": sorted(
                    loaded.feedforward_modules or []
                ),
            }
        else:
            expected = {
                "r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "target_modules": sorted(targets),
                "lora_dropout": config.lora_dropout,
                "bias": config.lora_bias,
                "use_rslora": config.lora_use_rslora,
                "use_dora": config.lora_use_dora,
                "layers_to_transform": (
                    sorted(config.lora_layers_to_transform)
                    if config.lora_layers_to_transform is not None
                    else None
                ),
                "layers_pattern": config.lora_layers_pattern,
            }
            actual = {
                "r": int(loaded.r),
                "lora_alpha": int(loaded.lora_alpha),
                "target_modules": sorted(loaded.target_modules or []),
                "lora_dropout": float(loaded.lora_dropout),
                "bias": str(loaded.bias),
                "use_rslora": bool(getattr(loaded, "use_rslora", False)),
                "use_dora": bool(getattr(loaded, "use_dora", False)),
                "layers_to_transform": (
                    sorted(loaded.layers_to_transform)
                    if getattr(loaded, "layers_to_transform", None) is not None
                    else None
                ),
                "layers_pattern": getattr(loaded, "layers_pattern", None),
            }
        mismatches = {
            name: {"adapter": actual[name], "requested": value}
            for name, value in expected.items()
            if actual[name] != value
        }
        adapter_base = getattr(loaded, "base_model_name_or_path", None)
        expected_base = config.model_metadata_name or config.model_name
        # A local checkpoint has two legitimate identities at this boundary:
        # its host-visible metadata path and its container-visible model path.
        # Historical public exports recorded the latter; current exports use
        # the former.  Both point to the same mounted checkpoint and neither
        # is a model-family/backend routing signal.
        def base_identities(value):
            if not value:
                return set()
            raw = str(value)
            identities = {raw}
            path = Path(raw).expanduser()
            if path.exists():
                identities.add(str(path.resolve()))
            return identities

        accepted_base_identities = set().union(*(
            base_identities(value)
            for value in (config.model_metadata_name, config.model_name)
        ))
        adapter_base_identities = base_identities(adapter_base)
        if (
            adapter_base
            and not adapter_base_identities.intersection(
                accepted_base_identities
            )
        ):
            mismatches["base_model_name_or_path"] = {
                "adapter": str(adapter_base),
                "requested": str(expected_base),
            }
        if mismatches:
            raise ValueError(
                "Imported PEFT adapter does not match the requested training "
                "topology: " + json.dumps(mismatches, sort_keys=True)
            )
        initial_adapter = {
            "path": config.initial_adapter_path,
            "name": config.initial_adapter_name,
            "base_model_name_or_path": adapter_base,
            "topology": actual,
        }

    if targets == ["all-linear"]:
        selected = select_all_linear_modules(model, linear_modules)
    else:
        matched_targets = {
            target
            for target in targets
            if any(
                name == target or name.rsplit(".", 1)[-1] == target
                for name in linear_modules
            )
        }
        missing_targets = sorted(set(targets) - matched_targets)
        if missing_targets:
            raise ValueError(
                f"LoRA targets not found in {config.model_name}: {missing_targets}. "
                f"Available linear leaf names: {leaf_names}"
            )
        selected = {
            name: module
            for name, module in linear_modules.items()
            if name in targets or name.rsplit(".", 1)[-1] in targets
        }

    canonical_paths = {item.path for item in weight_map.lora_targets}
    selected_paths = set(selected)
    noncanonical_targets = sorted(selected_paths - canonical_paths)
    if noncanonical_targets:
        raise ValueError(
            "Adapter targets are linear modules but do not map to supported "
            "canonical decoder projections: "
            f"{noncanonical_targets}"
        )
    if config.lora_layers_to_transform is not None:
        requested_layers = set(config.lora_layers_to_transform)
        invalid_layers = sorted(
            index
            for index in requested_layers
            if index < 0 or index >= architecture.layer_count
        )
        if invalid_layers:
            raise ValueError(
                "Requested LoRA layers are outside the canonical decoder "
                f"range 0..{architecture.layer_count - 1}: {invalid_layers}"
            )
        layer_by_path = {
            item.path: item.layer_index for item in weight_map.lora_targets
        }
        selected = {
            name: module
            for name, module in selected.items()
            if layer_by_path.get(name) in requested_layers
        }
        if not selected:
            raise ValueError(
                "The requested canonical roles do not exist in the selected "
                f"decoder layers: {sorted(requested_layers)}"
            )
        selected_paths = set(selected)
    grouped_geometry = grouped_lora_target_geometry(
        weight_map, selected_paths=selected_paths
    )
    execution_plan = plan_execution(
        architecture,
        # IA3 uses the same base-architecture admission but no rank-space
        # matmuls.  R1 is a conservative compatibility proxy; the exact IA3
        # trainable size is measured from the PEFT topology below.
        rank=(config.lora_r if adapter_type == "lora" else 1),
        batch_size=config.batch_size,
        sequence_length=config.max_length,
        lora_targets=tuple(
            LoRATargetGeometry(role, input_features, output_features, instances)
            for role, input_features, output_features, instances
            in grouped_geometry
        ),
    )
    execution_plan.require_admitted()

    estimated_lora_parameters = sum(
        (config.lora_r if adapter_type == "lora" else 1)
        * (
            module.in_features + module.out_features
            if isinstance(module, nn.Linear)
            else module.weight.shape[0] + module.weight.shape[1]
        )
        for module in selected.values()
    )
    base_model_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    peft_targets = "all-linear" if targets == ["all-linear"] else targets
    if adapter_type == "ia3":
        topology_config = IA3Config(
            target_modules=peft_targets,
            feedforward_modules=feedforward_targets,
            task_type=config.lora_task_type,
            modules_to_save=config.lora_modules_to_save,
            # Validate topology on meta tensors without materialization.
            init_ia3_weights=False,
        )
    else:
        topology_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=peft_targets,
            task_type=config.lora_task_type,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            modules_to_save=config.lora_modules_to_save,
            use_rslora=config.lora_use_rslora,
            use_dora=config.lora_use_dora,
            layers_to_transform=config.lora_layers_to_transform,
            layers_pattern=config.lora_layers_pattern,
            # Validate the topology without running data-dependent
            # initialization on meta tensors.
            init_lora_weights=False,
        )
    topology_model = get_peft_model(model, topology_config)
    exact_trainable_parameters = sum(
        parameter.numel()
        for parameter in topology_model.parameters()
        if parameter.requires_grad
    )
    if targets == ["all-linear"]:
        targets = sorted(
            topology_model.peft_config["default"].target_modules
        )
        raw["lora_target_modules"] = targets
    train_dataset = get_dataset(config, "train")
    eval_dataset = get_dataset(config, "validation") if config.do_eval else None
    train = dataset_report(train_dataset)
    validation = (
        dataset_report(eval_dataset) if eval_dataset is not None else None
    )
    if train["accepted_rows"] < config.batch_size:
        raise ValueError(
            f"Only {train['accepted_rows']} train rows fit max_length, but "
            f"batch_size={config.batch_size}. The drop-last dataloader would "
            "produce zero training batches."
        )
    if validation is not None and validation["accepted_rows"] < config.batch_size:
        raise ValueError(
            f"Only {validation['accepted_rows']} validation rows fit max_length, "
            f"but batch_size={config.batch_size}. Reduce batch size, add eval "
            "rows, or disable evaluation."
        )
    train_batches_per_epoch = train["accepted_rows"] // config.batch_size
    optimizer_steps_per_epoch = math.ceil(
        train_batches_per_epoch / config.gradient_accumulation_steps
    )
    dtype_bytes = 4 if config.dtype == "torch.float32" else 2
    suggested_length = min(
        config.max_length,
        max(32, math.ceil(train["p95_tokens"] / 32) * 32),
    )
    warnings = []
    if train["coverage_rate"] < 0.95:
        warnings.append(
            f"Only {train['coverage_rate']:.1%} of loaded train rows fit "
            f"max_length={config.max_length}."
        )
    if config.max_length >= suggested_length * 2:
        warnings.append(
            f"max_length={config.max_length} is at least twice the rounded "
            f"train p95 suggestion ({suggested_length}); a shorter static shape "
            "may compile and run faster."
        )

    report = {
        "status": "ready",
        "model": config.model_name,
        "architecture": hf_config.architectures,
        "model_type": hf_config.model_type,
        "model_revision": config.model_revision,
        "trust_remote_code": config.trust_remote_code,
        "vocab_size": hf_config.vocab_size,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "dataset_format": config.dataset_format,
        "adapter_type": adapter_type,
        "lora_rank": config.lora_r if adapter_type == "lora" else None,
        "lora_alpha": config.lora_alpha if adapter_type == "lora" else None,
        "lora_targets": targets,
        "lora_target_policy": target_policy or "explicit",
        "ia3_feedforward_modules": (
            feedforward_targets if adapter_type == "ia3" else None
        ),
        "initial_adapter": initial_adapter,
        "matched_linear_modules": len(selected),
        "matched_module_types": sorted(
            {type(module).__name__ for module in selected.values()}
        ),
        "estimated_lora_parameters": estimated_lora_parameters,
        "exact_trainable_parameters": exact_trainable_parameters,
        "model_parameters": base_model_parameters,
        "trainable_fraction": (
            exact_trainable_parameters / base_model_parameters
        ),
        "estimated_adapter_weight_bytes": (
            exact_trainable_parameters * dtype_bytes
        ),
        "effective_batch_size": (
            config.batch_size * config.gradient_accumulation_steps
        ),
        "train_batches_per_epoch": train_batches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "suggested_max_length_from_train_p95": suggested_length,
        "available_linear_leaf_names": leaf_names,
        "train": train,
        "validation": validation,
        "warnings": warnings,
        "architecture_descriptor": architecture.to_dict(),
        "architecture_capabilities": execution_plan.capabilities.to_dict(),
        "canonical_weight_map": weight_map.to_dict(),
        "canonical_lora_roles": sorted(
            {item.role for item in weight_map.lora_targets if item.path in selected_paths}
        ),
        "canonical_lora_target_paths": sorted(selected_paths),
        "execution_plan": execution_plan.to_dict(),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    if args.resolved_config:
        args.resolved_config.parent.mkdir(parents=True, exist_ok=True)
        args.resolved_config.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
