#!/usr/bin/env python3
"""Convert a Blacksmith trainable-only checkpoint into a standard PEFT adapter."""

import argparse
import json
from pathlib import Path

import torch
from peft import IA3Config, LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


_TT_SPARSE_MOE_SUFFIXES = {
    "lora_A_gate_up",
    "lora_B_gate_up",
    "lora_A_down",
    "lora_B_down",
}
_TT_GENERIC_SPARSE_NAMESPACES = {
    "routed_lora_A",
    "routed_lora_B",
    "shared_lora_A",
    "shared_lora_B",
}


def _is_tt_sparse_moe_extension(name: str) -> bool:
    """Return whether ``name`` is a canonical TT sparse-expert adapter.

    PEFT cannot currently attach LoRA directly to the three-dimensional
    parallel-expert weights used by GraniteMoE.  The runtime therefore owns
    those four expert tensors per block.  They are preserved beside the
    ordinary PEFT adapter as an explicit extension, never silently discarded.
    """

    parts = name.split(".")
    return bool(
        parts[-1] in _TT_SPARSE_MOE_SUFFIXES
        or any(part in _TT_GENERIC_SPARSE_NAMESPACES for part in parts)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--metadata-base-model",
        help=(
            "Portable model ID/path written to adapter_config.json. This "
            "prevents a container-only /workspace path leaking into a "
            "standard PEFT export."
        ),
    )
    parser.add_argument("--model-revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--slow-tokenizer", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-type", choices=("lora", "ia3"), default="lora")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=int)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", default="none")
    parser.add_argument("--modules-to-save", nargs="*")
    parser.add_argument("--use-rslora", action="store_true")
    parser.add_argument("--use-dora", action="store_true")
    parser.add_argument("--lora-plus-lr-ratio", type=float)
    parser.add_argument("--layers-to-transform", nargs="*", type=int)
    parser.add_argument("--layers-pattern")
    parser.add_argument("--init-weights", default="true")
    parser.add_argument("--feedforward-modules", nargs="*")
    parser.add_argument("--init-ia3-weights", default="true")
    parser.add_argument("--task-type", default="CAUSAL_LM")
    args = parser.parse_args()
    try:
        init_weights = json.loads(args.init_weights.lower())
    except (json.JSONDecodeError, AttributeError):
        init_weights = args.init_weights
    try:
        init_ia3_weights = json.loads(args.init_ia3_weights.lower())
    except (json.JSONDecodeError, AttributeError):
        init_ia3_weights = args.init_ia3_weights
    if args.adapter_type == "lora" and (args.rank is None or args.alpha is None):
        parser.error("--rank and --alpha are required for LoRA export")
    if args.adapter_type == "ia3" and not args.feedforward_modules:
        parser.error("--feedforward-modules is required for IA3 export")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_state = checkpoint["model_state_dict"]
    adapter_state = {
        key.removeprefix("_orig_mod."): value.to("cpu")
        for key, value in raw_state.items()
    }

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "dtype": torch.bfloat16,
        "use_cache": False,
        "revision": args.model_revision,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.adapter_type == "ia3":
        peft_config = IA3Config(
            target_modules=args.targets,
            feedforward_modules=args.feedforward_modules,
            task_type=args.task_type,
            modules_to_save=args.modules_to_save or None,
            init_ia3_weights=init_ia3_weights,
        )
    else:
        peft_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            target_modules=(
                "all-linear"
                if args.targets == ["all-linear"]
                else args.targets
            ),
            task_type=args.task_type,
            lora_dropout=args.dropout,
            bias=args.bias,
            modules_to_save=args.modules_to_save or None,
            use_rslora=args.use_rslora,
            use_dora=args.use_dora,
            layers_to_transform=args.layers_to_transform or None,
            layers_pattern=args.layers_pattern,
            init_lora_weights=init_weights,
        )
    model = get_peft_model(model, peft_config)
    model.peft_config["default"].base_model_name_or_path = (
        args.metadata_base_model or args.base_model
    )

    model_keys = set(model.state_dict())
    unexpected = sorted(set(adapter_state) - model_keys)
    tt_sparse_moe_state = {
        key: adapter_state[key]
        for key in unexpected
        if _is_tt_sparse_moe_extension(key)
    }
    unsupported_unexpected = sorted(
        set(unexpected) - set(tt_sparse_moe_state)
    )
    expected_trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trainable = sorted(expected_trainable - set(adapter_state))
    if unsupported_unexpected or missing_trainable:
        raise RuntimeError(
            "Checkpoint does not match requested PEFT topology.\n"
            "Unsupported unexpected keys "
            f"({len(unsupported_unexpected)}): {unsupported_unexpected[:10]}\n"
            f"Missing trainable keys ({len(missing_trainable)}): {missing_trainable[:10]}"
        )

    standard_adapter_state = {
        key: value
        for key, value in adapter_state.items()
        if key in model_keys
    }
    model.load_state_dict(standard_adapter_state, strict=False)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        use_fast=not args.slow_tokenizer,
    ).save_pretrained(args.output)

    # LoRA+ is an optimizer policy and therefore is not represented by PEFT's
    # adapter_config.json.  Keep the portable PEFT adapter standard while
    # recording the continuation-training contract in a sidecar understood by
    # tt-finetune and harmless to ordinary PEFT consumers.
    adapter_features = (
        ["ia3"]
        if args.adapter_type == "ia3"
        else [
            name
            for enabled, name in (
                (args.use_rslora, "rslora"),
                (args.use_dora, "dora"),
                (args.lora_plus_lr_ratio is not None, "lora_plus"),
            )
            if enabled
        ]
    )
    (args.output / "tt_finetune_training.json").write_text(
        json.dumps(
            {
                "format": "tt-finetune-training-metadata-v1",
                "base_model_name_or_path": (
                    args.metadata_base_model or args.base_model
                ),
                "base_model_revision": args.model_revision,
                "adapter_method": "+".join(adapter_features) or "lora",
                "adapter_features": adapter_features,
                "lora_plus_lr_ratio": args.lora_plus_lr_ratio,
                "layers_to_transform": args.layers_to_transform or None,
                "layers_pattern": args.layers_pattern,
                "ia3_feedforward_modules": (
                    args.feedforward_modules
                    if args.adapter_type == "ia3" else None
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if tt_sparse_moe_state:
        save_file(
            tt_sparse_moe_state,
            args.output / "tt_sparse_moe_adapter.safetensors",
        )
        extension = {
            "format": "tt-finetune-sparse-moe-v1",
            "base_model_name_or_path": args.base_model,
            "rank": args.rank,
            "alpha": args.alpha,
            "tensor_count": len(tt_sparse_moe_state),
            "parameter_count": sum(
                tensor.numel() for tensor in tt_sparse_moe_state.values()
            ),
            "standard_peft_adapter": "adapter_model.safetensors",
            "extension_weights": "tt_sparse_moe_adapter.safetensors",
            "semantics": (
                "canonical selected-only routed/shared expert LoRA; "
                "fused or separate gate/up/down storage"
            ),
            "tensor_roles": sorted(
                {
                    part
                    for name in tt_sparse_moe_state
                    for part in name.split(".")
                    if part in _TT_GENERIC_SPARSE_NAMESPACES
                    or part in _TT_SPARSE_MOE_SUFFIXES
                }
            ),
        }
        (args.output / "tt_sparse_moe_config.json").write_text(
            json.dumps(extension, indent=2) + "\n",
            encoding="utf-8",
        )

    tensor_count = len(adapter_state)
    parameter_count = sum(tensor.numel() for tensor in adapter_state.values())
    print(f"Exported {tensor_count} tensors / {parameter_count:,} parameters")
    print(f"PEFT adapter: {args.output}")
    if tt_sparse_moe_state:
        print(
            "TT sparse-MoE extension: "
            f"{len(tt_sparse_moe_state)} tensors / "
            f"{sum(t.numel() for t in tt_sparse_moe_state.values()):,} "
            "parameters"
        )


if __name__ == "__main__":
    main()
