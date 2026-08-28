#!/usr/bin/env python3
"""Create deterministic tiny holdout checkpoints for architecture qualification.

The checkpoints deliberately use real Hugging Face model classes while keeping
the tensor sizes small enough for repeated silicon qualification.  They are
holdouts by semantic geometry, not model-name routing fixtures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    FalconConfig,
    FalconForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
)


def _save(model, tokenizer, root: Path, metadata: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(root, safe_serialization=True)
    tokenizer.save_pretrained(root)
    (root / "holdout_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--only",
        choices=("mixtral", "qwen2moe", "falcon"),
        help="Create one holdout without rewriting the other deterministic fixtures.",
    )
    args = parser.parse_args()

    torch.manual_seed(3407)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    vocab_size = len(tokenizer)

    mixtral_config = MixtralConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        router_aux_loss_coef=0.001,
        output_router_logits=False,
    )
    if args.only in (None, "mixtral"):
        _save(
            MixtralForCausalLM(mixtral_config),
            tokenizer,
            args.output_root / "mixtral-e8-k2-gqa",
            {
                "purpose": "semantic execution-compiler holdout",
                "architecture": "Mixtral",
                "attention": "GQA",
                "expert_count": 8,
                "top_k": 2,
                "shared_experts": 0,
                "seed": 3407,
            },
        )

    torch.manual_seed(3407)
    qwen_config = Qwen2MoeConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=96,
        shared_expert_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        decoder_sparse_step=1,
        router_aux_loss_coef=0.001,
        output_router_logits=False,
    )
    if args.only in (None, "qwen2moe"):
        _save(
            Qwen2MoeForCausalLM(qwen_config),
            tokenizer,
            args.output_root / "qwen2moe-e8-k2-shared-gqa",
            {
                "purpose": "semantic execution-compiler holdout",
                "architecture": "Qwen2MoE",
                "attention": "GQA",
                "expert_count": 8,
                "top_k": 2,
                "shared_experts": 1,
                "seed": 3407,
            },
        )

    torch.manual_seed(3407)
    falcon_config = FalconConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        ffn_hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=1,
        new_decoder_architecture=False,
        multi_query=True,
        parallel_attn=True,
        max_position_embeddings=128,
        alibi=False,
        bias=False,
    )
    if args.only in (None, "falcon"):
        _save(
            FalconForCausalLM(falcon_config),
            tokenizer,
            args.output_root / "falcon-mqa-fused-qkv",
            {
                "purpose": "semantic execution-compiler holdout",
                "architecture": "Falcon",
                "attention": "MQA",
                "qkv": "fused_asymmetric",
                "normalization": "LayerNorm",
                "mlp": "ordinary GELU",
                "seed": 3407,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
