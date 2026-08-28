from __future__ import annotations

import torch
from transformers.models.falcon.modeling_falcon import FalconModel

from blacksmith.models.torch.huggingface.hf_models import (
    _functional_causal_mask_with_cache_position,
    _functional_fused_qkv_split_heads,
)


def test_functional_causal_mask_matches_hf_scatter_spelling() -> None:
    for sequence in (1, 3, 8):
        cache_position = torch.arange(sequence)
        for attention_mask in (
            None,
            torch.ones((2, sequence), dtype=torch.long),
            torch.tensor(
                [[1] * sequence, [1] * max(sequence - 2, 0) + [0] * min(sequence, 2)],
                dtype=torch.long,
            ),
        ):
            expected = FalconModel._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask=attention_mask,
                sequence_length=sequence,
                target_length=sequence,
                dtype=torch.bfloat16,
                cache_position=cache_position,
                batch_size=2,
            )
            actual = _functional_causal_mask_with_cache_position(
                attention_mask=attention_mask,
                sequence_length=sequence,
                target_length=sequence,
                dtype=torch.bfloat16,
                cache_position=cache_position,
                batch_size=2,
            )
            assert torch.equal(actual, expected)


def test_functional_causal_mask_preserves_four_dimensional_input() -> None:
    mask = torch.randn(2, 1, 4, 4, dtype=torch.bfloat16)
    actual = _functional_causal_mask_with_cache_position(
        attention_mask=mask,
        sequence_length=4,
        target_length=4,
        dtype=torch.bfloat16,
        cache_position=torch.arange(4),
        batch_size=2,
    )
    assert actual is mask


def test_functional_mqa_split_matches_falcon_values_and_gradients() -> None:
    class AttentionShape:
        new_decoder_architecture = False
        multi_query = True
        num_heads = 4
        num_kv_heads = 1
        head_dim = 8

    from transformers.models.falcon.modeling_falcon import FalconAttention

    reference_input = torch.randn(2, 7, 48, requires_grad=True)
    candidate_input = reference_input.detach().clone().requires_grad_(True)
    shape = AttentionShape()
    expected = FalconAttention._split_heads(shape, reference_input)
    actual = _functional_fused_qkv_split_heads(shape, candidate_input)
    for expected_value, actual_value in zip(expected, actual, strict=True):
        assert torch.equal(expected_value, actual_value)
    sum(value.square().sum() for value in expected).backward()
    sum(value.square().sum() for value in actual).backward()
    assert torch.equal(reference_input.grad, candidate_input.grad)
