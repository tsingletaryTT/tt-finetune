from __future__ import annotations

import pytest
import torch

from tt_finetune.runtime.resident_training_ops import (
    _normalize_resident_batch,
)


def _batch(sequence: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(sequence).reshape(1, sequence),
        "labels": (torch.arange(sequence) + 1).reshape(1, sequence),
        "attention_mask": torch.ones((1, sequence), dtype=torch.long),
        "position_ids": torch.arange(sequence).reshape(1, sequence),
    }


@pytest.mark.parametrize("logical_sequence", [32, 64, 128])
def test_compatible_logical_bucket_reuses_physical_s128_geometry(
    logical_sequence: int,
):
    source = _batch(logical_sequence)
    normalized, observed = _normalize_resident_batch(
        source,
        physical_batch=1,
        physical_sequence=128,
        ignored_index=-100,
    )

    assert observed == logical_sequence
    for name in ("input_ids", "labels", "attention_mask", "position_ids"):
        assert normalized[name].shape == (1, 128)
        assert torch.equal(
            normalized[name][:, :logical_sequence], source[name]
        )
    assert torch.count_nonzero(
        normalized["input_ids"][:, logical_sequence:]
    ) == 0
    assert torch.all(normalized["labels"][:, logical_sequence:] == -100)
    assert torch.count_nonzero(
        normalized["attention_mask"][:, logical_sequence:]
    ) == 0
    assert torch.equal(
        normalized["position_ids"], torch.arange(128).reshape(1, 128)
    )


def test_already_shifted_labels_are_reconstructed_before_bucket_padding():
    source = _batch(32)
    shifted = source["labels"][:, 1:].clone()
    source["labels"] = shifted
    normalized, _ = _normalize_resident_batch(
        source,
        physical_batch=1,
        physical_sequence=128,
        ignored_index=-100,
    )

    assert normalized["labels"][0, 0].item() == -100
    assert torch.equal(normalized["labels"][:, 1:32], shifted)
    assert torch.all(normalized["labels"][:, 32:] == -100)


@pytest.mark.parametrize("sequence", [0, 31, 33, 160])
def test_incompatible_logical_sequence_is_rejected(sequence: int):
    batch = {
        "input_ids": torch.zeros((1, sequence), dtype=torch.long),
        "labels": torch.zeros((1, sequence), dtype=torch.long),
    }
    with pytest.raises(ValueError, match="logical sequence bucket"):
        _normalize_resident_batch(
            batch,
            physical_batch=1,
            physical_sequence=128,
            ignored_index=-100,
        )


def test_noncanonical_positions_are_rejected_before_resident_execution():
    batch = _batch(32)
    batch["position_ids"][0, 16:] += 1
    with pytest.raises(ValueError, match="canonical position_ids"):
        _normalize_resident_batch(
            batch,
            physical_batch=1,
            physical_sequence=128,
            ignored_index=-100,
        )

