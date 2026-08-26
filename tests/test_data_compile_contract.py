from pathlib import Path
from types import SimpleNamespace

import torch

from blacksmith.datasets.torch.jsonl_causal import jsonl_causal_dataset
from blacksmith.datasets.torch.jsonl_causal.jsonl_causal_dataset import (
    _DynamicPaddingCollator,
    JSONLCausalDataset,
)
from tt_finetune.backends.direct import DirectBlackholeBackend
from tt_finetune.data_pipeline import PlannedBatch
from tt_finetune.runtime.autotuner import _peft_topology_rejections
from tt_finetune.runtime.loop import TrainingLoop


class _FakeDataCollator:
    def __init__(self, *, max_length, **_kwargs):
        self.max_length = max_length

    def __call__(self, batch):
        return {
            "input_ids": torch.zeros(
                (len(batch), self.max_length), dtype=torch.int64
            )
        }


def test_dynamic_padding_uses_declared_compile_bucket(monkeypatch):
    monkeypatch.setattr(
        jsonl_causal_dataset, "DataCollatorForSeq2Seq", _FakeDataCollator
    )
    collator = _DynamicPaddingCollator(
        tokenizer=object(),
        ignored_index=-100,
        max_length=128,
        bucketed=True,
        callback=None,
        boundaries=(32, 64, 128),
    )

    assert collator([{"input_ids": list(range(17))}])["input_ids"].shape[1] == 32
    assert collator([{"input_ids": list(range(63))}])["input_ids"].shape[1] == 64
    assert collator([{"input_ids": list(range(70))}])["input_ids"].shape[1] == 128


def test_training_loop_reports_the_actual_compiled_sequence_dimension():
    assert TrainingLoop._batch_sequence_length({
        "input_ids": torch.zeros((3, 96), dtype=torch.int64)
    }) == 96
    assert TrainingLoop._batch_sequence_length({
        "input_ids": [[1, 2, 3], [4, 5, 6]]
    }) == 3


def test_compile_warmup_round_robins_every_populated_bucket():
    dataset = object.__new__(JSONLCausalDataset)
    dataset.config = SimpleNamespace(
        max_length=128,
        dataset_sequence_bucketing=True,
        dataset_sequence_buckets=(32, 64, 128),
        test_config=None,
    )
    dataset._epoch_dataset = lambda _epoch: ([object()] * 6, ())
    batches = (
        PlannedBatch((0,), 64, 50, 64),
        PlannedBatch((1,), 128, 90, 128),
        PlannedBatch((2,), 32, 20, 32),
        PlannedBatch((3,), 64, 45, 64),
        PlannedBatch((4,), 128, 100, 128),
        PlannedBatch((5,), 32, 18, 32),
    )
    dataset.epoch_plan = lambda _epoch: SimpleNamespace(batches=batches)
    dataset._dataloader_from_batches = (
        lambda _rows, selected: tuple(selected)
    )

    selected = dataset.get_compile_warmup_dataloader(
        (32, 64, 128), repeats=2
    )

    assert [item.sequence_length for item in selected] == [
        32, 64, 128, 32, 64, 128
    ]
    assert dataset.compile_warmup_schedule == {
        "requested_buckets": [32, 64, 128],
        "populated_buckets": [32, 64, 128],
        "empty_buckets": [],
        "repeats": 2,
        "steps": 6,
    }


def test_native_peft_topology_is_rejected_before_silicon_measurement():
    qv = {"q", "v"}
    assert _peft_topology_rejections("native_grouped", qv)
    assert _peft_topology_rejections("native_resident_mlp", qv)
    assert _peft_topology_rejections(
        "native_resident_mlp_attention", qv
    )
    assert not _peft_topology_rejections(
        "native_grouped", {"gate", "up", "down"}
    )
    assert not _peft_topology_rejections(
        "native_resident_mlp_attention",
        {"q", "k", "v", "o", "gate", "up", "down"},
    )


def test_persistent_compile_warms_a_bounded_bucket_window(tmp_path):
    backend = object.__new__(DirectBlackholeBackend)
    backend.data_config = SimpleNamespace(
        sequence_bucketing="auto",
        sequence_buckets=None,
    )
    backend.args = SimpleNamespace(max_seq_length=128)
    backend.model = SimpleNamespace(max_seq_length=128)
    backend._ensure_persistent_service = lambda: (
        tmp_path / "runtime.sock", {"cache_keys": []}
    )
    backend._persistent_cache_id = lambda _report: "contract"
    backend._engine = SimpleNamespace(
        _to_container_path=lambda value: str(value)
    )
    observed = {}

    def execute(**kwargs):
        observed.update(kwargs)
        return {
            "step_sequence_lengths": [[32], [64], [128]],
            "persistent_runtime": {"client_cache_status": "miss"},
        }

    backend._execute_persistent = execute
    pipeline = SimpleNamespace(generated_dir=Path(tmp_path))

    report = backend._warm_persistent_profile(
        pipeline=pipeline,
        resolved=tmp_path / "resolved.yaml",
        report={},
    )

    assert observed["max_steps_override"] == 12
    assert observed["compile_sequence_buckets"] == (32, 64, 128)
    assert observed["compile_bucket_repeats"] == 4
    assert report["planned_sequence_buckets"] == [32, 64, 128]
    assert report["observed_sequence_buckets"] == [32, 64, 128]
