from types import SimpleNamespace
from pathlib import Path
import inspect

from tt_finetune.backends.direct import DirectBlackholeBackend
from tt_finetune.runtime import ExecutionGeometryCandidate
from tt_finetune.runtime.autotuner import (
    _default_graph_partition,
    _large_reduced_boundary_requires_fusion,
)


def _candidate(**overrides):
    values = {
        "candidate_id": "measured",
        "attention_query_block": 64,
        "attention_kv_block": 128,
        "attention_cores": 108,
        "mlp_cores": 110,
        "lora_cores": 110,
        "moe_cores": 96,
        "layer_group_size": 1,
        "rank_tile_width": 32,
        "optimizer_pack_bytes": 4 * 1024 * 1024,
        "optimizer_packing_strategy": "shape_grouped",
        "graph_partition": "forward_backward_fused",
        "activation_policy": "framework_safe",
        "frozen_weight_policy": "framework_managed",
        "moe_schedule": "dynamic_native_expert_bucketed",
        "prefetch_depth": 0,
        "double_buffered": False,
        "sequence_bucket": 128,
        "estimated_l1_peak_per_core": 256 * 1024,
        "estimated_dram_peak_bytes": 2 * 1024**3,
        "estimated_noc_bytes_per_step": 1024,
        "estimated_sync_points": 10,
        "tile_utilization": 1.0,
        "useful_token_fraction": 1.0,
        "analytical_score": 1.0,
        "admitted": True,
        "rejection_reasons": (),
        "attention_strategy": "native_streamed",
        "peft_strategy": "native_resident_mlp_attention",
    }
    values.update(overrides)
    return ExecutionGeometryCandidate(**values)


def _backend(candidate):
    backend = object.__new__(DirectBlackholeBackend)
    backend._selected_execution_candidate = candidate
    backend.args = SimpleNamespace(
        logging_steps=10,
        optimizer_layout="auto",
        optimizer_pack_bytes=1024 * 1024,
        optimizer_packing_strategy="contiguous",
        audit_training_state=True,
        profile_product_boundaries=False,
        learning_rate=2e-4,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        lr_scheduler_type="constant",
        warmup_steps=0,
        warmup_ratio=0.0,
        max_steps=20,
        num_train_epochs=1.0,
        experimental_fx_fusion=False,
        memory_mode="auto",
    )
    backend._persistent_container_path = lambda value: str(value)
    backend._compile_cache_id = lambda: "shared-shape-contract"
    return backend


def test_measured_candidate_exposes_complete_worker_abi():
    arguments = _backend(_candidate())._selected_worker_arguments()

    assert arguments == (
        "--optimizer-tile-packed",
        "--optimizer-pack-bytes", "4194304",
        "--optimizer-packing-strategy", "shape_grouped",
        "--attention-query-block", "64",
        "--attention-backend", "native",
        "--peft-execution-strategy", "native_resident_mlp_attention",
        "--moe-execution-strategy", "dynamic",
        "--split-clip-optimizer",
        "--split-optimizer-phases",
    )


def test_persistent_job_consumes_every_measured_runtime_choice():
    backend = _backend(_candidate())

    options = backend._persistent_run_options({}, Path("/tmp/xla-cache"))

    assert options["optimizer_tile_packed"] is True
    assert options["optimizer_pack_bytes"] == 4 * 1024 * 1024
    assert options["optimizer_packing_strategy"] == "shape_grouped"
    assert options["attention_query_block"] == 64
    assert options["attention_backend"] == "native"
    assert options["peft_execution_strategy"] == (
        "native_resident_mlp_attention"
    )
    assert options["moe_execution_strategy"] == "dynamic"
    assert options["split_clip_optimizer"] is True
    assert options["split_optimizer_phases"] is True
    assert "split_after_loss" not in options
    assert "split_after_forward" not in options


def test_measured_runtime_contract_changes_with_candidate_semantics():
    native = _backend(_candidate())
    safe = _backend(_candidate(
        candidate_id="safe",
        attention_strategy="auto",
        peft_strategy="framework_safe",
        moe_schedule=None,
    ))

    assert native._selected_worker_arguments() != safe._selected_worker_arguments()
    report = {
        "architecture_descriptor": {"attention": "gqa"},
        "canonical_lora_target_paths": ["layers.0.q_proj"],
        "execution_plan": {"sequence_length": 128},
    }
    assert native._persistent_cache_id(report) != safe._persistent_cache_id(
        report
    )


def test_persistent_contract_changes_with_data_compile_buckets():
    first = _backend(_candidate())
    second = _backend(_candidate())
    first.data_config = SimpleNamespace(
        sequence_bucketing=True, sequence_buckets=(32, 64, 128)
    )
    second.data_config = SimpleNamespace(
        sequence_bucketing=True, sequence_buckets=(64, 128)
    )
    report = {
        "architecture_descriptor": {"attention": "gqa"},
        "canonical_lora_target_paths": ["layers.0.q_proj"],
        "execution_plan": {"sequence_length": 128},
    }

    assert first._persistent_cache_id(report) != second._persistent_cache_id(
        report
    )


def test_runtime_fingerprint_tracks_measured_model_installation_sources():
    source = inspect.getsource(DirectBlackholeBackend._direct_fingerprint)
    for path in (
        "tt_finetune/runtime/execution_planner.py",
        "blacksmith/tools/templates/configs.py",
        "blacksmith/models/torch/huggingface/hf_models.py",
        "tt_finetune/integration/architecture_runtime.py",
        "tt_finetune/integration/streamed_attention.py",
        "tt_finetune/integration/sparse_moe.py",
        "tt_finetune/kernels/dynamic_sparse_expert.py",
        "tt_finetune/kernels/tt_lang_multi_layer_gradient_arena_schedule.py",
        "tt_finetune/kernels/cpp/attention_forward_prefix_compute.cpp",
        "tt_finetune/runtime/resident_engine.py",
    ):
        assert path in source
    assert "tt_finetune/_resident_bundles" in source
    assert "base.rglob" in source


def _partition_request(*, memory_mode, estimated_bytes, sequence=32, moe=None):
    return SimpleNamespace(
        memory_mode=memory_mode,
        estimated_device_bytes=estimated_bytes,
        architecture=SimpleNamespace(moe=moe),
        safe_plan=SimpleNamespace(
            estimated_device_bytes=estimated_bytes,
            sequence_length=sequence,
        ),
    )


def test_large_reduced_graph_uses_fragmentation_safe_fused_boundary():
    request = _partition_request(
        memory_mode="bf4_frozen",
        estimated_bytes=24 * 1024**3,
    )

    assert _large_reduced_boundary_requires_fusion(request)
    assert _default_graph_partition(request) == "forward_backward_fused"


def test_small_reduced_and_dense_bf16_keep_existing_static_boundaries():
    small = _partition_request(
        memory_mode="bf4_frozen",
        estimated_bytes=4 * 1024**3,
    )
    large_bf16 = _partition_request(
        memory_mode="bf16",
        estimated_bytes=12 * 1024**3,
    )

    assert _default_graph_partition(small) == "monolithic"
    assert _default_graph_partition(large_bf16) == "phase_split_auto"


def test_normal_train_executes_compiled_static_geometry_plan(tmp_path):
    """Ordinary train consumes one compiled ABI, never a measured cache."""

    candidate = _candidate(
        candidate_id="large-reduced-static",
        graph_partition="forward_backward_fused",
        attention_strategy="auto",
        peft_strategy="framework_safe",
        moe_schedule=None,
    )
    automatic = {
        "profile_key": "planner-only-key",
        "selection_status": "analytical_unmeasured",
        "cache_status": "disabled",
        "cache_reason": "cache disabled",
        "selected": candidate.to_dict(),
        "candidates_generated": 1,
        "candidates_admitted": 1,
        "candidates_profiled": 0,
        "candidate_audit": [candidate.to_dict()],
        "measurements": [],
        "safe_fallback": {},
        "request_contract": {"geometry": "large-reduced"},
    }
    compiled = {
        "schema": "tt-finetune-deterministic-execution-plan-v1",
        "plan_id": "cp-large-reduced",
        "semantic_fingerprint": "large-reduced-semantic",
        "execution_backend": "safe_torch_xla_autograd",
        "selection_authority": "deterministic calibrated static cost model",
        "geometry": {"rank": 8, "batch_size": 1, "sequence_length": 32},
        "graph_partition": "forward_backward_fused",
        "memory_policy": {},
        "component_policies": {},
        "numerical_contract": {},
        "execution_candidate": candidate.to_dict(),
        "worker_arguments": [],
        "request_contract": {"geometry": "large-reduced"},
        "online_candidate_search": False,
        "measured_profile_cache_consulted": False,
    }
    backend = object.__new__(DirectBlackholeBackend)
    backend.args = SimpleNamespace(auto_tune=True)
    backend._memory_support = lambda: SimpleNamespace(
        automatic_plan=automatic,
        compiled_plan=compiled,
        execution_path="safe_torch_xla_autograd",
    )
    backend._direct_fingerprint = lambda: "runtime-source"
    backend._engine = SimpleNamespace(
        _engine_fingerprint=lambda: "compiler-source"
    )

    profile = backend._autotune(
        report={},
        resolved=tmp_path / "resolved.yaml",
        pipeline=SimpleNamespace(generated_dir=tmp_path / "generated"),
    )

    assert profile is not None
    assert profile["selection_status"] == "deterministic_compiled"
    assert profile["cache_status"] == "not_consulted"
    assert profile["measurements"] == []
    assert profile["qualification"]["measured"] is False
    assert profile["qualification"]["online_candidate_search"] is False
    assert backend._selected_execution_candidate == candidate
    assert backend._selected_worker_arguments()[-2:] == (
        "--split-clip-optimizer",
        "--split-optimizer-phases",
    )
    assert "--split-after-forward" not in (
        backend._selected_worker_arguments()
    )
    assert "--auto-split-after-forward" not in (
        backend._selected_worker_arguments()
    )


def test_qualified_resident_train_compiles_one_deterministic_plan(tmp_path):
    """Normal resident training never resurrects a measured candidate ABI."""

    candidate = _candidate(candidate_id="irrelevant-measured-candidate")
    automatic = {
        "request_contract": {"geometry": "smol-r8-b1-s128"},
        "selected": candidate.to_dict(),
    }
    compiled = {
        "schema": "tt-finetune-deterministic-execution-plan-v1",
        "plan_id": "cp-smol-resident",
        "semantic_fingerprint": "smol-semantic-fingerprint",
        "execution_backend": "resident_exact",
        "selection_authority": "qualified semantic resident contract",
        "geometry": {"rank": 8, "batch_size": 1, "sequence_length": 128},
        "graph_partition": "resident_stack_five_call",
        "memory_policy": {},
        "component_policies": {},
        "numerical_contract": {},
        "execution_candidate": None,
        "worker_arguments": [],
        "request_contract": {"geometry": "smol-r8-b1-s128"},
        "online_candidate_search": False,
        "measured_profile_cache_consulted": False,
    }
    backend = object.__new__(DirectBlackholeBackend)
    backend.args = SimpleNamespace(auto_tune=False)
    backend._selected_execution_candidate = candidate
    backend._memory_support = lambda: SimpleNamespace(
        automatic_plan=automatic,
        compiled_plan=compiled,
        execution_path="resident_exact",
        rank=8,
        batch_size=1,
        sequence_length=128,
    )
    backend._direct_fingerprint = lambda: "resident-source-hash"
    backend._engine = SimpleNamespace(_engine_fingerprint=lambda: "compiler")

    profile = backend._autotune(
        report={
            "architecture_descriptor": {
                "semantic_fingerprint": "smol-semantic-fingerprint"
            }
        },
        resolved=tmp_path / "resolved.yaml",
        pipeline=SimpleNamespace(generated_dir=tmp_path / "generated"),
    )

    assert profile is not None
    assert profile["selection_status"] == "deterministic_compiled"
    assert profile["cache_status"] == "not_consulted"
    assert profile["measurements"] == []
    assert profile["selected"]["execution_backend"] == "resident_exact"
    assert profile["selected"]["runtime_source_hash"] == (
        "resident-source-hash"
    )
    assert profile["qualification"]["online_candidate_search"] is False
    assert (
        profile["qualification"]["measured_profile_cache_consulted"]
        is False
    )
    assert backend._selected_execution_candidate is None
