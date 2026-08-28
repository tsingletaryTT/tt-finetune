"""Exact N-layer resident forward/backward schedule with packed parameters.

This generalizes the accepted two-layer dataflow without expanding the public
ABI by 34 tensors per layer.  One shape-template block compiles the physical
programs; all actual frozen tensors are page views of one packed DRAM arena and
all LoRA tensors are page views of the packed optimizer state.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .optimizer_gradient_arena_planner import (
    plan_multi_layer_optimizer_gradient_arena,
)
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_arena_page_copy import (
    build_tt_lang_arena_page_copy,
    build_tt_lang_linear_packed_page_copy,
)
from .tt_lang_program_composition import (
    compile_serialized_child,
    compile_serialized_sequence_children,
    expand_serialized_child_bindings,
    make_serialized_packed_parameter_unpack_bitwise,
    make_serialized_page_fill_zero,
    remap_serialized_child,
    retarget_full_attention_gradient_outputs,
    retarget_full_mlp_gradient_outputs,
    retarget_tensor_reads_to_packed_pages,
    retarget_tensor_writes_to_packed_pages,
    serialized_child_tensor_ids,
)
from .tt_lang_resident_layer_autograd_split_owner import (
    build_tt_lang_resident_layer_autograd_split_owner,
)
from .tt_lang_resident_layer_forward_trainable_owner import (
    build_tt_lang_resident_layer_forward_trainable_owner,
)
from .tt_lang_resident_layer_forward_compact_owner import (
    _make_all_worker_barrier,
    build_tt_lang_resident_layer_forward_compact_owner,
)
from .tt_lang_semaphore_scrub import build_tt_lang_semaphore_scrub
from .tt_lang_two_layer_optimizer_arena_schedule import (
    _is_full_attention_writer,
)


ttl = None
tt_lang_operation = None


_ADAPTER_BLOCK_NAMES = {
    2: "q_a", 3: "q_b", 6: "k_a", 7: "k_b",
    10: "v_a", 11: "v_b", 13: "o_a", 14: "o_b",
    20: "gate_a", 21: "gate_b", 22: "up_a", 23: "up_b",
    24: "down_a", 25: "down_b",
}

_BACKWARD_LOCAL_TO_BLOCK = {
    2: 18, 3: 26, 7: 27, 8: 14, 9: 13, 10: 15, 11: 16,
    12: 28, 13: 3, 14: 2, 15: 29, 16: 7, 17: 6,
    18: 30, 19: 11, 20: 10, 21: 31, 22: 25, 23: 24,
    24: 32, 25: 21, 26: 20, 27: 33, 28: 23, 29: 22,
}


def _dump_composed_sequence(children: list[dict], *, label: str) -> None:
    """Optionally persist the final, post-retarget TT-Metal sources.

    Per-operation source dumps happen before the multi-layer owner rewrites
    tensor accessors and final arena destinations.  Exactness failures at that
    boundary therefore need an opt-in view of the actual program sequence sent
    to PJRT.  This is diagnostic-only and has no effect unless the environment
    variable is set.
    """

    root = os.environ.get("TT_FINETUNE_COMPOSED_SOURCE_DUMP_DIR")
    if not root:
        return
    dump = Path(root) / label
    dump.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for program_index, child in enumerate(children):
        kernels = child.get("kernels", ())
        entry: dict[str, object] = {
            "program": program_index,
            "kernel_count": len(kernels),
            "num_tensors": child.get("num_tensors"),
            "num_pipe_sync_semaphores": child.get(
                "num_pipe_sync_semaphores"
            ),
            "kernels": [],
        }
        kernel_entries = entry["kernels"]
        assert isinstance(kernel_entries, list)
        for kernel_index, kernel in enumerate(kernels):
            thread = kernel.get("thread_type", "unknown")
            filename = (
                f"program_{program_index:03d}_kernel_{kernel_index:02d}_"
                f"{thread}.cpp"
            )
            (dump / filename).write_text(
                kernel.get("cpp_source", ""), encoding="utf-8"
            )
            kernel_entries.append({
                "kernel": kernel_index,
                "thread_type": thread,
                "tensor_indices": kernel.get("tensor_indices", ()),
                "source": filename,
            })
        manifest.append(entry)
    (dump / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

def _remap_used(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


def _make_pure_semaphore_scrub(program: dict) -> dict:
    """Remove the synthetic CB relay from an epoch-boundary scrub.

    ``build_tt_lang_semaphore_scrub`` historically copied one tile through
    CB0/CB1 so TT-Lang would retain the operation.  Artifact-v4 sequences
    retain the physical program explicitly, so that relay is unnecessary.
    More importantly, a long resident sequence can enter this child with
    CB0/CB1 counters left by a differently shaped preceding program.  The
    synthetic reserve/wait then deadlocks before the scrub can establish the
    next clean semaphore epoch.

    Keep the qualified full-grid reader configuration and semaphore
    allocations, but replace its body with a data-movement-only scrub.  It
    has no CB producer/consumer state and therefore remains safe regardless
    of the immediately preceding program's circular-buffer topology. CB
    lifecycle remains owned by the physical program that configured it.
    """

    import copy

    result = copy.deepcopy(program)
    kernels = result.get("kernels", [])
    reader_candidates = [
        kernel for kernel in kernels
        if kernel.get("thread_type") == "noc"
        and "clear every" in kernel.get("cpp_source", "")
    ]
    if len(reader_candidates) != 1:
        raise ValueError(
            "semaphore scrub physical kernel set changed: "
            f"readers={len(reader_candidates)}"
        )
    reader = reader_candidates[0]
    reader["tensor_indices"] = []
    reader["cpp_source"] = r'''#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // A serialized program boundary drains the preceding kernels.  Clear all
    // resident mailboxes without touching any circular buffer: CB identities
    // and counters are program-local contracts and can differ at this point.
    for (uint32_t semaphore = 0; semaphore < 16; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }
}
'''
    result["kernels"] = [reader]
    result["cb_configs"] = []
    result.pop("operand_indices", None)
    return result


def _retarget_layer_parameter_reads(
    program: dict,
    *,
    layer: int,
    local_to_block: dict[int, int],
    block_page_offsets: tuple[tuple[int, ...], ...],
    gradient_plan,
    packed_frozen_tensor: int,
    packed_state_tensor: int,
    pack_frozen: bool = True,
    pack_trainable: bool = True,
) -> dict:
    frozen = {
        local: block_page_offsets[layer][block]
        for local, block in local_to_block.items()
        if block not in _ADAPTER_BLOCK_NAMES
    }
    trainable = {
        local: (
            gradient_plan.slice(layer, _ADAPTER_BLOCK_NAMES[block]).page_offset
            - gradient_plan.scratch_pages
        )
        for local, block in local_to_block.items()
        if block in _ADAPTER_BLOCK_NAMES
    }
    result = program
    if pack_frozen:
        result = retarget_tensor_reads_to_packed_pages(
            result,
            packed_tensor=packed_frozen_tensor,
            tensor_page_offsets=frozen,
        )
    if pack_trainable:
        result = retarget_tensor_reads_to_packed_pages(
            result,
            packed_tensor=packed_state_tensor,
            tensor_page_offsets=trainable,
        )
    return result


@lru_cache(maxsize=8)
def build_tt_lang_multi_layer_gradient_arena_schedule(
    *,
    layer_count: int,
    block_page_offsets: tuple[tuple[int, ...], ...],
    source_bundle: str,
    sequence_length: int,
    hidden_features: int,
    intermediate_features: int,
    kv_features: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_rank: int,
    lora_scale: float,
    attention_scale: float,
    rmsnorm_epsilon: float,
    has_bias: bool = False,
    active_lora_roles: tuple[str, ...] = (
        "q", "k", "v", "o", "gate", "up", "down",
    ),
    debug_forward_checkpoint_only: bool = False,
    debug_forward_checkpoint_layer: int = -1,
    debug_forward_checkpoint_separate_parameters: bool = False,
    debug_forward_checkpoint_public_destination: bool = False,
    debug_top_hidden_input_only: bool = False,
    debug_top_stop_after_mlp: bool = False,
    debug_top_stop_after_post_extract: bool = False,
    debug_top_stop_after_post_residual: bool = False,
    debug_top_stop_after_post_backward: bool = False,
    debug_top_stop_after_attention_input: bool = False,
    debug_top_stop_after_attention: bool = False,
    debug_top_attention_stage: str | None = None,
    debug_top_full_backward: bool = False,
    debug_top_attention_parameter_source: str = "packed",
    debug_top_gradient_output_layer: int = -1,
    debug_top_gradient_output_prefix: int = 6,
    debug_reverse_layer_limit: int = -1,
    reverse_handoff_storage: str = "private_l1",
    reuse_backward_scratch: bool | int = False,
    compact_forward: bool = False,
    canonical_exact_attention: bool = False,
    logical_rank: int | None = None,
    o_lora_k_tiles_per_phase: int | None = None,
    execution_phase: str = "full",
    backward_direct_public_checkpoints: bool = False,
) -> Callable:
    if layer_count < 2:
        raise ValueError("multi-layer schedule requires at least two layers")
    if execution_phase not in ("full", "forward", "backward"):
        raise ValueError("execution phase must be 'full', 'forward' or 'backward'")
    if execution_phase == "forward" and any((
        debug_top_hidden_input_only,
        debug_top_stop_after_mlp,
        debug_top_stop_after_post_extract,
        debug_top_stop_after_post_residual,
        debug_top_stop_after_post_backward,
        debug_top_stop_after_attention_input,
        debug_top_stop_after_attention,
        debug_top_attention_stage is not None,
        debug_top_full_backward,
        debug_reverse_layer_limit > 0,
    )):
        raise ValueError("backward diagnostics require a backward-capable phase")
    if execution_phase == "backward" and debug_forward_checkpoint_only:
        raise ValueError("forward checkpoint diagnostics require a forward phase")
    if physical_rank != 32:
        raise ValueError("multi-layer packed schedule requires physical rank 32")
    canonical_roles = {"q", "k", "v", "o", "gate", "up", "down"}
    active_lora_roles = tuple(dict.fromkeys(active_lora_roles))
    if not active_lora_roles or not set(active_lora_roles) <= canonical_roles:
        raise ValueError("active LoRA roles must be a non-empty canonical subset")
    if canonical_exact_attention and not compact_forward:
        raise ValueError(
            "canonical exact attention requires the compact forward owner"
        )
    if canonical_exact_attention and (
        logical_rank is None or not 1 <= logical_rank <= physical_rank
    ):
        raise ValueError(
            "canonical exact attention logical rank must fit the physical rank"
        )
    if len(block_page_offsets) != layer_count or any(
        len(offsets) != 34 for offsets in block_page_offsets
    ):
        raise ValueError("need exactly 34 packed page offsets per layer")
    if any(offset < 0 for offsets in block_page_offsets for offset in offsets):
        raise ValueError("packed block offsets must be non-negative")
    if debug_forward_checkpoint_layer < -1 or debug_forward_checkpoint_layer >= layer_count:
        raise ValueError("forward checkpoint layer is outside the owner depth")
    if debug_forward_checkpoint_layer >= 0 and not debug_forward_checkpoint_only:
        raise ValueError("checkpoint layer selection requires checkpoint diagnostic")
    if debug_reverse_layer_limit == 0 or debug_reverse_layer_limit > layer_count:
        raise ValueError("reverse layer limit must be -1 or within owner depth")
    if (
        debug_top_gradient_output_layer < -1
        or debug_top_gradient_output_layer >= layer_count
    ):
        raise ValueError("top gradient output layer is outside the owner depth")
    if debug_top_gradient_output_layer >= 0 and not debug_top_stop_after_mlp:
        raise ValueError(
            "top gradient output layer selection requires the top-MLP diagnostic"
        )
    if debug_top_gradient_output_prefix < 0 or debug_top_gradient_output_prefix > 6:
        raise ValueError("top gradient output prefix must be between zero and six")
    if debug_top_gradient_output_prefix != 6 and debug_top_gradient_output_layer < 0:
        raise ValueError(
            "partial top gradient output selection requires an output layer"
        )
    if (
        debug_forward_checkpoint_separate_parameters
        and not debug_forward_checkpoint_only
        and not canonical_exact_attention
    ):
        raise ValueError("separate forward parameters require checkpoint diagnostic")
    if debug_forward_checkpoint_public_destination and not debug_forward_checkpoint_only:
        raise ValueError("public forward destination requires checkpoint diagnostic")
    checkpoint_diagnostic_layer = (
        layer_count - 1
        if debug_forward_checkpoint_layer < 0
        else debug_forward_checkpoint_layer
    )
    if (
        debug_forward_checkpoint_public_destination
        and checkpoint_diagnostic_layer != layer_count - 1
    ):
        raise ValueError("public checkpoint destination requires the final layer")
    parameter_sources = {
        "packed", "separate_frozen", "separate_trainable", "separate_all",
    }
    if debug_top_attention_parameter_source not in parameter_sources:
        raise ValueError(
            "top attention parameter source must be one of "
            f"{sorted(parameter_sources)}"
        )
    if reverse_handoff_storage not in (
        "private_l1", "private_dram", "gradient_arena", "public_final_inplace",
        "top_public_then_private",
    ):
        raise ValueError(
            "reverse handoff storage must be 'private_l1', 'private_dram', "
            "'gradient_arena', 'public_final_inplace' or "
            "'top_public_then_private'"
        )
    if (
        debug_top_attention_parameter_source != "packed"
        and not (debug_top_full_backward or debug_top_stop_after_mlp)
    ):
        raise ValueError(
            "separate top parameters require a full-top or top-MLP diagnostic"
        )
    diagnostic_count = sum((
        debug_forward_checkpoint_only,
        debug_top_hidden_input_only,
        debug_top_stop_after_mlp,
        debug_top_stop_after_post_extract,
        debug_top_stop_after_post_residual,
        debug_top_stop_after_post_backward,
        debug_top_stop_after_attention_input,
        debug_top_stop_after_attention,
        debug_top_attention_stage is not None,
        debug_top_full_backward,
        debug_reverse_layer_limit > 0,
    ))
    if diagnostic_count > 1:
        raise ValueError(
            "multi-layer diagnostics are mutually exclusive"
        )

    plan = plan_multi_layer_optimizer_gradient_arena(
        layer_count=layer_count,
        tokens=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        physical_rank=physical_rank,
    )
    forward_builder = (
        build_tt_lang_resident_layer_forward_compact_owner
        if compact_forward
        else build_tt_lang_resident_layer_forward_trainable_owner
    )
    exact_forward_kwargs = (
        {
            "canonical_native_o_path": True,
            "exact_attention_tail": True,
            # The fixed causal-mask graph converts expanded Q/K/V to FP32,
            # performs both batched matmuls and softmax in FP32, and packs the
            # head output to BF16 only after P@V.  Keep that observed graph
            # contract; selecting the BF16 core changes the reduction tree.
            "canonical_bf16_attention": False,
            "canonical_grouped_qkv_rank": bool(
                not has_bias and logical_rank == 8
            ),
            "canonical_separate_qkv_rank": False,
            "logical_rank": logical_rank,
        }
        if canonical_exact_attention else {}
    )
    topology_forward_kwargs = (
        {"active_lora_roles": active_lora_roles}
        if compact_forward else {}
    )
    forward0 = forward_builder(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        has_bias=has_bias,
        input_rmsnorm_epsilon=rmsnorm_epsilon,
        post_attention_rmsnorm_epsilon=rmsnorm_epsilon,
        hidden_page_base=0,
        publish_layer_output_at_page_zero=canonical_exact_attention,
        o_lora_k_tiles_per_phase=o_lora_k_tiles_per_phase,
        **topology_forward_kwargs,
        **exact_forward_kwargs,
    )
    forward_later = forward_builder(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        has_bias=has_bias,
        input_rmsnorm_epsilon=rmsnorm_epsilon,
        post_attention_rmsnorm_epsilon=rmsnorm_epsilon,
        hidden_page_base=forward0.layer_output_page_offset,
        publish_layer_output_at_page_zero=canonical_exact_attention,
        o_lora_k_tiles_per_phase=o_lora_k_tiles_per_phase,
        **topology_forward_kwargs,
        **exact_forward_kwargs,
    )
    forward_internal_specs = (
        tuple(
            dict(spec) for spec in getattr(
                forward0, "internal_tensor_specs", ()
            )
        )
        if compact_forward else ()
    )
    qwen_prefix_internal_indices = tuple(
        int(index) for index in getattr(
            forward0, "qwen_grouped_prefix_internal_indices", ()
        )
    ) if compact_forward else ()
    if compact_forward and not forward_internal_specs:
        forward_internal_specs = (
            {"source_operand": 0, "memory": "l1"},
            {"source_operand": 0, "memory": "l1"},
            {"source_operand": 0, "memory": "l1"},
        )
    parameter_unpack = (
        build_tt_lang_linear_packed_page_copy(
            rows=hidden_features,
            columns=hidden_features,
            source_page_offset=0,
        )
        if canonical_exact_attention else None
    )
    hidden_copy = build_tt_lang_arena_page_copy(
        rows=sequence_length,
        columns=hidden_features,
        source_page_offset=forward0.layer_output_page_offset,
    )
    checkpoint_copy = build_tt_lang_arena_page_copy(
        rows=forward0.output_shape[0],
        columns=forward0.output_shape[1],
        source_page_offset=(checkpoint_diagnostic_layer // 2)
        * forward0.layer_allocated_pages,
    )
    checkpoint_zero_copy = build_tt_lang_arena_page_copy(
        rows=forward0.output_shape[0],
        columns=forward0.output_shape[1],
        source_page_offset=0,
    )
    checkpoint_even_import = build_tt_lang_arena_page_copy(
        rows=((layer_count + 1) // 2) * forward0.output_shape[0],
        columns=forward0.output_shape[1],
        source_page_offset=0,
    )
    checkpoint_odd_import = build_tt_lang_arena_page_copy(
        rows=(layer_count // 2) * forward0.output_shape[0],
        columns=forward0.output_shape[1],
        source_page_offset=0,
    )
    top_mlp_dx_copy = build_tt_lang_arena_page_copy(
        rows=sequence_length,
        columns=hidden_features,
        source_page_offset=0,
    )
    backward = build_tt_lang_resident_layer_autograd_split_owner(
        source_bundle=source_bundle,
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        rmsnorm_epsilon=rmsnorm_epsilon,
        pre_rms_copy=has_bias,
        debug_attention_stop_after=(
            "post_extract" if debug_top_stop_after_post_extract
            else "post_residual" if debug_top_stop_after_post_residual
            else "post_backward" if debug_top_stop_after_post_backward
            else "attention_input" if debug_top_stop_after_attention_input
            else debug_top_attention_stage
            if debug_top_attention_stage is not None
            else "attention" if debug_top_stop_after_attention
            else None
        ),
    )
    backward_private_count = int(backward.compiler_owned_l1_tensors)
    backward_private_specs = tuple(
        {
            **dict(spec),
            "source_operand": {0: 1, 1: 0, 4: 2}[
                int(spec.get("source_operand", 0))
            ],
        }
        for spec in getattr(backward, "internal_tensor_specs", ())
    )
    backward_handoff_shape = {
        key: value
        for key, value in (
            backward_private_specs[0].items()
            if backward_private_specs else ()
        )
        if key not in ("source_operand", "memory")
    }
    semaphore_scrub = build_tt_lang_semaphore_scrub()

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("multi-layer schedule requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
        t50, t51, t52,
    ):
        source = ttl.make_dataflow_buffer_like(t00, shape=(1, 1), block_count=1)
        target = ttl.make_dataflow_buffer_like(t51, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(t00[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as output:
                    output.store(tile)

        @ttl.datamovement()
        def write():
            with target.wait() as output:
                transaction = ttl.copy(output, t51[0:1, 0:1])
                transaction.wait()

    if isinstance(reuse_backward_scratch, bool):
        backward_scratch_group_count = (
            1 if reuse_backward_scratch else layer_count
        )
    else:
        backward_scratch_group_count = int(reuse_backward_scratch)
        if not 1 <= backward_scratch_group_count <= layer_count:
            raise ValueError(
                "backward scratch groups must be within model depth"
            )
    variant = (
        f"{Path(source_bundle).name}.l{layer_count}."
        f"{sequence_length}x{hidden_features}.i{intermediate_features}."
        f"kv{kv_features}.r{physical_rank}.lr{logical_rank}."
        f"bias{int(has_bias)}."
        f"targets{'-'.join(active_lora_roles)}."
        f"forwarddiag{int(debug_forward_checkpoint_only)}."
        f"forwardlayer{checkpoint_diagnostic_layer}."
        f"forwardsepparams{int(debug_forward_checkpoint_separate_parameters)}."
        f"forwardpublic{int(debug_forward_checkpoint_public_destination)}."
        f"checkpointzerofill2."
        f"tophidden{int(debug_top_hidden_input_only)}."
        f"topmlp{int(debug_top_stop_after_mlp)}."
        f"topextract{int(debug_top_stop_after_post_extract)}."
        f"topresidual{int(debug_top_stop_after_post_residual)}."
        f"toppostbw{int(debug_top_stop_after_post_backward)}."
        f"topattnin{int(debug_top_stop_after_attention_input)}.packedckpt.v18"
        f"topattn{int(debug_top_stop_after_attention)}."
        f"topattnstage{debug_top_attention_stage or 'none'}."
        f"topfull{int(debug_top_full_backward)}.packedckpt."
        f"reverselimit{debug_reverse_layer_limit}."
        f"params{debug_top_attention_parameter_source}."
        f"topgradlayer{debug_top_gradient_output_layer}."
        f"topgradprefix{debug_top_gradient_output_prefix}."
        f"reverse{reverse_handoff_storage}."
        f"phase{execution_phase}."
        f"phasepackoffset2."
        f"scratchperlayer1."
        f"scratchgroups{backward_scratch_group_count}."
        f"compactforward{int(compact_forward)}."
        f"canonicalexact{int(canonical_exact_attention)}."
        f"logicalrank{logical_rank}."
        f"oloraphase{o_lora_k_tiles_per_phase}."
        f"backwardpublicckpt{int(backward_direct_public_checkpoints)}."
        f"checkpointdirect1.v{43 if debug_top_full_backward else 41}"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.multi_layer_gradient_arena.{variant}",
        arg_roles=("in",) * 51 + ("out", "out"),
        version_tag=f"multi-layer-gradient-arena-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
        t50, t51, t52,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
            t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
            t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
            t50, t51, t52,
        )
        block = tensors[17:51]
        internal_base = 53
        if execution_phase == "forward":
            checkpoint_even, checkpoint_odd = 51, 52
        elif execution_phase == "backward" and not backward_direct_public_checkpoints:
            # Import the public forward results into the same private packed
            # arena ABI used by the combined owner.  Native attention kernels
            # are sensitive to the arena tensor's device layout metadata,
            # not only its bytes; consuming the public PJRT result directly
            # corrupted gradient regions while leaving dHidden plausible.
            checkpoint_even, checkpoint_odd = internal_base, internal_base + 1
            internal_base += 2
        elif execution_phase == "backward":
            # Consume the device-produced parity arenas without cloning their
            # BF16 container. Some checkpoint pages carry raw FP32 halves;
            # treating that container as a semantic BF16 clone changes those
            # bits. Reserve the same two private IDs so every later L1 tensor
            # retains the qualified address plan.
            checkpoint_even, checkpoint_odd = 5, 6
            internal_base += 2
        else:
            checkpoint_even, checkpoint_odd = internal_base, internal_base + 1
            internal_base += 2
        checkpoint_arenas = (checkpoint_even, checkpoint_odd)
        # The flattened backward kernels were qualified with the full-owner
        # tensor-pool ABI: two packed checkpoint arenas plus one checkpoint
        # staging tensor precede all reverse L1 allocations.  In a split
        # backward the arenas are public inputs, but omitting those three
        # private ABI slots changes the L1 allocation addresses and corrupts
        # the attention tail.  Preserve all three unused clone slots so the
        # reverse tensors retain the exact full-owner IDs/address plan.
        checkpoint_stage = internal_base
        hidden_scratch = checkpoint_stage + 1
        reverse0, reverse1 = hidden_scratch + 1, hidden_scratch + 2
        scratch_base = hidden_scratch + 3

        def layer_scratch(layer: int, local_tensor: int) -> int:
            """Give every reverse layer the accepted seven-tensor scratch ABI.

            Keep explicit per-layer ownership until a bounded-reuse schedule
            is independently qualified beyond the accepted depth-three path.
            """

            if not 39 <= local_tensor < 39 + backward_private_count:
                raise ValueError("backward scratch tensor is outside its ABI")
            scratch_group = layer % backward_scratch_group_count
            return (
                scratch_base
                + scratch_group * backward_private_count
                + local_tensor - 39
            )

        forward_children: list[list[dict]] = []
        compact_internal_specs = (
            forward_internal_specs if execution_phase != "backward" else ()
        )
        qwen_prefix_index_positions = {
            index: position
            for position, index in enumerate(qwen_prefix_internal_indices)
        }
        # Keep one tensor-pool ABI in every split phase. Native forward and
        # backward programs both proved sensitive to the IDs/address plan of
        # otherwise-unused resident tensors.
        existing_internal_count = (
            0 if execution_phase == "forward"
            else 6 + backward_private_count * backward_scratch_group_count
        )
        parameter_stage_base = 53 + existing_internal_count
        canonical_hidden_stage = (
            parameter_stage_base + 26
            if canonical_exact_attention else None
        )
        compact_l1_base = parameter_stage_base + (
            27 if canonical_exact_attention else 0
        )
        qwen_layer_private_base = compact_l1_base + len(
            compact_internal_specs
        )

        def compact_internal_tensor(layer: int, index: int) -> int:
            """Return a safe destination ID for a compact private tensor.

            The two TTNN outputs in Qwen's grouped-QKV prefix cannot share a
            destination object across multiple flattened layers.  All other
            compact scratch remains resident and shared as before.
            """

            position = qwen_prefix_index_positions.get(index)
            if position is None or layer == 0:
                return compact_l1_base + index
            return (
                qwen_layer_private_base
                + (layer - 1) * len(qwen_prefix_internal_indices)
                + position
            )
        parameter_unpack_children: list[dict] = []
        unpack_template = (
            compile_serialized_child(
                parameter_unpack, (tensors[15], block[0])
            )
            if parameter_unpack is not None else None
        )
        for layer in (
            range(layer_count) if execution_phase != "backward" else ()
        ):
            forward = forward0 if layer == 0 else forward_later
            if compact_forward:
                serialized_layer = compile_serialized_sequence_children(
                        forward,
                        (
                            tensors[0], *block[:26], tensors[2], tensors[4],
                            tensors[6], tensors[5],
                        ),
                    )
                if (
                    layer == 0
                    and os.environ.get("TT_FINETUNE_COMPACT_FORWARD_DEBUG")
                ):
                    print("TT_FINETUNE_COMPACT_FORWARD_CHILDREN", [
                        {
                            "child": index,
                            "num_tensors": child.get("num_tensors"),
                            "operand_indices": child.get("operand_indices"),
                            "used": sorted(serialized_child_tensor_ids(child)),
                        }
                        for index, child in enumerate(serialized_layer)
                    ])
                compiled_layer = [
                    expand_serialized_child_bindings(child)
                    for child in serialized_layer
                ]
                if len(compiled_layer) != forward.physical_programs:
                    raise RuntimeError(
                        "compact forward physical program count changed"
                    )
                packed_frozen_tensor, packed_state_tensor = 34, 35
                output_tensor = 30
            else:
                compiled_layer = [compile_serialized_child(
                    forward, (tensors[0], *block[:26], tensors[5])
                )]
                packed_frozen_tensor, packed_state_tensor = 28, 29
                output_tensor = 27

            if canonical_exact_attention:
                assert unpack_template is not None
                unpack_entries = []
                for block_index, template in enumerate(block[:26]):
                    rows, columns = (int(value) for value in template.shape)
                    page_count = ((rows + 31) // 32) * ((columns + 31) // 32)
                    if block_index in _ADAPTER_BLOCK_NAMES:
                        source_arena = 1
                        source_page_offset = (
                            plan.slice(
                                layer, _ADAPTER_BLOCK_NAMES[block_index]
                            ).page_offset
                            - plan.scratch_pages
                        )
                    else:
                        source_arena = 0
                        source_page_offset = block_page_offsets[layer][block_index]
                    unpack_entries.append(
                        (source_arena, source_page_offset, page_count)
                    )
                unpack = make_serialized_packed_parameter_unpack_bitwise(
                    unpack_template,
                    entries=unpack_entries,
                    smol_grouped_qvk=(
                        {
                            "q_a_source": unpack_entries[2][1],
                            "v_a_source": unpack_entries[10][1],
                            "k_a_source": unpack_entries[6][1],
                            "q_b_source": unpack_entries[3][1],
                            "v_b_source": unpack_entries[11][1],
                            "k_b_source": unpack_entries[7][1],
                            "a_destination": 2,
                            "q_b_destination": 3,
                            "v_b_destination": 11,
                            "k_b_destination": 7,
                            "a_pages": unpack_entries[2][2],
                            "q_b_pages": unpack_entries[3][2],
                            "v_b_pages": unpack_entries[11][2],
                            "k_b_pages": unpack_entries[7][2],
                        }
                        if not has_bias and logical_rank == 8 else None
                    ),
                    qwen_grouped_qkv=(
                        {
                            "q_weight_source": block_page_offsets[layer][28],
                            "k_weight_source": block_page_offsets[layer][29],
                            "v_weight_source": block_page_offsets[layer][30],
                            "q_weight_pages": hidden_features * hidden_features // 1024,
                            "k_weight_pages": kv_features * hidden_features // 1024,
                            "v_weight_pages": kv_features * hidden_features // 1024,
                            "q_bias_source": unpack_entries[1][1],
                            "k_bias_source": unpack_entries[5][1],
                            "v_bias_source": unpack_entries[9][1],
                            "q_bias_pages": hidden_features // 32,
                            "k_bias_pages": kv_features // 32,
                            "v_bias_pages": kv_features // 32,
                            "q_a_source": unpack_entries[2][1],
                            "k_a_source": unpack_entries[6][1],
                            "v_a_source": unpack_entries[10][1],
                            "a_column_tiles": hidden_features // 32,
                            "q_b_source": unpack_entries[3][1],
                            "k_b_source": unpack_entries[7][1],
                            "v_b_source": unpack_entries[11][1],
                            "q_b_pages": hidden_features // 32,
                            "k_b_pages": kv_features // 32,
                            "v_b_pages": kv_features // 32,
                            "q_b_destination": 3,
                            "k_b_destination": 7,
                            "v_b_destination": 11,
                            "weight_destination": 0,
                            "bias_destination": 1,
                            "a_destination": 2,
                        }
                        if has_bias and qwen_prefix_internal_indices else None
                    ),
                )
                parameter_unpack_children.append(_remap_used(unpack, {
                    0: 15,
                    1: 16,
                    **{
                        2 + index: parameter_stage_base + index
                        for index in range(26)
                    },
                }))

            transformed_layer = []
            for child_index, child in enumerate(compiled_layer):
                if compact_forward:
                    if canonical_exact_attention:
                        # The canonical artifact-v4 owner is flattened after
                        # all nested TTNN children have already been rebound to
                        # the compact owner's public ABI.  Parameter IDs 1..26
                        # therefore have one stable meaning across every child.
                        local_to_block = {
                            index + 1: index for index in range(26)
                        }
                    else:
                        compact_parameter_maps = (
                            {19: 18},
                            {index + 1: index for index in range(20)},
                            {},
                            {},
                            {20: 19},
                            {},
                            {},
                            {index: index - 1 for index in range(20, 27)},
                        )
                        local_to_block = compact_parameter_maps[child_index]
                else:
                    local_to_block = {
                        index + 1: index for index in range(26)
                    }
                if not canonical_exact_attention:
                    child = _retarget_layer_parameter_reads(
                        child,
                        layer=layer,
                        local_to_block=local_to_block,
                        block_page_offsets=block_page_offsets,
                        gradient_plan=plan,
                        packed_frozen_tensor=packed_frozen_tensor,
                        packed_state_tensor=packed_state_tensor,
                        pack_frozen=not (
                            debug_forward_checkpoint_separate_parameters
                            and layer == checkpoint_diagnostic_layer
                        ),
                        pack_trainable=not (
                            debug_forward_checkpoint_separate_parameters
                            and layer == checkpoint_diagnostic_layer
                        ),
                    )
            # Checkpoints are packed into two parity arenas. A later forward
            # reads the opposite arena and writes its own, avoiding a runtime
            # in-place alias while still keeping every checkpoint private.
            # The accessor wrapper adjusts page IDs before remapping, so the
            # qualified forward arithmetic and tile order remain unchanged.
                if layer > 0 and not canonical_exact_attention:
                    child = retarget_tensor_reads_to_packed_pages(
                        child,
                        packed_tensor=0,
                        tensor_page_offsets={
                            0: ((layer - 1) // 2)
                            * forward0.layer_allocated_pages,
                        },
                    )
            # The parity checkpoint arena is packed even when it is exposed as
            # a forward-phase DPS output.  Without retargeting here, every
            # layer starts writing at page zero, so layer 2 overwrites layer 0
            # (and layer 3 overwrites layer 1) instead of filling the second
            # slice of each parity arena.
                if not (
                    debug_forward_checkpoint_public_destination
                    and layer == checkpoint_diagnostic_layer
                ):
                    page_offset = (
                        (layer // 2) * forward0.layer_allocated_pages
                    )
                    if compact_forward:
                        # The compact sequence uses the checkpoint arena as
                        # both an input and an output across its six physical
                        # children. Apply one offset-aware accessor contract
                        # to every use; a write-only page-ID rewrite would
                        # leave later readers addressing slice zero.
                        child = retarget_tensor_reads_to_packed_pages(
                            child,
                            packed_tensor=output_tensor,
                            tensor_page_offsets={output_tensor: page_offset},
                        )
                    else:
                        child = retarget_tensor_writes_to_packed_pages(
                            child,
                            packed_tensor=output_tensor,
                            tensor_page_offsets={output_tensor: page_offset},
                        )
                hidden = (
                    0 if layer == 0
                    else checkpoint_arenas[(layer - 1) % 2]
                )
                if canonical_exact_attention and layer > 0:
                    assert canonical_hidden_stage is not None
                    hidden = canonical_hidden_stage
                output = (
                    checkpoint_arenas[layer % 2]
                    if execution_phase == "forward"
                    else 51
                    if debug_forward_checkpoint_public_destination
                    and layer == checkpoint_diagnostic_layer
                    else checkpoint_arenas[layer % 2]
                )
                mapping = {
                    0: hidden,
                    output_tensor: output,
                    packed_frozen_tensor: 15,
                    packed_state_tensor: 16,
                    **{index: 16 + index for index in range(1, 27)},
                }
                if canonical_exact_attention:
                    mapping.update({
                        index + 1: (
                            17 + index
                            if debug_forward_checkpoint_separate_parameters
                            else parameter_stage_base + index
                        )
                        for index in range(26)
                    })
                if compact_forward:
                    mapping.update({
                        27: 2,
                        28: 4,
                        29: 6,
                        **{
                            31 + index: compact_internal_tensor(layer, index)
                            for index in range(len(compact_internal_specs))
                        },
                    })
                transformed_layer.append(_remap_used(child, mapping))
            forward_children.append(transformed_layer)

        copy_child = compile_serialized_child(
            hidden_copy,
            (tensors[5], tensors[0] if execution_phase == "backward" else tensors[6]),
        )
        checkpoint_copy_child = compile_serialized_child(
            checkpoint_copy, (tensors[5], tensors[51])
        )
        checkpoint_zero_copy_child = compile_serialized_child(
            checkpoint_zero_copy, (tensors[5], tensors[51])
        )
        checkpoint_even_import_child = compile_serialized_child(
            checkpoint_even_import, (tensors[5], tensors[6])
        )
        checkpoint_odd_import_child = compile_serialized_child(
            checkpoint_odd_import, (tensors[6], tensors[5])
        )
        checkpoint_zero_copy_child = make_serialized_page_fill_zero(
            checkpoint_zero_copy_child,
            total_pages=forward0.layer_allocated_pages,
            column_tiles=forward0.output_shape[1] // 32,
        )
        top_mlp_dx_copy_child = compile_serialized_child(
            top_mlp_dx_copy, (tensors[6], tensors[52])
        )
        backward_arguments = (
            tensors[1], tensors[0], block[18], block[26],
            tensors[2], tensors[3], tensors[4],
            block[27], block[14], block[13], block[15], block[16],
            block[28], block[3], block[2], block[29], block[7], block[6],
            block[30], block[11], block[10], block[31], block[25], block[24],
            block[32], block[21], block[20], block[33], block[23], block[22],
            tensors[5], *tensors[7:14], tensors[52],
        )
        backward_template = (
            compile_serialized_sequence_children(backward, backward_arguments)
            if execution_phase != "forward"
            else ()
        )
        if backward_template and len(backward_template) != backward.physical_programs:
            raise RuntimeError("resident backward physical count changed")

        scrub_child = compile_serialized_child(
            semaphore_scrub, (tensors[6], tensors[6])
        )
        parameter_unpack_barrier = (
            _make_all_worker_barrier(scrub_child)
            if canonical_exact_attention else None
        )
        scrub_child = _make_pure_semaphore_scrub(scrub_child)
        scrub = _remap_used(
            scrub_child, {0: hidden_scratch, 1: hidden_scratch}
        )
        children = []
        if execution_phase == "backward" and not backward_direct_public_checkpoints:
            children.append(_remap_used(checkpoint_even_import_child, {
                0: 5, 1: checkpoint_even,
            }))
            children.append(_remap_used(checkpoint_odd_import_child, {
                0: 6, 1: checkpoint_odd,
            }))
        # `ttnn.zeros` currently loses its private tensor-pool entry before a
        # long artifact-v4 sequence reaches backward. Keep the stable EmptyOp
        # allocation lifetime and initialize each packed slice explicitly
        # from the framework-owned all-zero checkpoint template instead.
        for layer in (
            range(layer_count)
            if execution_phase == "full" or (
                execution_phase == "forward"
                and canonical_exact_attention
            ) else ()
        ):
            zero_child = checkpoint_zero_copy_child
            zero_child = retarget_tensor_reads_to_packed_pages(
                zero_child,
                packed_tensor=1,
                tensor_page_offsets={
                    1: (layer // 2) * forward0.layer_allocated_pages,
                },
            )
            children.append(_remap_used(zero_child, {
                1: checkpoint_arenas[layer % 2],
            }))
        for layer, layer_forward_children in enumerate(forward_children):
            if canonical_exact_attention:
                if layer > 0:
                    assert canonical_hidden_stage is not None
                    layer_hidden_copy = retarget_tensor_reads_to_packed_pages(
                        copy_child,
                        packed_tensor=0,
                        tensor_page_offsets={
                            0: ((layer - 1) // 2)
                            * forward0.layer_allocated_pages,
                        },
                    )
                    children.append(_remap_used(layer_hidden_copy, {
                        0: checkpoint_arenas[(layer - 1) % 2],
                        1: canonical_hidden_stage,
                    }))
                if not debug_forward_checkpoint_separate_parameters:
                    children.append(parameter_unpack_children[layer])
                    # The scatter is distributed over all 110 workers,
                    # whereas each following TTNN child may read any page
                    # from any core. Publish the staged tensors explicitly.
                    assert parameter_unpack_barrier is not None
                    children.append(parameter_unpack_barrier)
            children.extend(layer_forward_children)
            if (
                canonical_exact_attention
                and not debug_forward_checkpoint_separate_parameters
            ):
                # The canonical parameter and compact-private tensor pools are
                # deliberately reused by the next flattened layer.  A PJRT
                # custom-call boundary used to provide the global retirement
                # point implicitly; one physical-program boundary only drains
                # each worker locally and can let the next layer's distributed
                # unpack overwrite pages still consumed by a slower peer.
                # Rendezvous all workers before either pool is reused.
                assert parameter_unpack_barrier is not None
                children.append(parameter_unpack_barrier)
            if execution_phase == "full" or (
                compact_forward and execution_phase == "forward"
            ):
                children.append(scrub)
        # Combined execution reaches reverse immediately after the final
        # forward epoch's explicit semaphore scrub.  Split backward used to
        # skip that lifecycle boundary, exposing attention kernels to stale
        # semaphore/L1 state left by unrelated device work.
        if execution_phase == "backward":
            children.append(scrub)
        if debug_forward_checkpoint_only:
            if not debug_forward_checkpoint_public_destination:
                children.append(_remap_used(checkpoint_copy_child, {
                    0: checkpoint_arenas[checkpoint_diagnostic_layer % 2],
                    1: 51,
                }))
            backward_template = ()
        else:
            backward_template = backward_template
        dy = 1
        reverse_target = reverse0
        reverse_layers = (
            () if debug_forward_checkpoint_only or execution_phase == "forward" else tuple(
                reversed(range(layer_count))
            )[:(
                debug_reverse_layer_limit
                if debug_reverse_layer_limit > 0
                else layer_count
            )]
        )
        for reverse_index, layer in enumerate(reverse_layers):
            if layer > 0:
                layer_copy = retarget_tensor_reads_to_packed_pages(
                    copy_child,
                    packed_tensor=0,
                    tensor_page_offsets={
                        0: ((layer - 1) // 2)
                        * forward0.layer_allocated_pages,
                    },
                )
                children.append(_remap_used(layer_copy, {
                    0: checkpoint_arenas[(layer - 1) % 2],
                    1: hidden_scratch,
                }))
                hidden = hidden_scratch
            else:
                hidden = 0

            if debug_top_hidden_input_only:
                if layer != layer_count - 1:
                    raise RuntimeError(
                        "top-hidden diagnostic reached a non-terminal layer"
                    )
                children.append(_remap_used(top_mlp_dx_copy_child, {
                    0: hidden,
                    1: 52,
                }))
                break

            checkpoint_source = checkpoint_arenas[layer % 2]
            checkpoint_page_offset = (
                (layer // 2) * forward0.layer_allocated_pages
            )
            layer_children = [
                retarget_tensor_reads_to_packed_pages(
                    child,
                    packed_tensor=30,
                    tensor_page_offsets={30: checkpoint_page_offset},
                )
                for child in backward_template
            ]
            gradient_output_layer = (
                debug_top_gradient_output_layer
                if debug_top_stop_after_mlp
                and debug_top_gradient_output_layer >= 0
                else layer
            )
            mlp_output_order = {
                "gate_a": 0, "gate_b": 1, "up_a": 2,
                "up_b": 3, "down_a": 4, "down_b": 5,
            }
            def offset(name: str) -> int:
                output_layer = (
                    gradient_output_layer
                    if name in mlp_output_order
                    and mlp_output_order[name] < debug_top_gradient_output_prefix
                    else 0
                    if name in mlp_output_order
                    else layer
                )
                return plan.slice(output_layer, name).page_offset
            layer_children[0] = retarget_full_mlp_gradient_outputs(
                layer_children[0],
                gate_a_page_offset=offset("gate_a"),
                gate_b_page_offset=offset("gate_b"),
                up_a_page_offset=offset("up_a"),
                up_b_page_offset=offset("up_b"),
                down_a_page_offset=offset("down_a"),
                down_b_page_offset=offset("down_b"),
            )
            if debug_top_stop_after_mlp:
                if layer != layer_count - 1:
                    raise RuntimeError(
                        "top-MLP diagnostic reached a non-terminal layer"
                    )
                mlp_child = layer_children[0]
                base_mapping = {
                    0: dy, 1: hidden, 4: 2, 5: 3, 6: 4,
                    30: checkpoint_source,
                    **{index: 51 for index in range(31, 38)},
                    38: 52,
                    **{
                        index: layer_scratch(layer, index)
                        for index in range(39, 39 + backward_private_count)
                    },
                    46: 15, 47: 16,
                }
                mlp_child = _retarget_layer_parameter_reads(
                    mlp_child,
                    layer=layer,
                    local_to_block=_BACKWARD_LOCAL_TO_BLOCK,
                    block_page_offsets=block_page_offsets,
                    gradient_plan=plan,
                    packed_frozen_tensor=46,
                    packed_state_tensor=47,
                    pack_frozen=(
                        debug_top_attention_parameter_source
                        not in ("separate_frozen", "separate_all")
                    ),
                    pack_trainable=(
                        debug_top_attention_parameter_source
                        not in ("separate_trainable", "separate_all")
                    ),
                )
                base_mapping.update({
                    local: 17 + block
                    for local, block in _BACKWARD_LOCAL_TO_BLOCK.items()
                })
                children.append(_remap_used(mlp_child, base_mapping))
                children.append(_remap_used(top_mlp_dx_copy_child, {
                    0: layer_scratch(layer, 39),
                    1: 52,
                }))
                break
            if (
                debug_top_stop_after_post_extract
                or debug_top_stop_after_post_residual
                or debug_top_stop_after_post_backward
                or debug_top_stop_after_attention_input
                or debug_top_stop_after_attention
                or debug_top_attention_stage is not None
            ):
                if layer != layer_count - 1:
                    raise RuntimeError(
                        "top-attention diagnostic reached a non-terminal layer"
                    )
                base_mapping = {
                    0: dy, 1: hidden, 4: 2, 5: 3, 6: 4,
                    30: checkpoint_source,
                    **{index: 51 for index in range(31, 38)},
                    38: 52,
                    **{
                        index: layer_scratch(layer, index)
                        for index in range(39, 39 + backward_private_count)
                    },
                    46: 15, 47: 16,
                }
                for child in layer_children:
                    transformed = _retarget_layer_parameter_reads(
                        child,
                        layer=layer,
                        local_to_block=_BACKWARD_LOCAL_TO_BLOCK,
                        block_page_offsets=block_page_offsets,
                        gradient_plan=plan,
                        packed_frozen_tensor=46,
                        packed_state_tensor=47,
                    )
                    children.append(_remap_used(transformed, base_mapping))
                break
            attention_writers = [
                index for index, child in enumerate(layer_children)
                if _is_full_attention_writer(child)
            ]
            if len(attention_writers) != 1:
                raise RuntimeError(
                    f"layer {layer} attention writer count changed: "
                    f"{len(attention_writers)}"
                )
            attention_index = attention_writers[0]
            layer_children[attention_index] = (
                retarget_full_attention_gradient_outputs(
                    layer_children[attention_index],
                    o_a_page_offset=offset("o_a"),
                    o_b_page_offset=offset("o_b"),
                    q_a_page_offset=offset("q_a"),
                    k_a_page_offset=offset("k_a"),
                    v_a_page_offset=offset("v_a"),
                    q_b_page_offset=offset("q_b"),
                    k_b_page_offset=offset("k_b"),
                    v_b_page_offset=offset("v_b"),
                    logical_rank=(
                        physical_rank if logical_rank is None else logical_rank
                    ),
                )
            )
            # A stopped top-layer diagnostic has no lower-layer consumer, so
            # publish dHidden through the public result.  The original path
            # keeps it in a private reverse-chain L1 buffer.  The explicit
            # arena path instead reuses the gradient arena's page-zero
            # dHidden scratch prefix.  That prefix is dead after a layer's
            # input-RMS residual add and is consumed by the next layer's MLP
            # before its attention backward reuses it.  This gives both
            # producer and consumer one caller-owned, page-addressed storage
            # contract without exposing another PJRT value or adding a custom
            # call boundary.
            dhidden = (
                52 if layer == 0 or debug_top_full_backward
                or (
                    debug_reverse_layer_limit > 0
                    and reverse_index == len(reverse_layers) - 1
                )
                or reverse_handoff_storage == "public_final_inplace"
                or (
                    reverse_handoff_storage == "top_public_then_private"
                    and layer == layer_count - 1
                )
                else 51 if reverse_handoff_storage == "gradient_arena"
                else reverse_target
            )
            base_mapping = {
                0: dy, 1: hidden, 4: 2, 5: 3, 6: 4,
                30: checkpoint_source,
                **{index: 51 for index in range(31, 38)},
                38: dhidden,
                **{
                    index: layer_scratch(layer, index)
                    for index in range(39, 39 + backward_private_count)
                },
                46: 15, 47: 16,
                **{
                    local: 17 + block
                    for local, block in _BACKWARD_LOCAL_TO_BLOCK.items()
                },
            }
            for child in layer_children:
                top_attention_child = (
                    debug_top_full_backward
                    and layer == layer_count - 1
                    and _is_full_attention_writer(child)
                )
                separate_frozen = (
                    top_attention_child
                    and debug_top_attention_parameter_source
                    in ("separate_frozen", "separate_all")
                )
                separate_trainable = (
                    top_attention_child
                    and debug_top_attention_parameter_source
                    in ("separate_trainable", "separate_all")
                )
                transformed = _retarget_layer_parameter_reads(
                    child,
                    layer=layer,
                    local_to_block=_BACKWARD_LOCAL_TO_BLOCK,
                    block_page_offsets=block_page_offsets,
                    gradient_plan=plan,
                    packed_frozen_tensor=46,
                    packed_state_tensor=47,
                    pack_frozen=not separate_frozen,
                    pack_trainable=not separate_trainable,
                )
                children.append(_remap_used(transformed, base_mapping))
            if debug_top_full_backward:
                if layer != layer_count - 1:
                    raise RuntimeError(
                        "top-full-backward diagnostic reached a non-terminal layer"
                    )
                break
            dy = dhidden
            if dhidden in (reverse0, reverse1):
                reverse_target = (
                    reverse1 if reverse_target == reverse0 else reverse0
                )

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                _dump_composed_sequence(
                    children,
                    label=(
                        f"layers{layer_count}_topfull"
                        f"{int(debug_top_full_backward)}"
                    ),
                )
                compiled.program_sequence = children
                checkpoint_internals = (
                    [
                        {
                            "source_operand": 5,
                            "memory": "clone",
                            "shape": [
                                ((layer_count + 1) // 2)
                                * forward0.output_shape[0],
                                forward0.output_shape[1],
                            ],
                        },
                        {
                            "source_operand": 5,
                            "memory": "clone",
                            "shape": [
                                (layer_count // 2) * forward0.output_shape[0],
                                forward0.output_shape[1],
                            ],
                        },
                        {
                            "source_operand": 5,
                            "memory": "clone",
                            "shape": [
                                forward0.output_shape[0],
                                forward0.output_shape[1],
                            ],
                        },
                    ]
                    if execution_phase == "full"
                    else []
                )
                if execution_phase == "backward":
                    checkpoint_internals = [
                        {
                            "source_operand": 5,
                            "memory": "clone",
                            "shape": [
                                ((layer_count + 1) // 2)
                                * forward0.output_shape[0],
                                forward0.output_shape[1],
                            ],
                        },
                        {
                            "source_operand": 6,
                            "memory": "clone",
                            "shape": [
                                (layer_count // 2) * forward0.output_shape[0],
                                forward0.output_shape[1],
                            ],
                        },
                        {
                            "source_operand": 5,
                            "memory": "clone",
                            "shape": [
                                forward0.output_shape[0],
                                forward0.output_shape[1],
                            ],
                        },
                    ]
                reverse_internals = (
                    [
                        # hidden_scratch, reverse0 and reverse1 occupy exactly
                        # the three IDs computed above.  Backward-only mode
                        # receives both checkpoint arenas as public inputs;
                        # therefore it has no private checkpoint-stage slot.
                        # Adding one here shifts every padded layer scratch by
                        # one ID and makes kernels address the wrong L1 bank.
                        {
                            "source_operand": 0,
                            **backward_handoff_shape,
                            "memory": (
                                "clone" if reverse_handoff_storage == "private_dram"
                                else "l1"
                            ),
                        },
                        {
                            "source_operand": 1,
                            **backward_handoff_shape,
                            "memory": (
                                "clone" if reverse_handoff_storage == "private_dram"
                                else "l1"
                            ),
                        },
                        {
                            "source_operand": 1,
                            **backward_handoff_shape,
                            "memory": (
                                "clone" if reverse_handoff_storage == "private_dram"
                                else "l1"
                            ),
                        },
                        *(
                            {
                                **spec,
                                "memory": (
                                    "clone" if reverse_handoff_storage == "private_dram"
                                    else spec.get("memory", "l1")
                                ),
                            }
                            for _layer in range(
                                backward_scratch_group_count
                            )
                            for spec in backward_private_specs
                        ),
                    ]
                    if execution_phase != "forward"
                    else []
                )
                compiled.internal_tensors = [
                    *checkpoint_internals,
                    *reverse_internals,
                    *(
                        tuple(
                            {
                                "source_operand": 17 + index,
                                "memory": "clone",
                                **(
                                    {
                                        "shape": (
                                            [
                                                hidden_features
                                                + 2 * kv_features,
                                                hidden_features,
                                            ]
                                            if index == 0 else
                                            [1, hidden_features + 2 * kv_features]
                                            if index == 1 else
                                            [
                                                hidden_features
                                                if index == 3 else kv_features,
                                                64,
                                            ]
                                            if index in (3, 7, 11) else
                                            [3 * int(logical_rank), hidden_features]
                                        ),
                                        "layout": "tile",
                                    }
                                    if (
                                        has_bias
                                        and bool(qwen_prefix_internal_indices)
                                        and index in (0, 1, 2, 3, 7, 11)
                                    ) or (
                                        not has_bias
                                        and logical_rank == 8
                                        and index == 2
                                    )
                                    else {}
                                ),
                            }
                            for index in range(26)
                        )
                        if canonical_exact_attention else ()
                    ),
                    *(
                        ({
                            "source_operand": 0,
                            "memory": "clone",
                            "shape": [sequence_length, hidden_features],
                            "layout": "tile",
                        },)
                        if canonical_exact_attention else ()
                    ),
                    *(
                        {
                            **spec,
                            "source_operand": (
                                0
                                if int(spec.get("source_operand", 0)) == 0
                                else 16 + int(spec["source_operand"])
                                if 1 <= int(spec["source_operand"]) <= 26
                                else 2
                                if int(spec["source_operand"]) == 27
                                else 4
                                if int(spec["source_operand"]) == 28
                                else 6
                                if int(spec["source_operand"]) == 29
                                else 5
                                if int(spec["source_operand"]) == 30
                                else int(spec["source_operand"])
                            ),
                        }
                        for spec in compact_internal_specs
                    ),
                    # A flattened Qwen artifact executes every layer without
                    # returning to PJRT. Give only the grouped-QKV TTNN
                    # producer destinations unique per-layer storage so their
                    # runtime admission/lifecycle state cannot leak into the
                    # next layer. The mathematical inputs and programs are
                    # unchanged.
                    *(
                        {
                            **compact_internal_specs[index],
                            "source_operand": 0,
                        }
                        for _layer in range(1, layer_count)
                        for index in qwen_prefix_internal_indices
                    ),
                ]
            break
        return result

    operation.artifact_format = 4
    operation.layer_count = layer_count
    full_physical_programs = (
        3 * layer_count
        + (
            4 if debug_top_stop_after_mlp
            else 3 if debug_top_hidden_input_only
            else 4 if debug_top_stop_after_post_extract
            else 6 if debug_top_stop_after_post_residual
            else 9 if debug_top_stop_after_post_backward
            else 11 if debug_top_stop_after_attention_input
            else 11 if debug_top_stop_after_attention
            else (2 + backward.physical_programs) if debug_top_full_backward
            else 1 if debug_forward_checkpoint_only
            else (
                debug_reverse_layer_limit
                - int(debug_reverse_layer_limit == layer_count)
                + debug_reverse_layer_limit * backward.physical_programs
            ) if debug_reverse_layer_limit > 0
            else (layer_count - 1) + layer_count * backward.physical_programs
        )
    )
    forward_programs_per_layer = (
        forward0.physical_programs if compact_forward else 1
    )
    operation.physical_programs = (
        layer_count * (
            forward_programs_per_layer
            + int(compact_forward)
            + 3 * int(canonical_exact_attention)
        )
        + (layer_count - 1) * int(canonical_exact_attention)
        + 2
        if execution_phase == "forward"
        else (layer_count - 1) + layer_count * backward.physical_programs
        if execution_phase == "backward"
        else full_physical_programs
        + layer_count * (
            forward_programs_per_layer - 1
            + 2 * int(canonical_exact_attention)
        )
        + (layer_count - 1) * int(canonical_exact_attention)
    )
    operation.compiler_owned_internal_tensors = (
        (
            len(forward_internal_specs)
            + (layer_count - 1) * len(qwen_prefix_internal_indices)
            + (27 if canonical_exact_attention else 0)
            if compact_forward else 0
        )
        if execution_phase == "forward"
        else 6 + backward_private_count * backward_scratch_group_count
        if execution_phase == "backward"
        else 6 + backward_private_count * backward_scratch_group_count
        + (
            len(forward_internal_specs)
            + (layer_count - 1) * len(qwen_prefix_internal_indices)
            + (27 if canonical_exact_attention else 0)
            if compact_forward else 0
        )
    )
    operation.backward_scratch_groups = (
        backward_scratch_group_count
    )
    operation.compiler_owned_checkpoint_tensors = (
        2 if execution_phase in ("full", "forward", "backward") else 0
    )
    operation.checkpoint_arena_shapes = (
        (((layer_count + 1) // 2) * forward0.output_shape[0],
         forward0.output_shape[1]),
        ((layer_count // 2) * forward0.output_shape[0],
         forward0.output_shape[1]),
    )
    operation.checkpoint_arena_pages = (
        layer_count * forward0.layer_allocated_pages
    )
    operation.framework_visible_interlayer_tensors = (
        2 if execution_phase == "forward" else 0
    )
    operation.framework_visible_gradient_tensors = (
        0 if execution_phase == "forward" else 1
    )
    operation.gradient_arena_shape = plan.physical_shape
    operation.gradient_page_offset = plan.scratch_pages
    operation.gradient_pages = layer_count * plan.layer_gradient_pages
    operation.safe_default = False
    operation.experimental_backend = "multi_layer_packed_gradient_checkpoint_arena_v13"
    operation.debug_forward_checkpoint_only = debug_forward_checkpoint_only
    operation.debug_forward_checkpoint_layer = checkpoint_diagnostic_layer
    operation.debug_forward_checkpoint_separate_parameters = (
        debug_forward_checkpoint_separate_parameters
    )
    operation.debug_forward_checkpoint_public_destination = (
        debug_forward_checkpoint_public_destination
    )
    operation.debug_top_hidden_input_only = debug_top_hidden_input_only
    operation.debug_top_stop_after_mlp = debug_top_stop_after_mlp
    operation.debug_top_gradient_output_layer = debug_top_gradient_output_layer
    operation.debug_top_gradient_output_prefix = debug_top_gradient_output_prefix
    operation.debug_top_stop_after_post_extract = (
        debug_top_stop_after_post_extract
    )
    operation.debug_top_stop_after_post_residual = (
        debug_top_stop_after_post_residual
    )
    operation.debug_top_stop_after_post_backward = (
        debug_top_stop_after_post_backward
    )
    operation.debug_top_stop_after_attention_input = (
        debug_top_stop_after_attention_input
    )
    operation.debug_top_stop_after_attention = debug_top_stop_after_attention
    operation.debug_top_attention_stage = debug_top_attention_stage
    operation.debug_top_full_backward = debug_top_full_backward
    operation.debug_reverse_layer_limit = debug_reverse_layer_limit
    operation.execution_phase = execution_phase
    operation.compact_forward = bool(compact_forward)
    operation.canonical_exact_attention = bool(canonical_exact_attention)
    operation.logical_rank = logical_rank
    operation.reverse_handoff_storage = reverse_handoff_storage
    operation.reverse_handoff_dram_bytes_per_edge = (
        sequence_length * hidden_features * 2
        if reverse_handoff_storage in (
            "private_dram", "gradient_arena", "public_final_inplace",
            "top_public_then_private",
        ) else 0
    )
    operation.reverse_handoff_dram_edges = (
        1 if reverse_handoff_storage == "top_public_then_private"
        else layer_count - 1 if reverse_handoff_storage in (
            "private_dram", "gradient_arena", "public_final_inplace",
        ) else 0
    )
    operation._ttl_program = _probe
    return operation
