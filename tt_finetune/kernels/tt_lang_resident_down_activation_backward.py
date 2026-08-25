"""One-call mixed-island down projection and SwiGLU backward.

This is the first real user of TT-Lang artifact format v2.  The down LoRA
backward keeps its accepted HiFi4/FP32-destination schedule on a 10x10 worker
island.  Exact SwiGLU arithmetic runs on the remaining 1x10 island with FP32
destination accumulation disabled.  dProduct crosses only through L1/NoC.
"""

from __future__ import annotations

import hashlib
import re
import struct
from copy import copy
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID
from .tt_lang_frozen_linear_lora_backward import (
    _shift_compile_time_cb_indices,
)
from .tt_lang_lora import TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR
from .tt_lang_resident_down_backward import (
    resident_down_backward_support_error,
)


ttl = None
tt_lang_operation = None

_MATMUL_RANGE = {"start": [0, 0], "end": [9, 9]}
_ACTIVATION_RANGE = {"start": [10, 0], "end": [10, 9]}
_ALL_WORKERS_RANGE = {"start": [0, 0], "end": [10, 9]}
_MATMUL_NODES = 100
_ACTIVATION_NODES = 10
_TOTAL_CBS = 38


def resident_down_activation_output_tile_shape(
    *, tokens: int, input_features: int, output_features: int
) -> tuple[int, int]:
    """Arena rows are dGate, dUp, down dA, then down dB."""

    return (
        2 * (tokens // TILE_SIZE) + 1 + output_features // TILE_SIZE,
        input_features // TILE_SIZE,
    )


def resident_down_activation_backward_support_error(
    *,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    base_reduction_tiles_per_phase: int = 1,
) -> str | None:
    """Return why the mixed-island exact schedule cannot own this shape."""

    error = resident_down_backward_support_error(
        tokens=tokens,
        input_features=input_features,
        output_features=output_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=base_reduction_tiles_per_phase,
    )
    if error is not None:
        return error
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    jobs_per_node = -(-(m_tiles * k_tiles) // _MATMUL_NODES)
    da_k_tiles_per_node = -(-k_tiles // _MATMUL_NODES)
    n_tiles_per_node = -(-n_tiles // _MATMUL_NODES)
    if max(jobs_per_node, da_k_tiles_per_node, n_tiles_per_node) > 32:
        return "mixed down schedule exceeds the 32-block CB limit"
    return None


def _cpp(name: str) -> str:
    return (Path(__file__).with_name("cpp") / name).read_text(
        encoding="utf-8"
    )


def _replace_all(source: str, replacements: dict[str, str]) -> str:
    for placeholder, value in replacements.items():
        source = source.replace(placeholder, value)
    unresolved = re.findall(r"@[A-Z0-9_]+@", source)
    if unresolved:
        raise RuntimeError(f"unexpanded mixed-island placeholders: {unresolved}")
    return source


def _down_compute_source(replacements: dict[str, str]) -> str:
    lora_source = _shift_compile_time_cb_indices(
        _cpp("lora_backward_arena_exact_compute.cpp"), 4
    ).replace(
        "void kernel_main()",
        "FORCE_INLINE void lora_backward_phase()",
        1,
    )
    base_source = _cpp("frozen_linear_backward_base_phase.cpp").replace(
        "void kernel_main()",
        "FORCE_INLINE void resident_down_phase_one()",
        1,
    )
    helper_source = _cpp(
        "grouped2_frozen_lora_backward_tail_exact_compute.cpp"
    ).split("\nvoid kernel_main()", 1)[0]
    helper_source = helper_source.replace(
        "            dh.wait_front(reduction_block_tiles);\n",
        "            // Complete dH window is already resident.\n",
    ).replace(
        (
            "                    cb_transposed_x,\n"
            "                    cb_dh,\n"
            "                    inner,\n"
            "                    inner,\n"
        ),
        (
            "                    cb_transposed_x,\n"
            "                    cb_dh,\n"
            "                    inner,\n"
            "                    block * reduction_block_tiles + inner,\n"
        ),
    ).replace(
        "            dh.pop_front(reduction_block_tiles);\n",
        "            // Keep dH resident for LoRA dProduct.\n",
    )
    source = (
        lora_source
        + "\n"
        + base_source
        + "\n"
        + helper_source
        + "\n"
        + _cpp("resident_down_backward_tail.cpp")
    )
    source = source.replace(
        "constexpr uint32_t grid_cols = 11;",
        "constexpr uint32_t grid_cols = 10;",
    )
    return _replace_all(source, replacements)


def _down_reader_source(
    replacements: dict[str, str], *, num_semaphores: int, init_ack: int,
    total_cbs: int = _TOTAL_CBS,
) -> str:
    source = _cpp("resident_down_backward_reader.cpp")
    source = source.replace(
        "constexpr uint32_t kGridCols = 11;\n"
        "constexpr uint32_t kWorkerNodes = 110;",
        "constexpr uint32_t kGridCols = 10;\n"
        "constexpr uint32_t kWorkerNodes = 100;",
    ).replace(
        "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
        "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};",
        "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
        "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};\n"
        "constexpr uint8_t kActivationPhysicalX = 13;",
    ).replace(
        "for (uint32_t semaphore = 0; semaphore < 2 * m_tiles + 1; ++semaphore)",
        f"for (uint32_t semaphore = 0; semaphore < {num_semaphores}; ++semaphore)",
    )
    source = re.sub(
        r"get_tensor_accessor_args_cta_offset<(\d+), 25>",
        rf"get_tensor_accessor_args_cta_offset<\1, {total_cbs}>",
        source,
    )
    acknowledgement = f"""

    // Cross-island initialization barrier.  Send this after dH is resident,
    // but before filling the tail input CBs.  Sending it at kernel exit can
    // deadlock: the tail reader may wait for CB space while the down writer
    // waits for the activation island that is still waiting for this ack.
    noc_semaphore_inc(
        get_noc_addr(
            kActivationPhysicalX, physical_y(node % 10),
            get_semaphore({init_ack})),
        1);
    // Blackhole does not implicitly flush non-posted NoC atomics when a
    // kernel exits.  The activation island must observe this ack before this
    // reader can complete or move on to unrelated traffic.
    noc_async_atomic_barrier();
"""
    insertion_point = "    dh_resident.push_back(m_tiles);\n"
    if source.count(insertion_point) != 1:
        raise RuntimeError("resident down reader dH residency marker changed")
    source = source.replace(
        insertion_point, insertion_point + acknowledgement, 1
    )
    return _replace_all(source, replacements)


def _append_before_kernel_close(source: str, tail: str) -> str:
    """Append statements before the final ``kernel_main`` brace."""

    stripped = source.rstrip()
    if not stripped.endswith("}"):
        raise RuntimeError("generated kernel no longer ends with a brace")
    return stripped[:-1] + tail + "\n}\n"


def _banked_activation_compute_source(
    source: str, *, activation_banks: int
) -> str:
    if activation_banks == 1:
        return source
    if activation_banks != 2:
        raise ValueError("activation handoff supports one or two banks")
    source = source.replace(
        "constexpr uint32_t cb_dgate = get_compile_time_arg_val(35);",
        "constexpr uint32_t cb_dgate0 = get_compile_time_arg_val(35);\n"
        "constexpr uint32_t cb_dgate1 = get_compile_time_arg_val(38);",
    ).replace(
        "constexpr uint32_t cb_dup = get_compile_time_arg_val(36);",
        "constexpr uint32_t cb_dup0 = get_compile_time_arg_val(36);\n"
        "constexpr uint32_t cb_dup1 = get_compile_time_arg_val(39);",
    )
    marker = (
        "    for (uint32_t job = 0; job < local_jobs; ++job) {\n"
        "        dproduct.wait_front(1);"
    )
    replacement = (
        "    for (uint32_t job = 0; job < local_jobs; ++job) {\n"
        "        const uint32_t cb_dgate = "
        "job < 32 ? cb_dgate0 : cb_dgate1;\n"
        "        const uint32_t cb_dup = "
        "job < 32 ? cb_dup0 : cb_dup1;\n"
        "        dproduct.wait_front(1);"
    )
    if source.count(marker) != 1:
        raise RuntimeError("activation compute loop marker changed")
    return source.replace(marker, replacement, 1)


def _activation_compute_source(
    replacements: dict[str, str], *, producer_jobs: int, total_jobs: int,
    diagnostic_stage: int,
) -> str:
    source = _cpp("swiglu_backward_exact_compute.cpp")
    source = re.sub(
        r"constexpr uint32_t (cb_[a-z0-9_]+) = (\d+);",
        lambda match: (
            f"constexpr uint32_t {match.group(1)} = "
            "get_compile_time_arg_val("
            f"{int(match.group(2)) + 25});"
        ),
        source,
    )
    main = source.index("void kernel_main() {")
    if diagnostic_stage == -2:
        # Synchronization-only diagnostic.  Publish the two output CB slots
        # without touching the compute engine, so the reader/mailbox/writer
        # pipeline can be validated independently of unpack/math/pack.
        return source[:main] + f"""void kernel_main() {{
    constexpr uint32_t total_jobs = {total_jobs};
    constexpr uint32_t producer_jobs = {producer_jobs};
    constexpr uint32_t producer_nodes = {_MATMUL_NODES};
    constexpr uint32_t activation_nodes = {_ACTIVATION_NODES};
    const uint32_t node = get_absolute_logical_y();
    uint32_t local_jobs = 0;
    for (uint32_t producer = node; producer < producer_nodes;
         producer += activation_nodes) {{
        const uint32_t first = producer * producer_jobs;
        if (first >= total_jobs) continue;
        const uint32_t remaining = total_jobs - first;
        local_jobs += remaining < producer_jobs ? remaining : producer_jobs;
    }}
    CircularBuffer dproduct(cb_dproduct);
    CircularBuffer gate(cb_gate);
    CircularBuffer up(cb_up);
    CircularBuffer dgate(cb_dgate);
    CircularBuffer dup(cb_dup);
    for (uint32_t job = 0; job < local_jobs; ++job) {{
        dproduct.wait_front(1);
        gate.wait_front(1);
        up.wait_front(1);
        dgate.reserve_back(1);
        dup.reserve_back(1);
        dgate.push_back(1);
        dup.push_back(1);
        up.pop_front(1);
        gate.pop_front(1);
        dproduct.pop_front(1);
    }}
}}
"""
    if diagnostic_stage == -1:
        # Pipeline-only diagnostic: exercise the complete L1/NoC handoff and
        # both final-output CBs without invoking the SwiGLU arithmetic.  This
        # distinguishes a transport/schedule fault from a compute dependency.
        return source[:main] + f"""void kernel_main() {{
    constexpr uint32_t total_jobs = {total_jobs};
    constexpr uint32_t producer_jobs = {producer_jobs};
    constexpr uint32_t producer_nodes = {_MATMUL_NODES};
    constexpr uint32_t activation_nodes = {_ACTIVATION_NODES};
    const uint32_t node = get_absolute_logical_y();
    uint32_t local_jobs = 0;
    for (uint32_t producer = node; producer < producer_nodes;
         producer += activation_nodes) {{
        const uint32_t first = producer * producer_jobs;
        if (first >= total_jobs) continue;
        const uint32_t remaining = total_jobs - first;
        local_jobs += remaining < producer_jobs ? remaining : producer_jobs;
    }}
    CircularBuffer dproduct(cb_dproduct);
    CircularBuffer gate(cb_gate);
    CircularBuffer up(cb_up);
    for (uint32_t job = 0; job < local_jobs; ++job) {{
        dproduct.wait_front(1);
        gate.wait_front(1);
        up.wait_front(1);
        copy_one(cb_dproduct, cb_dgate, true);
        copy_one(cb_dproduct, cb_dup, true);
        up.pop_front(1);
        gate.pop_front(1);
        dproduct.pop_front(1);
    }}
}}
"""
    buffers = source.index("    CircularBuffer dproduct", main)
    schedule = f"""void kernel_main() {{
    constexpr uint32_t total_jobs = {total_jobs};
    constexpr uint32_t producer_jobs = {producer_jobs};
    constexpr uint32_t producer_nodes = {_MATMUL_NODES};
    constexpr uint32_t activation_nodes = {_ACTIVATION_NODES};
    constexpr uint32_t diagnostic_stage = {diagnostic_stage};
    const uint32_t node = get_absolute_logical_y();
    uint32_t local_jobs = 0;
    for (uint32_t producer = node; producer < producer_nodes;
         producer += activation_nodes) {{
        const uint32_t first = producer * producer_jobs;
        if (first >= total_jobs) continue;
        const uint32_t remaining = total_jobs - first;
        local_jobs += remaining < producer_jobs ? remaining : producer_jobs;
    }}

"""
    source = source[:main] + schedule + source[buffers:]
    return _replace_all(source, replacements)


@lru_cache(maxsize=16)
def build_tt_lang_resident_down_activation_backward(
    *,
    scale: float,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    base_reduction_tiles_per_phase: int = 1,
    activation_diagnostic_stage: int = 6,
    resident_activation_output_handoff: bool = False,
) -> Callable:
    """Build the experimental 100-core + 10-core exact custom call."""

    if activation_diagnostic_stage not in range(-2, 7):
        raise ValueError("activation diagnostic stage must be in [-2, 6]")
    error = resident_down_activation_backward_support_error(
        tokens=tokens,
        input_features=input_features,
        output_features=output_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=base_reduction_tiles_per_phase,
    )
    if error is not None:
        raise ValueError(error)
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    total_jobs = m_tiles * k_tiles
    jobs_per_node = -(-total_jobs // _MATMUL_NODES)
    activation_jobs = max(
        sum(
            min(jobs_per_node, total_jobs - producer * jobs_per_node)
            for producer in range(
                activation_node, _MATMUL_NODES, _ACTIVATION_NODES
            )
            if producer * jobs_per_node < total_jobs
        )
        for activation_node in range(_ACTIVATION_NODES)
    )
    activation_banks = -(-activation_jobs // 32)
    if resident_activation_output_handoff and activation_banks > 2:
        raise ValueError(
            "resident activation output handoff supports at most two banks"
        )
    da_k_tiles_per_node = -(-k_tiles // _MATMUL_NODES)
    n_tiles_per_node = -(-n_tiles // _MATMUL_NODES)
    mailbox_semaphore = 2 * m_tiles + 1
    ready_semaphore = mailbox_semaphore + 1
    init_ack_semaphore = mailbox_semaphore + 2
    # Blackhole has 16 hardware semaphore IDs/core. Once every activation
    # output exists, the dH and dProduct handoff mailboxes are dead, so the
    # output transport deliberately reuses IDs 0..6 instead of extending the
    # allocation. Four address slots cover Qwen's two dGate/dUp banks.
    output_address_semaphore_start = 0
    output_ready_semaphore = 4
    consumer_done_semaphore = 5
    output_release_semaphore = 6
    num_semaphores = init_ack_semaphore + 1
    activation_extra_cbs = (
        2 * (activation_banks - 1)
        if resident_activation_output_handoff
        else 0
    )
    pulled_dgate_arg = _TOTAL_CBS + activation_extra_cbs
    pulled_dup_arg = pulled_dgate_arg + 1
    total_cbs = (
        _TOTAL_CBS + activation_extra_cbs + 2
        if resident_activation_output_handoff
        else _TOTAL_CBS
    )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mixed resident backward requires the TT-XLA bridge"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    scale_bf16 = float(
        torch.tensor(scale, dtype=torch.bfloat16).float().item()
    )
    scale_bits = struct.unpack("<I", struct.pack("<f", scale_bf16))[0]

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _program(dy, weight, b, hidden, a, inputs, gate, up, arena):
        base_dy = ttl.make_dataflow_buffer_like(
            dy, shape=(1, base_reduction_tiles_per_phase), block_count=2
        )
        base_weight = ttl.make_dataflow_buffer_like(
            weight, shape=(base_reduction_tiles_per_phase, 1), block_count=2
        )
        base_partial = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        base_result = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, jobs_per_node)
        )
        dh_dy = ttl.make_dataflow_buffer_like(
            dy, shape=(1, n_tiles), block_count=1
        )
        dh_b = ttl.make_dataflow_buffer_like(
            b, shape=(n_tiles, 1), block_count=1
        )
        dh_scaled = ttl.make_dataflow_buffer_like(
            dy, shape=(1, n_tiles), block_count=1
        )
        dh_partial = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        dh_output = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, jobs_per_node)
        )
        db_dy = ttl.make_dataflow_buffer_like(
            dy, shape=(m_tiles, 1), block_count=1
        )
        db_hidden = ttl.make_dataflow_buffer_like(
            hidden, shape=(m_tiles, 1), block_count=1
        )
        db_scaled = ttl.make_dataflow_buffer_like(
            dy, shape=(m_tiles, 1), block_count=1
        )
        db_transposed = ttl.make_dataflow_buffer_like(
            hidden, shape=(1, m_tiles), block_count=1
        )
        db_partial = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        db_pretranspose = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        db_output = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        dh_resident = ttl.make_dataflow_buffer_like(
            arena, shape=(1, m_tiles), block_count=1
        )
        tail_a = ttl.make_dataflow_buffer_like(
            a, shape=(1, 1), block_count=max(2, jobs_per_node)
        )
        tail_x = ttl.make_dataflow_buffer_like(
            inputs,
            shape=(m_tiles, 1),
            block_count=max(2, da_k_tiles_per_node),
        )
        transposed_x = ttl.make_dataflow_buffer_like(
            inputs,
            shape=(1, m_tiles),
            block_count=max(2, da_k_tiles_per_node),
        )
        da_partial = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        da_pretranspose = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        da_output = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, 1),
            block_count=max(2, da_k_tiles_per_node),
        )
        lora_dx = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, jobs_per_node)
        )
        final_dx = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, jobs_per_node)
        )

        @ttl.datamovement()
        def read():
            # Device-less ABI probe. All six generated bodies are replaced.
            with base_dy.reserve() as block:
                tx = ttl.copy(dy[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            with base_dy.wait() as _value:
                with final_dx.reserve() as output:
                    # This body exists only to establish the TT-Lang ABI; all
                    # six generated kernels are replaced below. Keep the
                    # placeholder independent of the exact base reduction
                    # phase, which may make ``base_dy`` wider than one tile.
                    output.store(ttl.block.fill(0, shape=output.shape))
            node_col, node_row = ttl.node(dims=2)
            if node_col + node_row < 0:
                with base_weight.wait() as _value:
                    pass
                with base_partial.wait() as _value:
                    pass
                with base_result.wait() as _value:
                    pass
                with dh_dy.wait() as _value:
                    pass
                with dh_b.wait() as _value:
                    pass
                with dh_scaled.wait() as _value:
                    pass
                with dh_partial.wait() as _value:
                    pass
                with dh_output.wait() as _value:
                    pass
                with db_dy.wait() as _value:
                    pass
                with db_hidden.wait() as _value:
                    pass
                with db_scaled.wait() as _value:
                    pass
                with db_transposed.wait() as _value:
                    pass
                with db_partial.wait() as _value:
                    pass
                with db_pretranspose.wait() as _value:
                    pass
                with db_output.wait() as _value:
                    pass
                with dh_resident.wait() as _value:
                    pass
                with tail_a.wait() as _value:
                    pass
                with tail_x.wait() as _value:
                    pass
                with transposed_x.wait() as _value:
                    pass
                with da_partial.wait() as _value:
                    pass
                with da_pretranspose.wait() as _value:
                    pass
                with da_output.wait() as _value:
                    pass
                with lora_dx.wait() as _value:
                    pass

        @ttl.datamovement()
        def write():
            with final_dx.wait() as value:
                tx = ttl.copy(value, arena[0:1, 0:1])
                tx.wait()

    replacements = {
        "@M_TILES@": str(m_tiles),
        "@K_TILES@": str(k_tiles),
        "@N_TILES@": str(n_tiles),
        "@BASE_REDUCTION_TILES@": str(base_reduction_tiles_per_phase),
        "@JOBS_PER_NODE@": str(jobs_per_node),
        "@DA_K_TILES_PER_NODE@": str(da_k_tiles_per_node),
        "@N_TILES_PER_NODE@": str(n_tiles_per_node),
        "@TOTAL_JOBS@": str(total_jobs),
        "@REDUCTION_TILES_PER_PHASE@": str(
            base_reduction_tiles_per_phase
        ),
        "@GRID_COLS@": "10",
        "@NUM_N_BLOCKS@": str(n_tiles // 2),
        "@M_TILES_PER_NODE@": "1",
        "@TOTAL_M_TILES@": str(m_tiles),
        "@TOTAL_N_TILES@": str(n_tiles),
        "@REDUCTION_TILES@": str(m_tiles),
        "@SCALE_BITS@": str(scale_bits),
        "@MAILBOX_SEMAPHORE@": str(mailbox_semaphore),
        "@READY_SEMAPHORE@": str(ready_semaphore),
        "@INIT_ACK_SEMAPHORE@": str(init_ack_semaphore),
        "@NUM_SEMAPHORES@": str(num_semaphores),
        "@TOTAL_CBS@": str(total_cbs),
        "@ACTIVATION_JOBS@": str(activation_jobs),
        "@ACTIVATION_BANKS@": str(activation_banks),
        "@OUTPUT_ADDRESS_SEMAPHORE_START@": str(
            output_address_semaphore_start
        ),
        "@OUTPUT_READY_SEMAPHORE@": str(output_ready_semaphore),
        "@CONSUMER_DONE_SEMAPHORE@": str(consumer_done_semaphore),
        "@OUTPUT_RELEASE_SEMAPHORE@": str(output_release_semaphore),
        "@PULLED_DGATE_ARG@": str(pulled_dgate_arg),
        "@PULLED_DUP_ARG@": str(pulled_dup_arg),
    }
    variant = hashlib.sha256(
        (
            f"{scale_bf16:.17g}:{tokens}:{input_features}:"
            f"{output_features}:{physical_rank}:"
            f"{base_reduction_tiles_per_phase}:"
            f"activation-stage-{activation_diagnostic_stage}:"
            f"l1-output-handoff={resident_activation_output_handoff}:"
            "mixed-down-activation-v1"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.resident_down_activation.{variant}",
        arg_roles=("in",) * 8 + ("out",),
        version_tag=f"resident-down-activation-{variant}",
    )
    def operation(dy, weight, b, hidden, a, inputs, gate, up, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(
            dy, weight, b, hidden, a, inputs, gate, up, arena
        )
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 25:
                    raise RuntimeError(
                        "mixed down/activation probe expected 25 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                # CB indices are five-bit hardware identifiers. The two core
                # ranges are disjoint, so the activation island legally reuses
                # IDs 0..12 rather than consuming impossible IDs 25..37. The
                # extra descriptors are appended after TTL's placeholder has
                # compiled; each still occupies its own artifact position so
                # compile-time argument 25 resolves to local CB id 0, etc.
                activation_template = compiled.cb_configs[24]
                for activation_index in range(13):
                    activation_cb = copy(activation_template)
                    activation_cb.shape = (1, 1)
                    activation_cb.block_count = 2
                    activation_cb._cb_index = activation_index
                    compiled.cb_configs.append(activation_cb)
                if resident_activation_output_handoff:
                    # Output CBs 10/11 retain up to 32 exact activation
                    # tiles. Qwen needs a second bank because each activation
                    # core owns as many as 61 tiles.
                    first_bank_jobs = min(32, activation_jobs)
                    compiled.cb_configs[35].block_count = first_bank_jobs
                    compiled.cb_configs[36].block_count = first_bank_jobs
                    for extra_bank in range(1, activation_banks):
                        for local_id in (13, 14):
                            activation_cb = copy(activation_template)
                            activation_cb.shape = (1, 1)
                            activation_cb.block_count = max(
                                2, activation_jobs - 32 * extra_bank
                            )
                            activation_cb._cb_index = local_id
                            compiled.cb_configs.append(activation_cb)
                    for local_id in (25, 26):
                        pulled_cb = copy(activation_template)
                        pulled_cb.shape = (1, 1)
                        pulled_cb.block_count = max(2, jobs_per_node)
                        pulled_cb._cb_index = local_id
                        compiled.cb_configs.append(pulled_cb)
                for index in (2, 7, 13, 20):
                    compiled.cb_configs[index].tensor = (
                        _FP32_DTYPE_ONLY_TENSOR
                    )

                generated_dir = Path(compiled.kernel_paths[0][0]).parent
                matmul_reader_source = _down_reader_source(
                    replacements,
                    num_semaphores=num_semaphores,
                    init_ack=init_ack_semaphore,
                    total_cbs=total_cbs,
                )
                matmul_writer_source = _replace_all(
                    _cpp("resident_down_activation_matmul_writer.cpp"),
                    {**replacements, "@TOTAL_CBS@": str(total_cbs)},
                )
                activation_compute_source = _activation_compute_source(
                    replacements,
                    producer_jobs=jobs_per_node,
                    total_jobs=total_jobs,
                    diagnostic_stage=activation_diagnostic_stage,
                )
                if resident_activation_output_handoff:
                    if activation_diagnostic_stage != 6:
                        raise ValueError(
                            "L1 output handoff requires the complete exact "
                            "activation stage"
                        )
                    l1_replacements = {
                        **replacements,
                        "@TOTAL_CBS@": str(total_cbs),
                    }
                    matmul_reader_source = _append_before_kernel_close(
                        matmul_reader_source,
                        _replace_all(
                            _cpp(
                                "resident_down_activation_l1_reader_tail.cpp"
                            ),
                            l1_replacements,
                        ),
                    )
                    matmul_writer_source = _append_before_kernel_close(
                        matmul_writer_source,
                        _replace_all(
                            _cpp(
                                "resident_down_activation_l1_matmul_writer_tail.cpp"
                            ),
                            l1_replacements,
                        ),
                    )
                    activation_compute_source = (
                        _banked_activation_compute_source(
                            activation_compute_source,
                            activation_banks=activation_banks,
                        )
                    )
                    activation_writer_source = _replace_all(
                        _cpp("resident_down_activation_l1_writer.cpp"),
                        l1_replacements,
                    )
                else:
                    activation_writer_source = _replace_all(
                        _cpp("resident_down_activation_writer.cpp"),
                        replacements,
                    )
                sources = (
                    (
                        "resident_down_activation_matmul_reader.cpp",
                        "noc",
                        matmul_reader_source,
                    ),
                    (
                        "resident_down_activation_matmul_compute.cpp",
                        "compute",
                        _down_compute_source(replacements),
                    ),
                    (
                        "resident_down_activation_matmul_writer.cpp",
                        "noc",
                        matmul_writer_source,
                    ),
                    (
                        "resident_down_activation_reader.cpp",
                        "noc",
                        _replace_all(
                            _cpp("resident_down_activation_reader.cpp"),
                            replacements,
                        ),
                    ),
                    (
                        "resident_down_activation_compute.cpp",
                        "compute",
                        activation_compute_source,
                    ),
                    (
                        "resident_down_activation_writer.cpp",
                        "noc",
                        activation_writer_source,
                    ),
                )
                kernel_paths = []
                for filename, thread_type, source in sources:
                    path = generated_dir / filename
                    path.write_text(source, encoding="utf-8")
                    kernel_paths.append((str(path), thread_type))

                import ttnn

                fp32_compute = ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                )
                bf16_compute = ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                )
                compiled.kernel_paths = kernel_paths
                compiled.kernel_configs = [
                    ttnn.ReaderConfigDescriptor(),
                    fp32_compute,
                    ttnn.WriterConfigDescriptor(),
                    ttnn.ReaderConfigDescriptor(),
                    bf16_compute,
                    ttnn.WriterConfigDescriptor(),
                ]
                compiled.kernel_tensor_indices = [
                    list(range(6)), [], [8], [6, 7], [],
                    ([] if resident_activation_output_handoff else [8]),
                ]
                compiled.kernel_core_ranges = [
                    _MATMUL_RANGE,
                    _MATMUL_RANGE,
                    _MATMUL_RANGE,
                    _ACTIVATION_RANGE,
                    _ACTIVATION_RANGE,
                    _ACTIVATION_RANGE,
                ]
                compiled.kernel_noc_roles = [
                    "reader", None, "writer", "reader", None, "writer"
                ]
                compiled.cb_core_ranges = (
                    [_MATMUL_RANGE] * 25
                    + [_ACTIVATION_RANGE]
                    * (13 + activation_extra_cbs)
                    + (
                        [_MATMUL_RANGE] * 2
                        if resident_activation_output_handoff
                        else []
                    )
                )
                compiled.semaphore_configs = [
                    {
                        "id": semaphore,
                        "core_type": "worker",
                        "core_range": _ALL_WORKERS_RANGE,
                        "initial_value": 0,
                    }
                    for semaphore in range(num_semaphores)
                ]
                compiled.num_pipe_sync_semaphores = num_semaphores
                compiled.kernel_arg_specs = [[] for _ in range(6)]
            break
        return result

    operation.output_tile_shape = resident_down_activation_output_tile_shape(
        tokens=tokens,
        input_features=input_features,
        output_features=output_features,
    )
    operation.experimental_backend = "mixed_island_v2"
    operation.resident_activation_output_handoff = bool(
        resident_activation_output_handoff
    )
    return operation
