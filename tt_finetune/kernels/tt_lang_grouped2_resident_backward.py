"""One-call, device-resident exact backward for two LoRA members.

This is the experimental 60-call building block.  A single TT-Metal program
owns phase one, the L1/NoC handoff, and the exact grouped tail.  The generated
TT-Lang kernels are used only to establish the custom-call ABI, CB allocation,
and tensor accessors; reader/compute/writer bodies are replaced with reviewed
TT-Metal C++ before the first launch.
"""

from __future__ import annotations

import hashlib
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import (
    BLACKHOLE_P150A_CORE_COUNT,
    BLACKHOLE_P150A_GRID,
)
from .tt_lang_frozen_linear_lora_backward import (
    _shift_compile_time_cb_indices,
    estimate_fused_frozen_lora_backward_l1_bytes,
)
from .tt_lang_lora import TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR


ttl = None
tt_lang_operation = None


def resident_output_tile_shape(
    *, tokens: int, input_features: int, member_output_features: int
) -> tuple[int, int]:
    """Physical tile shape of [dX, dA0, dA1, dB0, dB1]."""

    return (
        tokens // TILE_SIZE
        + 2
        + 2 * (member_output_features // TILE_SIZE),
        input_features // TILE_SIZE,
    )


def estimate_resident_grouped_backward_l1_bytes(
    *,
    tokens: int,
    input_features: int,
    member_output_features: int,
    physical_rank: int,
    base_reduction_tiles_per_phase: int = 1,
    member_core_count: int = BLACKHOLE_P150A_CORE_COUNT // 2,
) -> int:
    """Estimate peak per-worker CB storage for the resident program.

    The phase-one estimator assumes all 110 P150a workers share the dX jobs.
    The resident grouped program instead assigns one projection to each
    55-core half, so this function also accounts for the additional local
    base-result and dH staging slots.  Keeping this calculation public lets
    larger block planners reuse the accepted allocation rather than inventing
    an aggregate-L1 estimate.
    """

    dimensions = (
        tokens,
        input_features,
        member_output_features,
        physical_rank,
    )
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("resident grouped dimensions must be positive /32")
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = member_output_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    member_nodes = int(member_core_count)
    if member_nodes <= 0:
        raise ValueError("member core count must be positive")
    jobs_per_node = -(-(m_tiles * k_tiles) // member_nodes)
    required_l1 = estimate_fused_frozen_lora_backward_l1_bytes(
        token_tiles=m_tiles,
        input_tiles=k_tiles,
        output_tiles=n_tiles,
        rank_tiles=rank_tiles,
        base_reduction_tiles_per_phase=base_reduction_tiles_per_phase,
    )
    full_grid_jobs_per_node = -(
        -(m_tiles * k_tiles) // BLACKHOLE_P150A_CORE_COUNT
    )
    grouped_staging_delta = max(
        0, jobs_per_node - full_grid_jobs_per_node
    )
    required_l1 += 2 * grouped_staging_delta * TILE_SIZE * TILE_SIZE
    required_l1 += TILE_SIZE * TILE_SIZE * (4 + 2 * m_tiles)
    return required_l1


def resident_backward_support_error(
    *,
    tokens: int,
    input_features: int,
    member_output_features: int,
    physical_rank: int,
    base_reduction_tiles_per_phase: int = 1,
    matmul_island_50x2: bool = False,
) -> str | None:
    """Return why the resident schedule is unsupported, or ``None``.

    This predicate has no TT-XLA dependency and is used before autograd graph
    construction.  Unsupported dynamic shapes therefore fall back to the
    accepted three-call path rather than partially entering a custom call.
    """

    dimensions = (
        tokens,
        input_features,
        member_output_features,
        physical_rank,
    )
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        return "resident grouped dimensions must be positive /32"
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = member_output_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    if rank_tiles > 4 or m_tiles % 2 or n_tiles % 2:
        return "resident exact path requires rank <=128 and even M/N"
    if k_tiles > 32:
        return "resident exact path currently supports K <= 32 tiles"
    if (
        base_reduction_tiles_per_phase <= 0
        or n_tiles % base_reduction_tiles_per_phase
    ):
        return "base reduction phase must divide N tiles"

    member_nodes = 50 if matmul_island_50x2 else (
        BLACKHOLE_P150A_CORE_COUNT // 2
    )
    jobs_per_node = -(-(m_tiles * k_tiles) // member_nodes)
    # TT-Lang CB block counts are limited to 32. The original resident
    # prototype admitted only two jobs because that was the first SmolLM2
    # schedule validated on silicon. The kernels already loop over the
    # compile-time job count, and Qwen's 4x28 tile grid needs three jobs on
    # the busiest worker in each 55-core half.
    if jobs_per_node > 32:
        return (
            "resident exact path supports at most 32 dX jobs/core"
        )

    required_l1 = estimate_resident_grouped_backward_l1_bytes(
        tokens=tokens,
        input_features=input_features,
        member_output_features=member_output_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=base_reduction_tiles_per_phase,
        member_core_count=member_nodes,
    )
    l1_budget = int(1.46 * 1024 * 1024 * 0.75)
    if required_l1 > l1_budget:
        return (
            "resident grouped backward exceeds 75% worker L1: "
            f"{required_l1} > {l1_budget}"
        )
    return None


@lru_cache(maxsize=32)
def build_tt_lang_grouped2_resident_backward(
    *,
    scale: float,
    tokens: int,
    input_features: int,
    member_output_features: int,
    physical_rank: int,
    base_reduction_tiles_per_phase: int = 1,
    matmul_island_50x2: bool = False,
) -> Callable:
    support_error = resident_backward_support_error(
        tokens=tokens,
        input_features=input_features,
        member_output_features=member_output_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=base_reduction_tiles_per_phase,
        matmul_island_50x2=matmul_island_50x2,
    )
    if support_error is not None:
        raise ValueError(support_error)
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = member_output_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE

    member_nodes = 50 if matmul_island_50x2 else (
        BLACKHOLE_P150A_CORE_COUNT // 2
    )
    grid_cols = 10 if matmul_island_50x2 else 11
    jobs = m_tiles * k_tiles
    jobs_per_node = -(-jobs // member_nodes)
    n_tiles_per_node = -(-n_tiles // member_nodes)

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "resident grouped backward requires TT-XLA"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    scale_bf16 = float(
        torch.tensor(scale, dtype=torch.bfloat16).float().item()
    )
    scale_bits = struct.unpack("<I", struct.pack("<f", scale_bf16))[0]
    arena_tile_rows, arena_tile_cols = resident_output_tile_shape(
        tokens=tokens,
        input_features=input_features,
        member_output_features=member_output_features,
    )
    variant = hashlib.sha256(
        (
            f"{scale_bf16:.17g}:{tokens}:{input_features}:"
            f"{member_output_features}:{physical_rank}:"
            f"{base_reduction_tiles_per_phase}:"
            f"grouped2-resident-backward-v1:island50={matmul_island_50x2}"
        ).encode("ascii")
    ).hexdigest()[:16]

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _program(
        dy0,
        weight0,
        b0,
        hidden0,
        a0,
        dy1,
        weight1,
        b1,
        hidden1,
        a1,
        inputs,
        arena,
    ):
        # Keep the accepted 16-CB phase-one allocation and add dedicated
        # dH, transpose, and peer-sum windows. This preserves one producer
        # and one consumer RISC per CB across the complete program.
        base_dy = ttl.make_dataflow_buffer_like(
            dy0,
            shape=(1, base_reduction_tiles_per_phase),
            block_count=2,
        )
        base_weight = ttl.make_dataflow_buffer_like(
            weight0,
            shape=(base_reduction_tiles_per_phase, 1),
            block_count=2,
        )
        base_partial = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        base_result = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, jobs_per_node)
        )
        dh_dy = ttl.make_dataflow_buffer_like(
            dy0, shape=(1, n_tiles), block_count=1
        )
        dh_b = ttl.make_dataflow_buffer_like(
            b0, shape=(n_tiles, rank_tiles), block_count=1
        )
        dh_scaled = ttl.make_dataflow_buffer_like(
            dy0, shape=(1, n_tiles), block_count=1
        )
        dh_partial = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )
        dh_output = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=max(2, jobs_per_node)
        )
        db_dy = ttl.make_dataflow_buffer_like(
            dy0, shape=(m_tiles, 1), block_count=1
        )
        db_hidden = ttl.make_dataflow_buffer_like(
            hidden0, shape=(m_tiles, rank_tiles), block_count=1
        )
        db_scaled = ttl.make_dataflow_buffer_like(
            dy0, shape=(m_tiles, 1), block_count=1
        )
        db_transposed = ttl.make_dataflow_buffer_like(
            hidden0, shape=(1, m_tiles), block_count=1
        )
        db_partial = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )
        db_pretranspose = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )
        db_output = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )
        dh_resident = ttl.make_dataflow_buffer_like(
            arena, shape=(rank_tiles, m_tiles), block_count=1
        )
        tail_transposed_x = ttl.make_dataflow_buffer_like(
            arena, shape=(1, m_tiles), block_count=1
        )
        peer_sum = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, jobs_per_node)
        )

        @ttl.datamovement()
        def read():
            # ABI probes: after lowering, the reader body is fully replaced.
            # Referencing every input establishes deterministic tensor indices.
            with base_dy.reserve() as block:
                tx = ttl.copy(dy0[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(weight0[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(b0[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(hidden0[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(a0[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(dy1[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(weight1[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(b1[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(hidden1[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(a1[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(inputs[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            # A valid minimal pipeline for lowering; never reaches silicon
            # because all three generated sources are replaced below.
            for _ in range(11):
                with base_dy.wait() as value:
                    with base_result.reserve() as output:
                        output.store(value)
            node_col, node_row = ttl.node(dims=2)
            impossible = node_row * 11 + node_col < 0
            if impossible:
                # Lexical references keep every CB in the compiled descriptor.
                with base_weight.wait() as _v:
                    pass
                with base_partial.wait() as _v:
                    pass
                with base_result.wait() as _v:
                    pass
                with dh_dy.wait() as _v:
                    pass
                with dh_b.wait() as _v:
                    pass
                with dh_scaled.wait() as _v:
                    pass
                with dh_partial.wait() as _v:
                    pass
                with dh_output.wait() as _v:
                    pass
                with db_dy.wait() as _v:
                    pass
                with db_hidden.wait() as _v:
                    pass
                with db_scaled.wait() as _v:
                    pass
                with db_transposed.wait() as _v:
                    pass
                with db_partial.wait() as _v:
                    pass
                with db_pretranspose.wait() as _v:
                    pass
                with db_output.wait() as _v:
                    pass
                with dh_resident.wait() as _v:
                    pass
                with tail_transposed_x.wait() as _v:
                    pass
                with peer_sum.wait() as _v:
                    pass

        @ttl.datamovement()
        def write():
            for _ in range(11):
                with base_result.wait() as value:
                    tx = ttl.copy(value, arena[0:1, 0:1])
                    tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.grouped2_resident_backward.{variant}",
        arg_roles=(
            "in",
            "in",
            "in",
            "in",
            "in",
            "in",
            "in",
            "in",
            "in",
            "in",
            "in",
            "out",
        ),
        version_tag=f"grouped2-resident-backward-{variant}",
    )
    def resident_backward(
        dy0,
        weight0,
        b0,
        hidden0,
        a0,
        dy1,
        weight1,
        b1,
        hidden1,
        a1,
        inputs,
        arena,
    ):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(
            dy0,
            weight0,
            b0,
            hidden0,
            a0,
            dy1,
            weight1,
            b1,
            hidden1,
            a1,
            inputs,
            arena,
        )
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 19:
                    raise RuntimeError(
                        "resident backward expected 19 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                for index in (2, 7, 13):
                    compiled.cb_configs[index].tensor = (
                        _FP32_DTYPE_ONLY_TENSOR
                    )
                compiled.num_pipe_sync_semaphores = 11

                reader_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread_type) in enumerate(
                        compiled.kernel_paths
                    )
                    if thread_type != "compute"
                    and "read" in Path(path).name
                ]
                writer_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread_type) in enumerate(
                        compiled.kernel_paths
                    )
                    if thread_type != "compute"
                    and "write" in Path(path).name
                ]
                if (
                    len(reader_indices) != 1
                    or sorted(reader_indices[0]) != list(range(11))
                ):
                    raise RuntimeError(
                        "resident reader ABI changed: "
                        f"{reader_indices!r}"
                    )
                if writer_indices != [[11]]:
                    raise RuntimeError(
                        "resident writer ABI changed: "
                        f"{writer_indices!r}"
                    )

                cpp_dir = Path(__file__).with_name("cpp")
                replacements = {
                    "@M_TILES@": str(m_tiles),
                    "@K_TILES@": str(k_tiles),
                    "@N_TILES@": str(n_tiles),
                    "@BASE_REDUCTION_TILES@": str(
                        base_reduction_tiles_per_phase
                    ),
                    "@JOBS_PER_NODE@": str(jobs_per_node),
                    "@TOTAL_JOBS@": str(jobs),
                    "@REDUCTION_TILES_PER_PHASE@": str(
                        base_reduction_tiles_per_phase
                    ),
                    "@GRID_COLS@": str(grid_cols),
                    "@NUM_N_BLOCKS@": str(n_tiles // 2),
                    "@M_TILES_PER_NODE@": "1",
                    "@TOTAL_M_TILES@": str(m_tiles),
                    "@N_TILES_PER_NODE@": str(n_tiles_per_node),
                    "@TOTAL_N_TILES@": str(n_tiles),
                    "@REDUCTION_TILES@": str(m_tiles),
                    "@SCALE_BITS@": str(scale_bits),
                    "@MEMBER_NODES@": str(member_nodes),
                    "@DX_JOBS_PER_NODE@": str(jobs_per_node),
                    "@TOTAL_DX_JOBS@": str(jobs),
                    "@K_TILES_PER_NODE@": "1",
                    "@TOTAL_K_TILES@": str(k_tiles),
                    "@RANK_TILES@": str(rank_tiles),
                }

                lora_template = (
                    cpp_dir / "lora_backward_arena_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                lora_source = _shift_compile_time_cb_indices(
                    lora_template, 4
                ).replace(
                    "void kernel_main()",
                    "FORCE_INLINE void lora_backward_phase()",
                    1,
                )
                base_source = (
                    cpp_dir / "frozen_linear_backward_base_phase.cpp"
                ).read_text(encoding="utf-8").replace(
                    "void kernel_main()",
                    "FORCE_INLINE void resident_phase_one()",
                    1,
                )
                partition_source = """const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();"""
                local_source = """const uint32_t global_node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();
    constexpr uint32_t member_nodes = @MEMBER_NODES@;
    const uint32_t node_index = global_node_index % member_nodes;"""
                phase_source = (lora_source + "\n" + base_source).replace(
                    partition_source, local_source
                )

                helper_source = (
                    cpp_dir
                    / "grouped2_frozen_lora_backward_tail_exact_compute.cpp"
                ).read_text(encoding="utf-8").split(
                    "\nvoid kernel_main()", 1
                )[0]
                # The resident LoRA-dX phase already waits for the complete
                # m-tile dH window and deliberately leaves it at the CB
                # front. Reissuing progressively smaller wait_front calls in
                # dA can leave the Blackhole CB waiter armed after the first
                # consumer. The data is already proven ready, so dA only
                # advances it with the existing pop_front calls.
                helper_source = helper_source.replace(
                    "            dh.wait_front(reduction_block_tiles);\n",
                    (
                        "            // dH was made fully resident and "
                        "waited before dA.\n"
                    ),
                )
                # dH has two consumers in the resident program. Traverse it
                # by absolute tile index during dA instead of advancing the
                # CB front, preserving the complete window for LoRA dX.
                helper_source = helper_source.replace(
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
                        "                    block * "
                        "reduction_block_tiles + inner,\n"
                    ),
                )
                helper_source = helper_source.replace(
                    "            dh.pop_front(reduction_block_tiles);\n",
                    (
                        "            // Keep dH resident for the following "
                        "LoRA dX consumer.\n"
                    ),
                )
                tail_source = (
                    cpp_dir / "grouped2_resident_backward_tail.cpp"
                ).read_text(encoding="utf-8")
                if matmul_island_50x2:
                    tail_source = tail_source.replace(
                        "constexpr uint32_t grid_cols = 11;",
                        "constexpr uint32_t grid_cols = 10;",
                    ).replace(
                        "constexpr uint32_t member_nodes = 55;",
                        "constexpr uint32_t member_nodes = 50;",
                    )
                compute_source = (
                    phase_source + "\n" + helper_source + "\n" + tail_source
                )
                reader_source = (
                    cpp_dir / "grouped2_resident_backward_reader.cpp"
                ).read_text(encoding="utf-8")
                if matmul_island_50x2:
                    reader_source = reader_source.replace(
                        "constexpr uint32_t kGridCols = 11;\n"
                        "constexpr uint32_t kMemberNodes = 55;",
                        "constexpr uint32_t kGridCols = 10;\n"
                        "constexpr uint32_t kMemberNodes = 50;",
                    ).replace(
                        "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
                        "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};",
                        "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
                        "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};",
                    )
                reader_common_position = {
                    tensor_index: common_position
                    for common_position, tensor_index in enumerate(
                        reader_indices[0]
                    )
                }
                reader_source = re.sub(
                    r"get_common_arg_val<uint32_t>\((\d+)\)",
                    lambda match: (
                        "get_common_arg_val<uint32_t>("
                        f"{reader_common_position[int(match.group(1))]})"
                    ),
                    reader_source,
                )
                reader_source = re.sub(
                    (
                        r"get_tensor_accessor_args_cta_offset<"
                        r"(\d+), 19>\(\),\s*(\d+)"
                    ),
                    lambda match: (
                        "get_tensor_accessor_args_cta_offset<"
                        f"{match.group(1)}, 19>(), "
                        f"{reader_common_position[int(match.group(1))]}"
                    ),
                    reader_source,
                )
                writer_source = (
                    cpp_dir / "grouped2_resident_backward_writer.cpp"
                ).read_text(encoding="utf-8")
                if matmul_island_50x2:
                    writer_source = writer_source.replace(
                        "constexpr uint32_t kGridCols = 11;\n"
                        "constexpr uint32_t kMemberNodes = 55;",
                        "constexpr uint32_t kGridCols = 10;\n"
                        "constexpr uint32_t kMemberNodes = 50;",
                    ).replace(
                        "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
                        "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};",
                        "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
                        "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};",
                    )
                sources = {
                    "compute": compute_source,
                    "read": reader_source,
                    "write": writer_source,
                }
                for key, source in tuple(sources.items()):
                    for placeholder, value in replacements.items():
                        source = source.replace(placeholder, value)
                    unresolved = [
                        placeholder
                        for placeholder in replacements
                        if placeholder in source
                    ]
                    if unresolved:
                        raise RuntimeError(
                            f"unexpanded resident {key}: {unresolved}"
                        )
                    sources[key] = source

                for path, thread_type in compiled.kernel_paths:
                    kind = (
                        "compute"
                        if thread_type == "compute"
                        else (
                            "read"
                            if "read" in Path(path).name
                            else "write"
                        )
                    )
                    Path(path).write_text(
                        sources[kind], encoding="utf-8"
                    )
                if matmul_island_50x2:
                    matmul_range = {
                        "start": [0, 0], "end": [9, 9]
                    }
                    compiled.kernel_core_ranges = [
                        matmul_range for _ in compiled.kernel_paths
                    ]
                    compiled.kernel_noc_roles = [
                        (
                            None
                            if thread_type == "compute"
                            else (
                                "reader"
                                if "read" in Path(path).name
                                else "writer"
                            )
                        )
                        for path, thread_type in compiled.kernel_paths
                    ]
                    compiled.cb_core_ranges = [matmul_range] * 19
                    compiled.semaphore_configs = [
                        {
                            "id": semaphore,
                            "core_type": "worker",
                            "core_range": matmul_range,
                            "initial_value": 0,
                        }
                        for semaphore in range(11)
                    ]
                    compiled.kernel_arg_specs = [
                        [] for _ in compiled.kernel_paths
                    ]
            break
        return result

    resident_backward.output_tile_shape = (
        arena_tile_rows,
        arena_tile_cols,
    )
    resident_backward.matmul_island_50x2 = bool(matmul_island_50x2)
    return resident_backward
