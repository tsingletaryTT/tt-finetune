"""Exact one-call LoRA backward tail for two shared-input PEFT members.

The first native phase has already produced the two frozen-base dX tensors
and the two BF16 dH tensors.  This program keeps all remaining member work
inside one reader/compute/writer schedule:

    lora_dX0 = BF16(dH0 @ A0)
    lora_dX1 = BF16(dH1 @ A1)
    dX = BF16(BF16(BF16(base1 + lora_dX1) + lora_dX0) + base0)
    dA0 = transpose(transpose(X) @ dH0)
    dA1 = transpose(transpose(X) @ dH1)

The staged BF16 additions deliberately preserve the accepted PEFT rounding
contract without materializing the intermediate sums in DRAM.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID
from .tt_lang_lora import TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR


ttl = None
tt_lang_operation = None


def _offset_generated_reader_tensor_arg(
    source: str, argument_index: int, tile_offset: int
) -> str:
    """Offset every generated page id for one custom-call tensor argument."""

    pattern = re.compile(
        r"    int32_t v\d+ = get_common_arg_val<uint32_t>\(v\d+\);\n"
        r"    auto tensor_accessor_args_\d+ = TensorAccessorArgs<"
        r"tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
        + str(argument_index)
        + r", \d+>\(\), \d+>\(\);\n"
        r"    TensorAccessor (v\d+) = TensorAccessor\([^\n]+\);"
    )
    accessors = pattern.findall(source)
    if not accessors:
        raise RuntimeError(
            f"cannot find generated reader tensor arg {argument_index}"
        )
    for accessor in accessors:
        read_pattern = re.compile(
            r"(noc0\.async_read\("
            + accessor
            + r", [^\n]+\.page_id = static_cast<uint32_t>\()"
            r"([^)]+)(\)\}, \{\}\);)"
        )
        source, replacements = read_pattern.subn(
            lambda match: (
                match.group(1)
                + f"({match.group(2)}) + {tile_offset}"
                + match.group(3)
            ),
            source,
            count=1,
        )
        if replacements != 1:
            raise RuntimeError(
                f"cannot find generated read for tensor arg {argument_index}"
            )
    return source


def estimate_grouped2_backward_tail_l1_bytes(
    *,
    token_tiles: int,
    staged_dx_tiles: int,
) -> int:
    """Estimate CB storage per worker, excluding code/runtime reservations."""

    if token_tiles <= 0 or staged_dx_tiles <= 0:
        raise ValueError("grouped-tail tile counts must be positive")
    bf16_tile_bytes = TILE_SIZE * TILE_SIZE * 2
    fp32_tile_bytes = TILE_SIZE * TILE_SIZE * 4
    # Six dX inputs and five staged BF16 dX/add outputs.
    dx_bytes = 11 * staged_dx_tiles * bf16_tile_bytes
    # Per member: dH, X and transposed-X token strips, one FP32 partial,
    # one BF16 pretranspose tile, and one BF16 output tile.
    da_member_bytes = (
        3 * token_tiles * bf16_tile_bytes
        + fp32_tile_bytes
        + 2 * bf16_tile_bytes
    )
    return dx_bytes + 2 * da_member_bytes


def _shift_compile_time_cb_indices(source: str, offset: int) -> str:
    import re

    pattern = re.compile(r"get_compile_time_arg_val\((\d+)\)")
    return pattern.sub(
        lambda match: (
            "get_compile_time_arg_val("
            f"{int(match.group(1)) + offset})"
        ),
        source,
    )


@lru_cache(maxsize=32)
def build_tt_lang_grouped2_frozen_lora_backward_tail(
    *,
    tokens: int,
    input_features: int,
    phase_one_arena_input: bool = False,
    member_output_features: int | None = None,
) -> Callable:
    dimensions = (tokens, input_features)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("grouped tail dimensions must be positive /32")
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    if phase_one_arena_input:
        if (
            member_output_features is None
            or member_output_features <= 0
            or member_output_features % TILE_SIZE
        ):
            raise ValueError(
                "arena-input grouped tail requires tile-aligned member "
                "output features"
            )
    member_n_tiles = (
        int(member_output_features) // TILE_SIZE
        if member_output_features is not None
        else 0
    )
    if m_tiles % 2:
        raise ValueError("exact grouped dA requires even token tiles")
    if k_tiles > 32:
        raise ValueError("grouped tail currently supports K <= 32 tiles")
    total_workers = BLACKHOLE_P150A_GRID[0] * BLACKHOLE_P150A_GRID[1]
    dx_jobs = m_tiles * k_tiles
    jobs_per_node = -(-dx_jobs // total_workers)
    k_tiles_per_node = -(-k_tiles // total_workers)
    staged = max(2, jobs_per_node)
    required_l1 = estimate_grouped2_backward_tail_l1_bytes(
        token_tiles=m_tiles,
        staged_dx_tiles=staged,
    )
    l1_budget = int(1.46 * 1024 * 1024 * 0.75)
    if required_l1 > l1_budget:
        raise ValueError(
            "grouped tail exceeds the 75% per-worker L1 budget: "
            f"{required_l1} > {l1_budget} bytes"
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
                "grouped native tail requires TT-XLA"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    variant = hashlib.sha256(
        (
            f"{tokens}:{input_features}:"
            f"{int(phase_one_arena_input)}:"
            f"{member_n_tiles}:"
            "grouped2-frozen-lora-backward-tail-v8-da-first"
        ).encode("ascii")
    ).hexdigest()[:16]

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _program(
        base0,
        dh0_value,
        a0,
        base1,
        dh1_value,
        a1,
        inputs,
        arena,
    ):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows

        base0_dfb = ttl.make_dataflow_buffer_like(
            base0, shape=(1, 1), block_count=staged
        )
        base1_dfb = ttl.make_dataflow_buffer_like(
            base1, shape=(1, 1), block_count=staged
        )
        dh0_dfb = ttl.make_dataflow_buffer_like(
            dh0_value, shape=(1, 1), block_count=staged
        )
        a0_dfb = ttl.make_dataflow_buffer_like(
            a0, shape=(1, 1), block_count=staged
        )
        dh1_dfb = ttl.make_dataflow_buffer_like(
            dh1_value, shape=(1, 1), block_count=staged
        )
        a1_dfb = ttl.make_dataflow_buffer_like(
            a1, shape=(1, 1), block_count=staged
        )
        lora0_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=staged
        )
        lora1_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=staged
        )
        member1_sum_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=staged
        )
        with_member0_lora_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=staged
        )
        combined_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=staged
        )

        da0_dh_dfb = ttl.make_dataflow_buffer_like(
            dh0_value, shape=(m_tiles, 1), block_count=1
        )
        da0_x_dfb = ttl.make_dataflow_buffer_like(
            inputs, shape=(m_tiles, 1), block_count=1
        )
        da0_transposed_x_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, m_tiles), block_count=1
        )
        da0_partial_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        da0_pretranspose_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        da0_output_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )

        da1_dh_dfb = ttl.make_dataflow_buffer_like(
            dh1_value, shape=(m_tiles, 1), block_count=1
        )
        da1_x_dfb = ttl.make_dataflow_buffer_like(
            inputs, shape=(m_tiles, 1), block_count=1
        )
        da1_transposed_x_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, m_tiles), block_count=1
        )
        da1_partial_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        da1_pretranspose_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        da1_output_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(jobs_per_node):
                job = node_index * jobs_per_node + local_job
                if job < dx_jobs:
                    m_tile = job // k_tiles
                    k_tile = job - m_tile * k_tiles
                    with (
                        base0_dfb.reserve() as base0_block,
                        base1_dfb.reserve() as base1_block,
                        dh0_dfb.reserve() as dh0_block,
                        a0_dfb.reserve() as a0_block,
                        dh1_dfb.reserve() as dh1_block,
                        a1_dfb.reserve() as a1_block,
                    ):
                        tx_base0 = ttl.copy(
                            base0[
                                m_tile : m_tile + 1,
                                k_tile : k_tile + 1,
                            ],
                            base0_block,
                        )
                        tx_base1 = ttl.copy(
                            base1[
                                m_tile : m_tile + 1,
                                k_tile : k_tile + 1,
                            ],
                            base1_block,
                        )
                        tx_dh0 = ttl.copy(
                            dh0_value[m_tile : m_tile + 1, 0:1],
                            dh0_block,
                        )
                        tx_a0 = ttl.copy(
                            a0[0:1, k_tile : k_tile + 1],
                            a0_block,
                        )
                        tx_dh1 = ttl.copy(
                            dh1_value[m_tile : m_tile + 1, 0:1],
                            dh1_block,
                        )
                        tx_a1 = ttl.copy(
                            a1[0:1, k_tile : k_tile + 1],
                            a1_block,
                        )
                        tx_base0.wait()
                        tx_base1.wait()
                        tx_dh0.wait()
                        tx_a0.wait()
                        tx_dh1.wait()
                        tx_a1.wait()

            for local_k in range(k_tiles_per_node):
                k_tile = node_index * k_tiles_per_node + local_k
                if k_tile < k_tiles:
                    with (
                        da0_dh_dfb.reserve() as dh0_block,
                        da0_x_dfb.reserve() as x0_block,
                        da1_dh_dfb.reserve() as dh1_block,
                        da1_x_dfb.reserve() as x1_block,
                    ):
                        tx_dh0 = ttl.copy(
                            dh0_value[0:m_tiles, 0:1], dh0_block
                        )
                        tx_x0 = ttl.copy(
                            inputs[
                                0:m_tiles,
                                k_tile : k_tile + 1,
                            ],
                            x0_block,
                        )
                        tx_dh1 = ttl.copy(
                            dh1_value[0:m_tiles, 0:1], dh1_block
                        )
                        tx_x1 = ttl.copy(
                            inputs[
                                0:m_tiles,
                                k_tile : k_tile + 1,
                            ],
                            x1_block,
                        )
                        tx_dh0.wait()
                        tx_x0.wait()
                        tx_dh1.wait()
                        tx_x1.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(jobs_per_node):
                job = node_index * jobs_per_node + local_job
                if job < dx_jobs:
                    with (
                        dh0_dfb.wait() as dh0_block,
                        a0_dfb.wait() as a0_block,
                    ):
                        with lora0_dfb.reserve() as lora0:
                            lora0.store(ttl.matmul(dh0_block, a0_block))
                    with (
                        dh1_dfb.wait() as dh1_block,
                        a1_dfb.wait() as a1_block,
                    ):
                        with lora1_dfb.reserve() as lora1:
                            lora1.store(ttl.matmul(dh1_block, a1_block))
                    with (
                        base1_dfb.wait() as base1_block,
                        lora1_dfb.wait() as lora1,
                    ):
                        with member1_sum_dfb.reserve() as member1_sum:
                            member1_sum.store(base1_block + lora1)
                    with (
                        member1_sum_dfb.wait() as member1_sum,
                        lora0_dfb.wait() as lora0,
                    ):
                        with (
                            with_member0_lora_dfb.reserve()
                            as with_member0_lora
                        ):
                            with_member0_lora.store(
                                member1_sum + lora0
                            )
                    with (
                        with_member0_lora_dfb.wait()
                        as with_member0_lora,
                        base0_dfb.wait() as base0_block,
                    ):
                        with combined_dfb.reserve() as combined:
                            combined.store(
                                with_member0_lora + base0_block
                            )

            for local_k in range(k_tiles_per_node):
                k_tile = node_index * k_tiles_per_node + local_k
                if k_tile < k_tiles:
                    with da0_x_dfb.wait() as x:
                        with (
                            da0_transposed_x_dfb.reserve()
                            as transposed
                        ):
                            transposed.store(ttl.transpose(x))
                    with (
                        da0_transposed_x_dfb.wait() as transposed,
                        da0_dh_dfb.wait() as dh,
                    ):
                        with da0_partial_dfb.reserve() as partial:
                            partial.store(ttl.matmul(transposed, dh))
                    with da0_partial_dfb.wait() as partial:
                        with (
                            da0_pretranspose_dfb.reserve() as before
                        ):
                            before.store(partial)
                    with da0_pretranspose_dfb.wait() as before:
                        with da0_output_dfb.reserve() as output:
                            output.store(ttl.transpose(before))

                    with da1_x_dfb.wait() as x:
                        with (
                            da1_transposed_x_dfb.reserve()
                            as transposed
                        ):
                            transposed.store(ttl.transpose(x))
                    with (
                        da1_transposed_x_dfb.wait() as transposed,
                        da1_dh_dfb.wait() as dh,
                    ):
                        with da1_partial_dfb.reserve() as partial:
                            partial.store(ttl.matmul(transposed, dh))
                    with da1_partial_dfb.wait() as partial:
                        with (
                            da1_pretranspose_dfb.reserve() as before
                        ):
                            before.store(partial)
                    with da1_pretranspose_dfb.wait() as before:
                        with da1_output_dfb.reserve() as output:
                            output.store(ttl.transpose(before))

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(jobs_per_node):
                job = node_index * jobs_per_node + local_job
                if job < dx_jobs:
                    m_tile = job // k_tiles
                    k_tile = job - m_tile * k_tiles
                    with combined_dfb.wait() as combined:
                        tx = ttl.copy(
                            combined,
                            arena[
                                m_tile : m_tile + 1,
                                k_tile : k_tile + 1,
                            ],
                        )
                        tx.wait()
            for local_k in range(k_tiles_per_node):
                k_tile = node_index * k_tiles_per_node + local_k
                if k_tile < k_tiles:
                    with da0_output_dfb.wait() as da:
                        tx = ttl.copy(
                            da,
                            arena[
                                m_tiles : m_tiles + 1,
                                k_tile : k_tile + 1,
                            ],
                        )
                        tx.wait()
                    with da1_output_dfb.wait() as da:
                        tx = ttl.copy(
                            da,
                            arena[
                                m_tiles + 1 : m_tiles + 2,
                                k_tile : k_tile + 1,
                            ],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.grouped2_backward_tail.{variant}",
        arg_roles=("in", "in", "in", "in", "in", "in", "in", "out"),
        version_tag=f"grouped2-backward-tail-{variant}",
    )
    def grouped_tail(
        base0,
        dh0,
        a0,
        base1,
        dh1,
        a1,
        inputs,
        arena,
    ):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(
            base0, dh0, a0, base1, dh1, a1, inputs, arena
        )
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 23:
                    raise RuntimeError(
                        "grouped tail expected 23 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                compiled.cb_configs[14].tensor = _FP32_DTYPE_ONLY_TENSOR
                compiled.cb_configs[20].tensor = _FP32_DTYPE_ONLY_TENSOR
                compute_paths = [
                    Path(path)
                    for path, thread_type in compiled.kernel_paths
                    if thread_type == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError(
                        "grouped tail expected one compute kernel"
                    )
                if phase_one_arena_input:
                    read_paths = [
                        Path(path)
                        for path, thread_type in compiled.kernel_paths
                        if thread_type != "compute"
                        and "read" in Path(path).name
                    ]
                    if len(read_paths) != 1:
                        raise RuntimeError(
                            "arena-input grouped tail expected one reader"
                        )
                    arena_columns = k_tiles + 1
                    member_rows = m_tiles + member_n_tiles
                    reader_source = read_paths[0].read_text(
                        encoding="utf-8"
                    )
                    reader_source = (
                        _offset_generated_reader_tensor_arg(
                            reader_source,
                            1,
                            k_tiles,
                        )
                    )
                    reader_source = (
                        _offset_generated_reader_tensor_arg(
                            reader_source,
                            3,
                            member_rows * arena_columns,
                        )
                    )
                    reader_source = (
                        _offset_generated_reader_tensor_arg(
                            reader_source,
                            4,
                            member_rows * arena_columns + k_tiles,
                        )
                    )
                    read_paths[0].write_text(
                        reader_source, encoding="utf-8"
                    )
                grid = compiled.core_ranges.bounding_box().grid_size()
                total_nodes = int(grid.x) * int(grid.y)
                replacements = {
                    "@DX_JOBS_PER_NODE@": str(
                        -(-(m_tiles * k_tiles) // total_nodes)
                    ),
                    "@TOTAL_DX_JOBS@": str(m_tiles * k_tiles),
                    "@K_TILES_PER_NODE@": str(
                        -(-k_tiles // total_nodes)
                    ),
                    "@TOTAL_K_TILES@": str(k_tiles),
                    "@REDUCTION_TILES@": str(m_tiles),
                    "@RANK_TILES@": "1",
                    "@GRID_COLS@": str(int(grid.x)),
                }
                cpp_directory = Path(__file__).with_name("cpp")
                da_source = (
                    cpp_directory
                    / "lora_backward_da_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                da0_source = _shift_compile_time_cb_indices(
                    da_source, 11
                ).replace(
                    "void kernel_main()",
                    "FORCE_INLINE void da0_phase()",
                    1,
                )
                da1_source = _shift_compile_time_cb_indices(
                    da_source, 17
                ).replace(
                    "void kernel_main()",
                    "FORCE_INLINE void da1_phase()",
                    1,
                )
                tail_source = (
                    Path(__file__).with_name("cpp")
                    / "grouped2_frozen_lora_backward_tail_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                source = (
                    da0_source
                    + "\n"
                    + da1_source
                    + "\n"
                    + tail_source
                )
                for placeholder, value in replacements.items():
                    source = source.replace(placeholder, value)
                if any(key in source for key in replacements):
                    raise RuntimeError(
                        "unexpanded grouped-tail placeholder"
                    )
                compute_paths[0].write_text(source, encoding="utf-8")
            break
        return result

    return grouped_tail
