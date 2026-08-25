"""One-call exact phase-one backward for two equal-shape LoRA members.

The two members reuse one 16-CB working set in sequence. Reader, compute, and
writer remain concurrent, but there is no PJRT/XLA boundary between members.
The output arena stores two ordinary member arenas back-to-back:

    member 0 rows [0:M+N]
    member 1 rows [M+N:2*(M+N)]

Within each member:

    rows [0:M],   cols [0:K]   = frozen-base dX
    rows [0:M],   cols [K:K+R] = LoRA dH
    rows [M:M+N], cols [0:R]   = LoRA dB
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


def _partition_generated_node_index(
    source: str, member_nodes: int
) -> tuple[str, str]:
    """Remap a generated full-grid kernel to one of two local core groups."""

    pattern = re.compile(
        r"(  size_t (v\d+) = get_absolute_logical_x\(\);\n"
        r"  size_t (v\d+) = get_absolute_logical_y\(\);\n"
        r"  size_t (v\d+) = \3 \* (v\d+);\n)"
        r"  size_t (v\d+) = \4 \+ \2;"
    )
    match = pattern.search(source)
    if match is None:
        raise RuntimeError("cannot find generated logical node index")
    prefix, x_value, _y_value, row_value, _grid_value, node_value = (
        match.groups()
    )
    replacement = (
        prefix
        + f"  size_t global_node_index = {row_value} + {x_value};\n"
        + f"  constexpr size_t member_nodes = {member_nodes};\n"
        + "  size_t member = global_node_index / member_nodes;\n"
        + f"  size_t {node_value} = global_node_index"
        " - member * member_nodes;"
    )
    return source[: match.start()] + replacement + source[match.end() :], (
        node_value
    )


def _partition_generated_reader(source: str, member_nodes: int) -> str:
    source, _ = _partition_generated_node_index(source, member_nodes)
    declaration_pattern = re.compile(
        r"^  ptrdiff_t v\d+ = .*;\n"
        r"  ptrdiff_t v\d+ = .*;\n"
        r"  bool (v\d+) = .*;\n"
        r"(?=  if \(\1\) \{)",
        re.MULTILINE,
    )
    declarations = list(declaration_pattern.finditer(source))
    if len(declarations) != 2:
        raise RuntimeError(
            "generated grouped reader must contain two shared conditions"
        )
    declaration_source = "".join(
        match.group(0) for match in declarations
    )
    for match in reversed(declarations):
        source = source[: match.start()] + source[match.end() :]

    base_starts = [
        match.start()
        for match in re.finditer(
            r"^  for \(size_t i\d+ = [^\n]+\) \{$",
            source,
            re.MULTILINE,
        )
    ]
    if len(base_starts) != 2:
        raise RuntimeError(
            "generated grouped reader must contain two base sections"
        )
    first_start, second_start = base_starts
    source = (
        source[:first_start]
        + declaration_source
        + source[first_start:]
    )
    first_start += len(declaration_source)
    second_start += len(declaration_source)
    return_position = source.rfind("  return;")
    if return_position <= second_start:
        raise RuntimeError("cannot find generated reader return")
    return (
        source[:first_start]
        + "  if (member == 0) {\n"
        + source[first_start:second_start]
        + "  } else {\n"
        + source[second_start:return_position]
        + "  }\n"
        + source[return_position:]
    )


def _partition_generated_writer(source: str, member_nodes: int) -> str:
    source, _ = _partition_generated_node_index(source, member_nodes)
    loop = re.search(
        r"  for \(size_t (i\d+) = v\d+; "
        r"\1 < v\d+; \1 \+= v\d+\) \{",
        source,
    )
    if loop is None:
        raise RuntimeError("cannot find generated two-member writer loop")
    replacement = f"  {{\n    const size_t {loop.group(1)} = member;"
    return source[: loop.start()] + replacement + source[loop.end() :]


@lru_cache(maxsize=32)
def build_tt_lang_grouped2_frozen_linear_lora_backward(
    *,
    scale: float,
    tokens: int,
    input_features: int,
    member_output_features: int,
    physical_rank: int,
    base_reduction_tiles_per_phase: int = 1,
    concurrent_members: bool = False,
) -> Callable:
    dimensions = (
        tokens,
        input_features,
        member_output_features,
        physical_rank,
    )
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("grouped phase-one dimensions must be positive /32")
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = member_output_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    if (
        base_reduction_tiles_per_phase <= 0
        or n_tiles % base_reduction_tiles_per_phase
    ):
        raise ValueError(
            "base dX reduction phase must divide member output tiles"
        )
    if rank_tiles != 1 or m_tiles % 2 or n_tiles % 2:
        raise ValueError(
            "exact grouped phase one requires rank 32 and even M/N tiles"
        )
    required_l1 = estimate_fused_frozen_lora_backward_l1_bytes(
        token_tiles=m_tiles,
        input_tiles=k_tiles,
        output_tiles=n_tiles,
        rank_tiles=rank_tiles,
        base_reduction_tiles_per_phase=(
            base_reduction_tiles_per_phase
        ),
    )
    l1_budget = int(1.46 * 1024 * 1024 * 0.75)
    if required_l1 > l1_budget:
        raise ValueError(
            "shared-CB grouped phase one exceeds 75% worker L1: "
            f"{required_l1} > {l1_budget}"
        )

    base_jobs = m_tiles * k_tiles
    scheduling_nodes = (
        BLACKHOLE_P150A_CORE_COUNT // 2
        if concurrent_members
        else BLACKHOLE_P150A_CORE_COUNT
    )
    base_jobs_per_node = -(-base_jobs // scheduling_nodes)
    base_reduction_blocks = (
        n_tiles // base_reduction_tiles_per_phase
    )
    member_row_tiles = m_tiles + n_tiles

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "grouped native phase one requires TT-XLA"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    scale_bf16 = float(
        torch.tensor(scale, dtype=torch.bfloat16).float().item()
    )
    scale_bits = struct.unpack(
        "<I", struct.pack("<f", scale_bf16)
    )[0]
    variant = hashlib.sha256(
        (
            f"{scale_bf16:.17g}:{tokens}:{input_features}:"
            f"{member_output_features}:{physical_rank}:"
            f"{base_reduction_tiles_per_phase}:"
            f"{int(concurrent_members)}:"
            "grouped2-phase1-v5-shared-cbs"
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
        dy1,
        weight1,
        b1,
        hidden1,
        arena,
    ):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        local_base_jobs = -(-base_jobs // scheduling_nodes)
        m_tiles_per_node = -(-m_tiles // scheduling_nodes)
        n_tiles_per_node = -(-n_tiles // scheduling_nodes)

        base_dy_dfb = ttl.make_dataflow_buffer_like(
            dy0,
            shape=(1, base_reduction_tiles_per_phase),
            block_count=2,
        )
        base_weight_dfb = ttl.make_dataflow_buffer_like(
            weight0,
            shape=(base_reduction_tiles_per_phase, 1),
            block_count=2,
        )
        base_partial_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        base_result_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, 1),
            block_count=max(2, base_jobs_per_node),
        )

        dh_dy_dfb = ttl.make_dataflow_buffer_like(
            dy0, shape=(1, n_tiles), block_count=1
        )
        dh_b_dfb = ttl.make_dataflow_buffer_like(
            b0, shape=(n_tiles, rank_tiles), block_count=1
        )
        dh_scaled_dy_dfb = ttl.make_dataflow_buffer_like(
            dy0, shape=(1, n_tiles), block_count=1
        )
        dh_partial_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )
        dh_output_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )

        db_dy_dfb = ttl.make_dataflow_buffer_like(
            dy0, shape=(m_tiles, 1), block_count=1
        )
        db_hidden_dfb = ttl.make_dataflow_buffer_like(
            hidden0, shape=(m_tiles, rank_tiles), block_count=1
        )
        db_scaled_dy_dfb = ttl.make_dataflow_buffer_like(
            dy0, shape=(m_tiles, 1), block_count=1
        )
        db_transposed_hidden_dfb = ttl.make_dataflow_buffer_like(
            hidden0, shape=(rank_tiles, m_tiles), block_count=1
        )
        db_partial_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(rank_tiles, 1), block_count=1
        )
        db_pretranspose_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(rank_tiles, 1), block_count=1
        )
        db_output_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col

            for local_job in range(local_base_jobs):
                job = node_index * local_base_jobs + local_job
                if job < base_jobs:
                    m_tile = job // k_tiles
                    k_tile = job - m_tile * k_tiles
                    for block in range(base_reduction_blocks):
                        n_start = (
                            block * base_reduction_tiles_per_phase
                        )
                        with (
                            base_dy_dfb.reserve() as dy,
                            base_weight_dfb.reserve() as weight,
                        ):
                            tx_dy = ttl.copy(
                                dy0[
                                    m_tile : m_tile + 1,
                                    n_start : n_start
                                    + base_reduction_tiles_per_phase,
                                ],
                                dy,
                            )
                            tx_weight = ttl.copy(
                                weight0[
                                    n_start : n_start
                                    + base_reduction_tiles_per_phase,
                                    k_tile : k_tile + 1,
                                ],
                                weight,
                            )
                            tx_dy.wait()
                            tx_weight.wait()
            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    with (
                        dh_dy_dfb.reserve() as dy,
                        dh_b_dfb.reserve() as b,
                    ):
                        tx_dy = ttl.copy(
                            dy0[m_tile : m_tile + 1, 0:n_tiles], dy
                        )
                        tx_b = ttl.copy(b0[0:n_tiles, 0:rank_tiles], b)
                        tx_dy.wait()
                        tx_b.wait()
            for local_n in range(n_tiles_per_node):
                n_tile = node_index * n_tiles_per_node + local_n
                if n_tile < n_tiles:
                    with (
                        db_dy_dfb.reserve() as dy,
                        db_hidden_dfb.reserve() as hidden,
                    ):
                        tx_dy = ttl.copy(
                            dy0[0:m_tiles, n_tile : n_tile + 1], dy
                        )
                        tx_hidden = ttl.copy(
                            hidden0[0:m_tiles, 0:rank_tiles], hidden
                        )
                        tx_dy.wait()
                        tx_hidden.wait()

            for local_job in range(local_base_jobs):
                job = node_index * local_base_jobs + local_job
                if job < base_jobs:
                    m_tile = job // k_tiles
                    k_tile = job - m_tile * k_tiles
                    for block in range(base_reduction_blocks):
                        n_start = (
                            block * base_reduction_tiles_per_phase
                        )
                        with (
                            base_dy_dfb.reserve() as dy,
                            base_weight_dfb.reserve() as weight,
                        ):
                            tx_dy = ttl.copy(
                                dy1[
                                    m_tile : m_tile + 1,
                                    n_start : n_start
                                    + base_reduction_tiles_per_phase,
                                ],
                                dy,
                            )
                            tx_weight = ttl.copy(
                                weight1[
                                    n_start : n_start
                                    + base_reduction_tiles_per_phase,
                                    k_tile : k_tile + 1,
                                ],
                                weight,
                            )
                            tx_dy.wait()
                            tx_weight.wait()
            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    with (
                        dh_dy_dfb.reserve() as dy,
                        dh_b_dfb.reserve() as b,
                    ):
                        tx_dy = ttl.copy(
                            dy1[m_tile : m_tile + 1, 0:n_tiles], dy
                        )
                        tx_b = ttl.copy(b1[0:n_tiles, 0:rank_tiles], b)
                        tx_dy.wait()
                        tx_b.wait()
            for local_n in range(n_tiles_per_node):
                n_tile = node_index * n_tiles_per_node + local_n
                if n_tile < n_tiles:
                    with (
                        db_dy_dfb.reserve() as dy,
                        db_hidden_dfb.reserve() as hidden,
                    ):
                        tx_dy = ttl.copy(
                            dy1[0:m_tiles, n_tile : n_tile + 1], dy
                        )
                        tx_hidden = ttl.copy(
                            hidden1[0:m_tiles, 0:rank_tiles], hidden
                        )
                        tx_dy.wait()
                        tx_hidden.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for _member in range(2):
                for local_job in range(local_base_jobs):
                    job = node_index * local_base_jobs + local_job
                    if job < base_jobs:
                        with base_partial_dfb.reserve() as partial:
                            partial.store(
                                ttl.block.fill(0, shape=partial.shape)
                            )
                        for _ in range(base_reduction_blocks):
                            with (
                                base_dy_dfb.wait() as dy,
                                base_weight_dfb.wait() as weight,
                                base_partial_dfb.wait() as previous,
                            ):
                                with (
                                    base_partial_dfb.reserve()
                                    as next_value
                                ):
                                    next_value.store(
                                        previous + ttl.matmul(dy, weight)
                                    )
                        with base_partial_dfb.wait() as partial:
                            with base_result_dfb.reserve() as result:
                                result.store(partial)

                for local_m in range(m_tiles_per_node):
                    m_tile = node_index * m_tiles_per_node + local_m
                    if m_tile < m_tiles:
                        with dh_dy_dfb.wait() as dy:
                            with (
                                dh_scaled_dy_dfb.reserve() as scaled
                            ):
                                scaled.store(dy * scale_bf16)
                        with (
                            dh_scaled_dy_dfb.wait() as scaled,
                            dh_b_dfb.wait() as b,
                        ):
                            with dh_partial_dfb.reserve() as partial:
                                partial.store(ttl.matmul(scaled, b))
                        with dh_partial_dfb.wait() as partial:
                            with dh_output_dfb.reserve() as dh:
                                dh.store(partial)

                for local_n in range(n_tiles_per_node):
                    n_tile = node_index * n_tiles_per_node + local_n
                    if n_tile < n_tiles:
                        with db_dy_dfb.wait() as dy:
                            with (
                                db_scaled_dy_dfb.reserve() as scaled
                            ):
                                scaled.store(dy * scale_bf16)
                        with db_hidden_dfb.wait() as hidden:
                            with (
                                db_transposed_hidden_dfb.reserve()
                                as transposed
                            ):
                                transposed.store(ttl.transpose(hidden))
                        with (
                            db_transposed_hidden_dfb.wait() as transposed,
                            db_scaled_dy_dfb.wait() as scaled,
                        ):
                            with db_partial_dfb.reserve() as partial:
                                partial.store(
                                    ttl.matmul(transposed, scaled)
                                )
                        with db_partial_dfb.wait() as partial:
                            with (
                                db_pretranspose_dfb.reserve() as before
                            ):
                                before.store(partial)
                        with db_pretranspose_dfb.wait() as before:
                            with db_output_dfb.reserve() as db:
                                db.store(before)

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for member in range(2):
                member_offset = member * member_row_tiles
                for local_job in range(local_base_jobs):
                    job = node_index * local_base_jobs + local_job
                    if job < base_jobs:
                        m_tile = job // k_tiles
                        k_tile = job - m_tile * k_tiles
                        with base_result_dfb.wait() as result:
                            tx = ttl.copy(
                                result,
                                arena[
                                    member_offset
                                    + m_tile : member_offset
                                    + m_tile
                                    + 1,
                                    k_tile : k_tile + 1,
                                ],
                            )
                            tx.wait()
                for local_m in range(m_tiles_per_node):
                    m_tile = node_index * m_tiles_per_node + local_m
                    if m_tile < m_tiles:
                        with dh_output_dfb.wait() as dh:
                            tx = ttl.copy(
                                dh,
                                arena[
                                    member_offset
                                    + m_tile : member_offset
                                    + m_tile
                                    + 1,
                                    k_tiles : k_tiles + rank_tiles,
                                ],
                            )
                            tx.wait()
                for local_n in range(n_tiles_per_node):
                    n_tile = node_index * n_tiles_per_node + local_n
                    if n_tile < n_tiles:
                        with db_output_dfb.wait() as db:
                            tx = ttl.copy(
                                db,
                                arena[
                                    member_offset
                                    + m_tiles
                                    + n_tile : member_offset
                                    + m_tiles
                                    + n_tile
                                    + 1,
                                    0:rank_tiles,
                                ],
                            )
                            tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.grouped2_phase1.{variant}",
        arg_roles=(
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
        version_tag=f"grouped2-phase1-{variant}",
    )
    def grouped_phase_one(
        dy0,
        weight0,
        b0,
        hidden0,
        dy1,
        weight1,
        b1,
        hidden1,
        arena,
    ):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(
            dy0,
            weight0,
            b0,
            hidden0,
            dy1,
            weight1,
            b1,
            hidden1,
            arena,
        )
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 16:
                    raise RuntimeError(
                        "grouped phase one expected 16 shared CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                for index in (2, 7, 13):
                    compiled.cb_configs[index].tensor = (
                        _FP32_DTYPE_ONLY_TENSOR
                    )
                compute_paths = [
                    Path(path)
                    for path, thread_type in compiled.kernel_paths
                    if thread_type == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError(
                        "grouped phase one expected one compute kernel"
                    )
                if concurrent_members:
                    read_paths = [
                        Path(path)
                        for path, thread_type in compiled.kernel_paths
                        if thread_type != "compute"
                        and "read" in Path(path).name
                    ]
                    write_paths = [
                        Path(path)
                        for path, thread_type in compiled.kernel_paths
                        if thread_type != "compute"
                        and "write" in Path(path).name
                    ]
                    if len(read_paths) != 1 or len(write_paths) != 1:
                        raise RuntimeError(
                            "concurrent grouped phase one expected one "
                            "reader and one writer"
                        )
                grid = compiled.core_ranges.bounding_box().grid_size()
                total_nodes = int(grid.x) * int(grid.y)
                active_nodes = (
                    total_nodes // 2
                    if concurrent_members
                    else total_nodes
                )
                replacements = {
                    "@JOBS_PER_NODE@": str(
                        -(-base_jobs // active_nodes)
                    ),
                    "@TOTAL_JOBS@": str(base_jobs),
                    "@K_TILES@": str(k_tiles),
                    "@N_TILES@": str(n_tiles),
                    "@REDUCTION_TILES_PER_PHASE@": str(
                        base_reduction_tiles_per_phase
                    ),
                    "@GRID_COLS@": str(int(grid.x)),
                    "@NUM_N_BLOCKS@": str(n_tiles // 2),
                    "@M_TILES_PER_NODE@": str(
                        -(-m_tiles // active_nodes)
                    ),
                    "@TOTAL_M_TILES@": str(m_tiles),
                    "@N_TILES_PER_NODE@": str(
                        -(-n_tiles // active_nodes)
                    ),
                    "@TOTAL_N_TILES@": str(n_tiles),
                    "@REDUCTION_TILES@": str(m_tiles),
                    "@SCALE_BITS@": str(scale_bits),
                    "@MEMBER_NODES@": str(active_nodes),
                }
                if concurrent_members:
                    read_source = read_paths[0].read_text(
                        encoding="utf-8"
                    )
                    write_source = write_paths[0].read_text(
                        encoding="utf-8"
                    )
                    read_paths[0].write_text(
                        _partition_generated_reader(
                            read_source, active_nodes
                        ),
                        encoding="utf-8",
                    )
                    write_paths[0].write_text(
                        _partition_generated_writer(
                            write_source, active_nodes
                        ),
                        encoding="utf-8",
                    )
                cpp_dir = Path(__file__).with_name("cpp")
                lora_template = (
                    cpp_dir / "lora_backward_arena_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                base_template = (
                    cpp_dir / "frozen_linear_backward_base_phase.cpp"
                ).read_text(encoding="utf-8")

                if concurrent_members:
                    lora_name = "lora_backward_phase"
                    lora_source = _shift_compile_time_cb_indices(
                        lora_template, 4
                    ).replace(
                        "void kernel_main()",
                        f"FORCE_INLINE void {lora_name}()",
                        1,
                    )
                    base_source = base_template.replace(
                        "lora_backward_phase()", f"{lora_name}()"
                    )
                    source = lora_source + "\n" + base_source
                    node_index_source = """const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();"""
                    partitioned_node_source = """const uint32_t global_node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();
    constexpr uint32_t member_nodes = @MEMBER_NODES@;
    const uint32_t node_index = global_node_index % member_nodes;"""
                    source = source.replace(
                        node_index_source,
                        partitioned_node_source,
                    )
                else:
                    member_sources = []
                    for member in range(2):
                        lora_name = f"lora_backward_phase_{member}"
                        member_name = f"grouped_member_phase_{member}"
                        lora_source = _shift_compile_time_cb_indices(
                            lora_template, 4
                        ).replace(
                            "void kernel_main()",
                            f"FORCE_INLINE void {lora_name}()",
                            1,
                        )
                        base_source = base_template.replace(
                            "lora_backward_phase()",
                            f"{lora_name}()",
                        ).replace(
                            "void kernel_main()",
                            f"FORCE_INLINE void {member_name}()",
                            1,
                        )
                        member_sources.append(
                            lora_source + "\n" + base_source
                        )
                    source = "\n".join(member_sources) + """

void kernel_main() {
    grouped_member_phase_0();
    grouped_member_phase_1();
}
"""
                for placeholder, value in replacements.items():
                    source = source.replace(placeholder, value)
                if any(key in source for key in replacements):
                    raise RuntimeError(
                        "unexpanded grouped phase-one placeholder"
                    )
                compute_paths[0].write_text(source, encoding="utf-8")
            break
        return result

    return grouped_phase_one
