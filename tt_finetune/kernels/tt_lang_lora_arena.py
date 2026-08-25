"""Single-output-arena exact LoRA forward for the installed Shardy stack."""

from __future__ import annotations

import hashlib
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_lora import (
    TILE_SIZE,
    _FP32_DTYPE_ONLY_TENSOR,
    plan_exact_lora_forward_blocks,
)


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=32)
def build_tt_lang_lora_forward_arena(
    *,
    scale: float,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
) -> Callable:
    """Return exact ``[output | H]`` in one TT-Lang DPS output tensor."""

    plan = plan_exact_lora_forward_blocks(
        tokens=tokens,
        input_features=input_features,
        output_features=output_features,
        physical_rank=physical_rank,
    )
    m_block = plan.m_tiles_per_block
    n_block = plan.n_tiles_per_block
    k_block = plan.k_tiles_per_block
    if (m_block, k_block, physical_rank) != (1, 2, TILE_SIZE):
        raise ValueError("forward arena currently requires exact M1/K2/R32")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The native LoRA arena requires the TT-XLA worker"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    scale_bf16 = float(
        torch.tensor(scale, dtype=torch.bfloat16).float().item()
    )
    scale_bits = struct.unpack(
        "<I", struct.pack("<f", scale_bf16)
    )[0]
    n_tiles = output_features // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    total_m_tiles = tokens // TILE_SIZE
    variant_payload = (
        f"{scale_bf16:.17g}:{tokens}:{input_features}:{output_features}:"
        f"{physical_rank}:{n_block}:forward-arena-v1"
    )
    variant = hashlib.sha256(
        variant_payload.encode("ascii")
    ).hexdigest()[:16]

    @ttl.operation(
        grid="full",
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _forward_arena_program(inputs, lora_a, lora_b, base, arena):
        m_tiles = inputs.shape[0] // TILE_SIZE
        n_blocks = n_tiles // n_block
        k_blocks = k_tiles // k_block
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        m_tiles_per_node = -(-m_tiles // total_nodes)

        inputs_dfb = ttl.make_dataflow_buffer_like(
            inputs,
            shape=(1, k_block),
            block_count=max(2, -(-n_block // k_block)),
        )
        a_dfb = ttl.make_dataflow_buffer_like(
            lora_a,
            shape=(rank_tiles, k_block),
            block_count=2,
        )
        hidden_partial_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, rank_tiles),
            block_count=2,
        )
        hidden_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, rank_tiles),
            block_count=2,
        )
        saved_hidden_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, rank_tiles),
            block_count=1,
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            lora_b,
            shape=(n_block, rank_tiles),
            block_count=2,
        )
        base_dfb = ttl.make_dataflow_buffer_like(
            base,
            shape=(1, n_block),
            block_count=2,
        )
        delta_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, n_block),
            block_count=2,
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, n_block),
            block_count=2,
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    for k_index in range(k_blocks):
                        k_start = k_index * k_block
                        with (
                            inputs_dfb.reserve() as x_block,
                            a_dfb.reserve() as a_block,
                        ):
                            tx_x = ttl.copy(
                                inputs[
                                    m_tile : m_tile + 1,
                                    k_start : k_start + k_block,
                                ],
                                x_block,
                            )
                            tx_a = ttl.copy(
                                lora_a[
                                    0:rank_tiles,
                                    k_start : k_start + k_block,
                                ],
                                a_block,
                            )
                            tx_x.wait()
                            tx_a.wait()
                    for n_index in range(n_blocks):
                        n_start = n_index * n_block
                        with (
                            b_dfb.reserve() as b_block,
                            base_dfb.reserve() as base_block,
                        ):
                            tx_b = ttl.copy(
                                lora_b[
                                    n_start : n_start + n_block,
                                    0:rank_tiles,
                                ],
                                b_block,
                            )
                            tx_base = ttl.copy(
                                base[
                                    m_tile : m_tile + 1,
                                    n_start : n_start + n_block,
                                ],
                                base_block,
                            )
                            tx_b.wait()
                            tx_base.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    with hidden_partial_dfb.reserve() as partial:
                        partial.store(
                            ttl.block.fill(0, shape=partial.shape)
                        )
                    for _ in range(k_blocks):
                        with (
                            inputs_dfb.wait() as x_block,
                            a_dfb.wait() as a_block,
                            hidden_partial_dfb.wait() as previous,
                        ):
                            with hidden_partial_dfb.reserve() as next_value:
                                next_value.store(
                                    previous
                                    + ttl.matmul(
                                        x_block,
                                        a_block,
                                        transpose_rhs=True,
                                    )
                                )
                    with hidden_partial_dfb.wait() as partial:
                        with (
                            hidden_dfb.reserve() as hidden,
                            saved_hidden_dfb.reserve() as saved,
                        ):
                            hidden.store(partial)
                            saved.store(partial)
                    with hidden_dfb.wait() as hidden:
                        for _ in range(n_blocks):
                            with (
                                b_dfb.wait() as b_block,
                                base_dfb.wait() as base_block,
                            ):
                                with delta_dfb.reserve() as delta:
                                    delta.store(
                                        ttl.matmul(
                                            hidden,
                                            b_block,
                                            transpose_rhs=True,
                                        )
                                    )
                                with (
                                    delta_dfb.wait() as delta,
                                    output_dfb.reserve() as output,
                                ):
                                    output.store(
                                        base_block + delta * scale_bf16
                                    )

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_m in range(m_tiles_per_node):
                m_tile = node_index * m_tiles_per_node + local_m
                if m_tile < m_tiles:
                    for n_index in range(n_blocks):
                        n_start = n_index * n_block
                        with output_dfb.wait() as output:
                            tx = ttl.copy(
                                output,
                                arena[
                                    m_tile : m_tile + 1,
                                    n_start : n_start + n_block,
                                ],
                            )
                            tx.wait()
                    with saved_hidden_dfb.wait() as saved:
                        tx_h = ttl.copy(
                            saved,
                            arena[
                                m_tile : m_tile + 1,
                                n_tiles : n_tiles + rank_tiles,
                            ],
                        )
                        tx_h.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.lora_forward_arena.{variant}",
        arg_roles=("in", "in", "in", "in", "out"),
        version_tag=f"lora-forward-arena-{variant}",
    )
    def forward_arena(inputs, lora_a, lora_b, base, arena):
        _clear_ttl_compile_wrapper_cache(_forward_arena_program)
        result = _forward_arena_program(
            inputs, lora_a, lora_b, base, arena
        )
        closure = _forward_arena_program.__closure__ or ()
        for name, cell in zip(
            _forward_arena_program.__code__.co_freevars,
            closure,
        ):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 9:
                    raise RuntimeError(
                        "exact forward arena expected 9 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                compiled.cb_configs[2].tensor = (
                    _FP32_DTYPE_ONLY_TENSOR
                )
                compute_paths = [
                    Path(path)
                    for path, thread_type in compiled.kernel_paths
                    if thread_type == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError(
                        "exact forward arena expected one compute kernel"
                    )
                grid = compiled.core_ranges.bounding_box().grid_size()
                total_nodes = int(grid.x) * int(grid.y)
                m_per_node = -(-total_m_tiles // total_nodes)
                source = (
                    Path(__file__).with_name("cpp")
                    / "lora_forward_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                original_cb_block = """constexpr uint32_t cb_b = get_compile_time_arg_val(4);
    constexpr uint32_t cb_base = get_compile_time_arg_val(5);
    constexpr uint32_t cb_delta = get_compile_time_arg_val(6);
    constexpr uint32_t cb_output = get_compile_time_arg_val(7);"""
                arena_cb_block = """constexpr uint32_t cb_saved_hidden = get_compile_time_arg_val(4);
    constexpr uint32_t cb_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_base = get_compile_time_arg_val(6);
    constexpr uint32_t cb_delta = get_compile_time_arg_val(7);
    constexpr uint32_t cb_output = get_compile_time_arg_val(8);"""
                if original_cb_block not in source:
                    raise RuntimeError(
                        "exact forward CB declaration block changed"
                    )
                source = source.replace(
                    original_cb_block, arena_cb_block, 1
                )
                source = source.replace(
                    "CircularBuffer b_cb(cb_b);",
                    "CircularBuffer saved_hidden_cb(cb_saved_hidden);\n"
                    "    CircularBuffer b_cb(cb_b);",
                )
                # Duplicate the final long-K destination tile while it is
                # still live in the destination registers.  Copying H after
                # its BF16 pack would require another compute startup and
                # changes the destination-mode state used by the following
                # projection on Blackhole.
                source = source.replace(
                    """            if (last) {
                hidden_cb.reserve_back(1);
            } else {""",
                    """            if (last) {
                hidden_cb.reserve_back(1);
                saved_hidden_cb.reserve_back(1);
            } else {""",
                    1,
                )
                source = source.replace(
                    """            pack_tile_block(0, target_cb, 1);
            tile_regs_release();
            if (last) {
                hidden_cb.push_back(1);""",
                    """            pack_tile_block(0, target_cb, 1);
            if (last) {
                PACK((pack_reconfig_data_format(cb_saved_hidden)));
                pack_tile_block(0, cb_saved_hidden, 1);
            }
            tile_regs_release();
            if (last) {
                hidden_cb.push_back(1);
                saved_hidden_cb.push_back(1);""",
                    1,
                )
                replacements = {
                    "@NUM_K_BLOCKS@": str(k_tiles // 2),
                    "@NUM_N_BLOCKS@": str(n_tiles // n_block),
                    "@N_TILES_PER_BLOCK@": str(n_block),
                    "@SECOND_SUBBLOCK_W@": str(min(4, n_block)),
                    "@M_BLOCKS_PER_NODE@": str(m_per_node),
                    "@TOTAL_M_BLOCKS@": str(total_m_tiles),
                    "@GRID_COLS@": str(int(grid.x)),
                    "@SCALE_BITS@": str(scale_bits),
                }
                for placeholder, value in replacements.items():
                    source = source.replace(placeholder, value)
                compute_paths[0].write_text(source, encoding="utf-8")
            break
        return result

    return forward_arena
