"""Exact grouped frozen-linear LoRA forward from four logical PEFT tensors."""

from __future__ import annotations

import hashlib
import inspect
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
    _exact_biasless_add_source,
)
from .tt_lang_lora import TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR


ttl = None
tt_lang_operation = None


def _replace_compute_fragment_once(
    source: str,
    *,
    pattern: str,
    replacement: str | Callable[[re.Match[str]], str],
    description: str,
) -> str:
    """Replace one generated-compute fragment without depending on spacing.

    TT-Lang owns the formatting of the generated C++ source.  Matching a
    complete emitted line with ``str.replace`` made the resident MLP path
    silently depend on the particular printer version in the active PJRT
    runtime.  Keep the rewrite strict about structure (exactly one match),
    while allowing inconsequential whitespace changes.
    """

    rewritten, count = re.subn(
        pattern,
        replacement,
        source,
        count=2,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(
            "grouped input checkpoint expected exactly one "
            f"{description}, found {count}"
        )
    return rewritten


def _inject_grouped_input_checkpoint_compute(source: str) -> str:
    """Add the input-checkpoint copy to a generated grouped compute kernel."""

    source = _replace_compute_fragment_once(
        source,
        pattern=(
            r"^(?P<indent>[ \t]*)constexpr\s+uint32_t\s+cb_output\s*=\s*"
            r"get_compile_time_arg_val\(\s*12\s*\)\s*;[ \t]*$"
        ),
        replacement=(
            r"\g<indent>constexpr uint32_t cb_output = "
            r"get_compile_time_arg_val(12);\n"
            r"\g<indent>constexpr uint32_t cb_checkpoint_input = "
            r"get_compile_time_arg_val(13);\n"
            r"\g<indent>constexpr uint32_t cb_checkpoint_output = "
            r"get_compile_time_arg_val(14);"
        ),
        description="cb_output declaration",
    )
    source = _replace_compute_fragment_once(
        source,
        pattern=(
            r"^(?P<indent>[ \t]*)CircularBuffer\s+output\s*\(\s*"
            r"cb_output\s*\)\s*;[ \t]*$"
        ),
        replacement=(
            r"\g<indent>CircularBuffer output(cb_output);\n"
            r"\g<indent>CircularBuffer checkpoint_input("
            r"cb_checkpoint_input);\n"
            r"\g<indent>CircularBuffer checkpoint_output("
            r"cb_checkpoint_output);"
        ),
        description="output circular-buffer declaration",
    )

    def inject_after_output(match: re.Match[str]) -> str:
        indent = match.group("indent")
        body_indent = indent + "    "
        deep_indent = body_indent + "    "
        return (
            f"{indent}output.push_back(n_tiles_per_block);\n"
            f"{indent}if ((job % num_n_blocks) == 0) {{\n"
            f"{body_indent}constexpr uint32_t checkpoint_tiles = "
            "num_k_blocks * k_block_tiles;\n"
            f"{body_indent}for (uint32_t tile = 0; "
            "tile < checkpoint_tiles; ++tile) {\n"
            f"{deep_indent}checkpoint_input.wait_front(1);\n"
            f"{deep_indent}checkpoint_output.reserve_back(1);\n"
            f"{deep_indent}tile_regs_acquire();\n"
            f"{deep_indent}copy_tile_init(cb_checkpoint_input);\n"
            f"{deep_indent}copy_tile(cb_checkpoint_input, 0, 0);\n"
            f"{deep_indent}tile_regs_commit_bf16();\n"
            f"{deep_indent}tile_regs_wait();\n"
            f"{deep_indent}pack_reconfig_data_format_bf16("
            "cb_checkpoint_output);\n"
            f"{deep_indent}pack_tile_bf16("
            "0, cb_checkpoint_output, 0);\n"
            f"{deep_indent}tile_regs_release_bf16();\n"
            f"{deep_indent}checkpoint_output.push_back(1);\n"
            f"{deep_indent}checkpoint_input.pop_front(1);\n"
            f"{body_indent}}}\n"
            f"{indent}}}\n"
            f"{indent}add_tiles_init(cb_base, cb_scaled_delta);\n"
            f"{indent}scaled_delta.pop_front"
        )

    source = _replace_compute_fragment_once(
        source,
        pattern=(
            r"^(?P<indent>[ \t]*)output\.push_back\(\s*"
            r"n_tiles_per_block\s*\)\s*;[ \t]*\n"
            r"(?P=indent)scaled_delta\.pop_front"
        ),
        replacement=inject_after_output,
        description="output-to-scaled-delta boundary",
    )
    return source


@lru_cache(maxsize=32)
def build_tt_lang_grouped2_logical_forward(
    *,
    scale: float,
    tokens: int,
    input_features: int,
    member_output_features: int,
    base_k_tiles_per_phase: int,
    physical_rank: int = TILE_SIZE,
    save_input_checkpoint: bool = False,
    precomputed_rank_delta: bool = False,
) -> Callable:
    dimensions = (
        tokens, input_features, member_output_features, physical_rank
    )
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("logical grouped dimensions must be positive /32")
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    member_n_tiles = member_output_features // TILE_SIZE
    total_n_tiles = 2 * member_n_tiles
    rank_tiles = physical_rank // TILE_SIZE
    if k_tiles % 2:
        raise ValueError("exact H reduction requires an even K tile count")
    if (
        base_k_tiles_per_phase <= 0
        or k_tiles % base_k_tiles_per_phase
    ):
        raise ValueError("invalid exact base K phase")
    k_block = 2
    k_blocks = k_tiles // k_block
    base_k_blocks = k_tiles // base_k_tiles_per_phase
    total_jobs = m_tiles * total_n_tiles
    jobs_per_node = -(-total_jobs // BLACKHOLE_P150A_CORE_COUNT)
    if jobs_per_node > 32:
        raise ValueError("logical grouped staging exceeds 32 CB blocks")
    checkpoint_stage = 1 if save_input_checkpoint else 0
    precomputed_stage = 1 if precomputed_rank_delta else 0

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "logical grouped forward requires TT-XLA"
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
            f"{member_output_features}:{base_k_tiles_per_phase}:"
            f"physical_rank={physical_rank}:"
            f"checkpoint={save_input_checkpoint}:"
            f"precomputed={precomputed_rank_delta}:"
            "grouped2-logical-forward-v10-oracle-h-pack"
        ).encode("ascii")
    ).hexdigest()[:16]

    options = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": True,
        "options": "--ttl-matmul-full-fp32",
    }
    if "math_approx_mode" in inspect.signature(
        ttl.operation
    ).parameters:
        options["math_approx_mode"] = False

    @ttl.operation(**options)
    def _program(inputs, frozen_weight, a0, b0, a1, b1, arena):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        local_jobs = -(-total_jobs // total_nodes)

        h_x_dfb = ttl.make_dataflow_buffer_like(
            inputs, shape=(1, k_block), block_count=2
        )
        a_dfb = ttl.make_dataflow_buffer_like(
            a0, shape=(rank_tiles, k_block), block_count=2
        )
        h_acc_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=2
        )
        h_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=2
        )
        saved_h_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )
        base_x_dfb = ttl.make_dataflow_buffer_like(
            inputs,
            shape=(1, base_k_tiles_per_phase),
            block_count=2,
        )
        w_dfb = ttl.make_dataflow_buffer_like(
            frozen_weight,
            shape=(base_k_tiles_per_phase, 1),
            block_count=2,
        )
        base_acc_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        base_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, 1),
            block_count=max(2, local_jobs),
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b0, shape=(1, rank_tiles), block_count=2,
        )
        delta_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        scaled_delta_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, 1),
            block_count=max(2, local_jobs),
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        # TT-Lang type-checks the body of a compile-time-disabled Python
        # branch while resolving its nested threads.  Define correctly shaped
        # checkpoint buffers unconditionally so those captures are never
        # empty and the dormant copies still type-check.  The branch emits no
        # transactions when checkpointing is disabled; placing these two CBs
        # last also preserves every existing compute CB index.
        input_checkpoint_dfb = ttl.make_dataflow_buffer_like(
            inputs,
            shape=(1, 1),
            block_count=k_tiles if checkpoint_stage == 1 else 2,
        )
        output_checkpoint_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        precomputed_delta_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        # A DFB has exactly one producer.  The ordinary path produces the
        # saved rank checkpoint from the compute thread, while the high-rank
        # exact path copies a TTNN-precomputed checkpoint in the reader.  Do
        # not alias those lifecycles even though the Python branches are
        # compile-time constant: TT-Lang validates producer ownership before
        # it folds the branch.
        precomputed_saved_h_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=2
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    m_tile = job // total_n_tiles
                    n_index = job - m_tile * total_n_tiles
                    member = n_index // member_n_tiles
                    member_n = n_index - member * member_n_tiles
                    # Even when the wide delta is precomputed by TTNN, the
                    # accepted training ABI still needs the exact grouped2 H
                    # checkpoint. Recompute that narrow value with the
                    # oracle's reduction schedule; it never crosses PJRT and
                    # is emitted only for member_n == 0.
                    for k_index in range(k_blocks):
                        k_start = k_index * k_block
                        with (
                            h_x_dfb.reserve() as x_block,
                            a_dfb.reserve() as a_block,
                        ):
                            tx_x = ttl.copy(
                                inputs[
                                    m_tile : m_tile + 1,
                                    k_start : k_start + k_block,
                                ],
                                x_block,
                            )
                            if member == 0:
                                tx_a = ttl.copy(
                                    a0[
                                        0:rank_tiles,
                                        k_start : k_start + k_block,
                                    ],
                                    a_block,
                                )
                                tx_a.wait()
                            if member == 1:
                                tx_a = ttl.copy(
                                    a1[
                                        0:rank_tiles,
                                        k_start : k_start + k_block,
                                    ],
                                    a_block,
                                )
                                tx_a.wait()
                            tx_x.wait()
                    for k_index in range(base_k_blocks):
                        k_start = (
                            k_index * base_k_tiles_per_phase
                        )
                        with (
                            base_x_dfb.reserve() as x_block,
                            w_dfb.reserve() as w_block,
                        ):
                            tx_x = ttl.copy(
                                inputs[
                                    m_tile : m_tile + 1,
                                    k_start : k_start
                                    + base_k_tiles_per_phase,
                                ],
                                x_block,
                            )
                            tx_w = ttl.copy(
                                frozen_weight[
                                    k_start : k_start
                                    + base_k_tiles_per_phase,
                                    n_index : n_index + 1,
                                ],
                                w_block,
                            )
                            tx_w.wait()
                            tx_x.wait()
                    if precomputed_stage == 1:
                        with precomputed_delta_dfb.reserve() as b_block:
                            if member == 0:
                                tx_b = ttl.copy(
                                    b0[
                                        m_tile : m_tile + 1,
                                        member_n : member_n + 1,
                                    ],
                                    b_block,
                                )
                                tx_b.wait()
                            if member == 1:
                                tx_b = ttl.copy(
                                    b1[
                                        m_tile : m_tile + 1,
                                        member_n : member_n + 1,
                                    ],
                                    b_block,
                                )
                                tx_b.wait()
                    if precomputed_stage == 0:
                        with b_dfb.reserve() as b_block:
                            if member == 0:
                                tx_b = ttl.copy(
                                    b0[
                                        member_n : member_n + 1, 0:rank_tiles
                                    ],
                                    b_block,
                                )
                                tx_b.wait()
                            if member == 1:
                                tx_b = ttl.copy(
                                    b1[
                                        member_n : member_n + 1, 0:rank_tiles
                                    ],
                                    b_block,
                                )
                                tx_b.wait()
                    # One output job per token row owns the input checkpoint.
                    # This copy is last so shallow-CB backpressure cannot
                    # prevent the compute operands from being staged.
                    if checkpoint_stage == 1:
                        if n_index == 0:
                            for checkpoint_k in range(k_tiles):
                                with input_checkpoint_dfb.reserve() as block:
                                    transaction = ttl.copy(
                                        inputs[
                                            m_tile : m_tile + 1,
                                            checkpoint_k : checkpoint_k + 1,
                                        ],
                                        block,
                                    )
                                    transaction.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    m_tile = job // total_n_tiles
                    n_index = job - m_tile * total_n_tiles
                    member_n = n_index % member_n_tiles
                    if precomputed_stage != 2:
                        with h_acc_dfb.reserve() as partial:
                            partial.store(
                                ttl.block.fill(0, shape=partial.shape)
                            )
                        for _ in range(k_blocks):
                            with (
                                h_x_dfb.wait() as x_block,
                                a_dfb.wait() as a_block,
                                h_acc_dfb.wait() as previous,
                            ):
                                with h_acc_dfb.reserve() as next_value:
                                    next_value.store(
                                        previous
                                        + ttl.matmul(
                                            x_block,
                                            a_block,
                                            transpose_rhs=True,
                                        )
                                    )
                        with h_acc_dfb.wait() as partial:
                            with h_dfb.reserve() as hidden:
                                hidden.store(partial)
                            if member_n == 0:
                                with saved_h_dfb.reserve() as saved:
                                    saved.store(partial)

                    with base_acc_dfb.reserve() as partial:
                        partial.store(
                            ttl.block.fill(0, shape=partial.shape)
                        )
                    for _ in range(base_k_blocks):
                        with (
                            base_x_dfb.wait() as x_block,
                            w_dfb.wait() as w_block,
                            base_acc_dfb.wait() as previous,
                        ):
                            with base_acc_dfb.reserve() as next_value:
                                next_value.store(
                                    previous + ttl.matmul(x_block, w_block)
                                )
                    with base_acc_dfb.wait() as partial:
                        with base_dfb.reserve() as base:
                            base.store(partial)
                    if precomputed_stage == 0:
                        with (
                            h_dfb.wait() as hidden,
                            b_dfb.wait() as b_block,
                        ):
                            with delta_dfb.reserve() as delta:
                                delta.store(
                                    ttl.matmul(
                                        hidden,
                                        b_block,
                                        transpose_rhs=True,
                                    )
                                )
                    if precomputed_stage == 1:
                        with precomputed_delta_dfb.wait() as delta:
                            with scaled_delta_dfb.reserve() as scaled:
                                scaled.store(delta * scale_bf16)
                    else:
                        with delta_dfb.wait() as delta:
                            with scaled_delta_dfb.reserve() as scaled:
                                scaled.store(delta * scale_bf16)
                    with (
                        base_dfb.wait() as base,
                        scaled_delta_dfb.wait() as scaled,
                    ):
                        with output_dfb.reserve() as output:
                            output.store(base + scaled)
                    if checkpoint_stage == 1:
                        if n_index == 0:
                            for _ in range(k_tiles):
                                with (
                                    input_checkpoint_dfb.wait() as source,
                                    output_checkpoint_dfb.reserve() as target,
                                ):
                                    target.store(source)

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    m_tile = job // total_n_tiles
                    n_index = job - m_tile * total_n_tiles
                    member = n_index // member_n_tiles
                    member_n = n_index - member * member_n_tiles
                    with output_dfb.wait() as output:
                        tx = ttl.copy(
                            output,
                            arena[
                                m_tile : m_tile + 1,
                                n_index : n_index + 1,
                            ],
                        )
                        tx.wait()
                    if member_n == 0:
                        if precomputed_stage == 1:
                            with saved_h_dfb.wait() as saved:
                                tx_h = ttl.copy(
                                    saved,
                                    arena[
                                        m_tile : m_tile + 1,
                                        total_n_tiles + member * rank_tiles :
                                        total_n_tiles + (member + 1) * rank_tiles,
                                    ],
                                )
                                tx_h.wait()
                        else:
                            with saved_h_dfb.wait() as saved:
                                tx_h = ttl.copy(
                                    saved,
                                    arena[
                                        m_tile : m_tile + 1,
                                        total_n_tiles + member * rank_tiles :
                                        total_n_tiles + (member + 1) * rank_tiles,
                                    ],
                                )
                                tx_h.wait()
                    if checkpoint_stage == 1:
                        if n_index == 0:
                            for checkpoint_k in range(k_tiles):
                                with output_checkpoint_dfb.wait() as block:
                                    transaction = ttl.copy(
                                        block,
                                        arena[
                                            m_tile : m_tile + 1,
                                            total_n_tiles + 2 + checkpoint_k :
                                            total_n_tiles + 3 + checkpoint_k,
                                        ],
                                    )
                                    transaction.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.grouped2_logical_forward.{variant}",
        arg_roles=("in", "in", "in", "in", "in", "in", "out"),
        version_tag=f"grouped2-logical-forward-{variant}",
    )
    def logical_forward(inputs, frozen_weight, a0, b0, a1, b1, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(
            inputs, frozen_weight, a0, b0, a1, b1, arena
        )
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                expected_cb_count = 16
                if len(compiled.cb_configs) != expected_cb_count:
                    raise RuntimeError(
                        "logical grouped forward expected "
                        f"{expected_cb_count} CBs"
                    )
                compiled.cb_configs[2].tensor = _FP32_DTYPE_ONLY_TENSOR
                compiled.cb_configs[7].tensor = _FP32_DTYPE_ONLY_TENSOR
                compute_paths = [
                    Path(path)
                    for path, thread_type in compiled.kernel_paths
                    if thread_type == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError(
                        "logical grouped forward expected one compute kernel"
                    )
                grid = compiled.core_ranges.bounding_box().grid_size()
                replacements = {
                    "@NUM_K_BLOCKS@": str(k_blocks),
                    "@H_K_BLOCK_TILES@": "2",
                    "@RANK_TILES@": str(rank_tiles),
                    "@DELTA_RANK_TILES@": str(rank_tiles),
                    "@DELTA_H_TILE_OFFSET@": "0",
                    "@BASE_K_BLOCK_TILES@": str(
                        base_k_tiles_per_phase
                    ),
                    "@NUM_BASE_K_BLOCKS@": str(base_k_blocks),
                    "@BASE_TRANSPOSE_RHS@": "false",
                    "@NUM_N_BLOCKS@": str(total_n_tiles),
                    "@N_TILES_PER_BLOCK@": "1",
                    "@JOBS_PER_NODE@": str(
                        -(
                            -total_jobs
                            // (int(grid.x) * int(grid.y))
                        )
                    ),
                    "@TOTAL_JOBS@": str(total_jobs),
                    "@GRID_COLS@": str(int(grid.x)),
                    "@SCALE_BITS@": str(scale_bits),
                    # The grouped MLP path predates the diagnostic O-rank
                    # reduction variants.  It must keep the accepted legacy
                    # initialization schedule, but it still consumes the
                    # shared projection compute template and therefore must
                    # expand the placeholder explicitly.
                    "@LORA_INIT_SCHEDULE@": "0",
                    "@BIAS_DECLARATIONS@": "",
                    "@BIAS_BUFFER_OBJECTS@": "",
                    "@BASE_BIAS_WAIT@": "",
                    "@BASE_BIAS_ACCUMULATE@": "",
                    "@BASE_BIAS_POP@": "",
                    "@BASE_FINAL_TARGET@": (
                        "(last ? cb_base : cb_base_partial)"
                    ),
                    "@BASE_RESERVE@": (
                        "if (last) { base.reserve_back("
                        "n_tiles_per_block); } else { "
                        "base_partial.reserve_back("
                        "n_tiles_per_block); }"
                    ),
                    "@BASE_PUSH@": (
                        "if (last) { base.push_back("
                        "n_tiles_per_block); } else { "
                        "base_partial.push_back("
                        "n_tiles_per_block); }"
                    ),
                    "@BASE_BIAS_FINISH@": "",
                    "@FINAL_ADD_BODY@": _exact_biasless_add_source(),
                }
                source = (
                    Path(__file__).with_name("cpp")
                    / "frozen_linear_lora_forward_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                source = source.replace(
                    "constexpr uint32_t n_tiles_per_block = "
                    "@N_TILES_PER_BLOCK@;",
                    "constexpr uint32_t n_tiles_per_block = "
                    "@N_TILES_PER_BLOCK@;\n"
                    "    constexpr uint32_t member_n_tiles = "
                    f"{member_n_tiles};",
                )
                source = source.replace(
                    "const uint32_t n_index = job % num_n_blocks;",
                    "const uint32_t n_index = job % num_n_blocks;\n"
                    "        const uint32_t member_n_index = "
                    "n_index % member_n_tiles;",
                )
                source = source.replace(
                    "if (n_index == 0)", "if (member_n_index == 0)"
                )
                source = source.replace(
                    "last && n_index == 0",
                    "last && member_n_index == 0",
                )
                for placeholder, value in replacements.items():
                    source = source.replace(placeholder, value)
                if precomputed_rank_delta:
                    source = source.replace(
                        "constexpr uint32_t cb_output = "
                        "get_compile_time_arg_val(12);",
                        "constexpr uint32_t cb_output = "
                        "get_compile_time_arg_val(12);\n"
                        "    constexpr uint32_t cb_precomputed_delta = "
                        "get_compile_time_arg_val(15);",
                        1,
                    )
                    source = source.replace(
                        "CircularBuffer output(cb_output);",
                        "CircularBuffer output(cb_output);\n"
                        "    CircularBuffer precomputed_delta("
                        "cb_precomputed_delta);",
                        1,
                    )
                    # Preserve the oracle's two-pack H finalization exactly:
                    # its second pack into saved_h is numerically observable
                    # for multi-tile ranks.  The normal delta consumer is
                    # replaced below, so retire the ordinary H queue locally
                    # only after both packs and pushes have completed.
                    h_push_marker = '''            if (last) {
                h.push_back(rank_tiles);
                if (member_n_index == 0) {
                    saved_h.push_back(rank_tiles);
                }
            } else {'''
                    if source.count(h_push_marker) != 1:
                        raise RuntimeError(
                            "logical grouped H finalization changed"
                        )
                    source = source.replace(
                        h_push_marker,
                        h_push_marker.replace(
                            "            } else {",
                            "                h.pop_front(rank_tiles);\n"
                            "            } else {",
                        ),
                        1,
                    )
                    delta_start = source.index("        // delta = H @ B.T.")
                    scale_start = source.index(
                        "        // Preserve BF16(delta * scale)."
                    )
                    scale_tail = source[scale_start:]
                    scale_tail = re.sub(
                        r"\bcb_delta\b", "cb_precomputed_delta", scale_tail
                    )
                    scale_tail = re.sub(
                        r"\bdelta\.", "precomputed_delta.", scale_tail
                    )
                    source = source[:delta_start] + scale_tail
                # The checkpoint tail is inserted into FINAL_ADD_BODY.  It
                # cannot be located until all template placeholders have been
                # expanded; doing this earlier made the combined resident MLP
                # owner fail before TT-Metal compilation even though the
                # generated arithmetic was valid.
                if save_input_checkpoint:
                    source = _inject_grouped_input_checkpoint_compute(source)
                if any(key in source for key in replacements):
                    raise RuntimeError(
                        "unexpanded logical grouped placeholder"
                    )
                compute_paths[0].write_text(source, encoding="utf-8")
            break
        return result

    # Expose the private TT-Lang entry so larger artifact-v4 owners can embed
    # this already-qualified physical program without a PJRT boundary.
    logical_forward._ttl_program = _program
    return logical_forward
