"""One-call resident exact SwiGLU plus frozen down-projection LoRA.

This is the first forward MLP boundary that deliberately owns an intermediate
instead of returning it to PJRT.  SwiGLU product tiles are produced in BF16
destination mode, distributed through worker L1 over the NoC, and consumed by
the exact FP32-spilled down projection in the same TT-Metal program.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .model_workloads import EXACT_BASE_K_PHASES
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import (
    BLACKHOLE_P150A_CORE_COUNT,
    BLACKHOLE_P150A_GRID,
)
from .tt_lang_lora import BF16_TILE_BYTES, TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR


# Keep BF16 SwiGLU/SFPU work and FP32 down-reduction work on disjoint Tensix
# cores. This scheduling remains experimental; the safe path is selected by
# the higher-level capability gate.
RESIDENT_MLP_OUTPUT_CORE_COUNT = 72
RESIDENT_MLP_PRODUCT_CORE_COUNT = (
    BLACKHOLE_P150A_CORE_COUNT - RESIDENT_MLP_OUTPUT_CORE_COUNT
)


ttl = None
tt_lang_operation = None


def resident_swiglu_down_output_shape(
    *, tokens: int, hidden_features: int, physical_rank: int
) -> tuple[int, int]:
    return tokens, hidden_features + physical_rank


def estimate_resident_swiglu_down_l1_bytes(
    *,
    intermediate_features: int,
    product_jobs_per_core: int,
    output_jobs_per_core: int,
    rank_tiles: int = 1,
) -> int:
    """Conservative peak local CB allocation for the combined program."""

    values = (
        intermediate_features,
        product_jobs_per_core,
        output_jobs_per_core,
        rank_tiles,
    )
    if min(values) <= 0 or intermediate_features % TILE_SIZE:
        raise ValueError("resident SwiGLU/down dimensions must be positive /32")
    i_tiles = intermediate_features // TILE_SIZE
    # BF16-equivalent tiles. FP32 partial buffers cost two BF16 tiles each.
    tiles = (
        2  # gate stream
        + 2  # up stream
        + 2  # packed SiLU
        + product_jobs_per_core  # local products awaiting NoC distribution
        + output_jobs_per_core * i_tiles  # resident input rows
        + 4 * rank_tiles  # two K tiles of A, double buffered
        + 4 * rank_tiles  # FP32 H partial, depth two
        + 1  # H multicast source
        + output_jobs_per_core * rank_tiles  # received H per output job
        + rank_tiles  # saved H
        + 2  # W stream
        + 4  # FP32 base partial, depth two
        + max(2, output_jobs_per_core)  # staged base
        + 2 * rank_tiles  # B stream
        + 2  # delta
        + max(2, output_jobs_per_core)  # staged scaled delta
        + 2  # final output
        + 2  # decomposed-training sigmoid staging
    )
    return tiles * BF16_TILE_BYTES


def resident_swiglu_down_support_error(
    *,
    tokens: int,
    hidden_features: int,
    intermediate_features: int,
    physical_rank: int,
    base_k_tiles_per_phase: int | None = None,
    rank_k_tiles_per_phase: int = 2,
    allow_unvalidated_base_phase: bool = False,
) -> str | None:
    dimensions = (tokens, hidden_features, intermediate_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        return "all resident SwiGLU/down dimensions must be positive /32"
    rank_tiles = physical_rank // TILE_SIZE
    i_tiles = intermediate_features // TILE_SIZE
    if (
        rank_k_tiles_per_phase <= 0
        or i_tiles % rank_k_tiles_per_phase
    ):
        return "LoRA H reduction phase must divide intermediate tiles"
    registered = EXACT_BASE_K_PHASES.get(
        (intermediate_features, hidden_features)
    )
    if registered is None:
        return (
            "down projection shape has no silicon-validated exact reduction "
            f"contract: {(intermediate_features, hidden_features)}"
        )
    selected = registered if base_k_tiles_per_phase is None else base_k_tiles_per_phase
    if selected != registered and not allow_unvalidated_base_phase:
        return (
            "requested down reduction phase differs from the silicon-validated "
            f"contract: {selected} != {registered}"
        )
    if i_tiles % selected:
        return "down reduction phase must divide intermediate tiles"
    product_jobs = (tokens // TILE_SIZE) * i_tiles
    output_jobs = (tokens // TILE_SIZE) * (hidden_features // TILE_SIZE)
    product_per_core = -(
        -product_jobs // RESIDENT_MLP_PRODUCT_CORE_COUNT
    )
    output_per_core = -(
        -output_jobs // RESIDENT_MLP_OUTPUT_CORE_COUNT
    )
    required = estimate_resident_swiglu_down_l1_bytes(
        intermediate_features=intermediate_features,
        product_jobs_per_core=product_per_core,
        output_jobs_per_core=output_per_core,
        rank_tiles=rank_tiles,
    )
    budget = int(1.46 * 1024 * 1024 * 0.75)
    if required > budget:
        return f"resident SwiGLU/down exceeds 75% worker L1: {required} > {budget}"
    return None


@lru_cache(maxsize=16)
def build_tt_lang_resident_swiglu_down_forward(
    *,
    scale: float,
    tokens: int,
    hidden_features: int,
    intermediate_features: int,
    physical_rank: int,
    base_k_tiles_per_phase: int | None = None,
    rank_k_tiles_per_phase: int = 2,
    allow_unvalidated_base_phase: bool = False,
    fp32_activation: bool = False,
    decomposed_activation: bool = False,
    save_rank_hidden: bool = True,
    packed_gate_up_arena: bool = False,
    packed_gate_up_row_tiles: int | None = None,
) -> Callable:
    """Build the experimental one-call resident MLP forward boundary."""

    support_error = resident_swiglu_down_support_error(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        physical_rank=physical_rank,
        base_k_tiles_per_phase=base_k_tiles_per_phase,
        rank_k_tiles_per_phase=rank_k_tiles_per_phase,
        allow_unvalidated_base_phase=allow_unvalidated_base_phase,
    )
    if support_error is not None:
        raise ValueError(support_error)
    base_phase = (
        EXACT_BASE_K_PHASES[(intermediate_features, hidden_features)]
        if base_k_tiles_per_phase is None
        else int(base_k_tiles_per_phase)
    )
    m_tiles = tokens // TILE_SIZE
    h_tiles = hidden_features // TILE_SIZE
    i_tiles = intermediate_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    if packed_gate_up_arena:
        if (
            packed_gate_up_row_tiles is None
            or packed_gate_up_row_tiles < 2 * i_tiles
        ):
            raise ValueError(
                "packed gate/up arena row must contain both projections"
            )
        gate_page_id = (
            f"(job / {i_tiles}) * {packed_gate_up_row_tiles} "
            f"+ (job % {i_tiles})"
        )
        up_page_id = (
            f"(job / {i_tiles}) * {packed_gate_up_row_tiles} "
            f"+ {i_tiles} + (job % {i_tiles})"
        )
    else:
        if packed_gate_up_row_tiles is not None:
            raise ValueError(
                "packed_gate_up_row_tiles requires packed_gate_up_arena"
            )
        gate_page_id = "job"
        up_page_id = "job"
    product_jobs = m_tiles * i_tiles
    output_jobs = m_tiles * h_tiles
    product_per_core = -(
        -product_jobs // RESIDENT_MLP_PRODUCT_CORE_COUNT
    )
    output_per_core = -(
        -output_jobs // RESIDENT_MLP_OUTPUT_CORE_COUNT
    )
    scale_bf16 = float(torch.tensor(scale, dtype=torch.bfloat16).float().item())
    scale_bits = struct.unpack("<I", struct.pack("<f", scale_bf16))[0]

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("resident SwiGLU/down requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    options: dict[str, object] = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": True,
        "options": "--ttl-matmul-full-fp32",
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        options["math_approx_mode"] = False

    @ttl.operation(**options)
    def _program(gate, up, frozen_weight_t, lora_a, lora_b, arena):
        gate_cb = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), block_count=2)
        up_cb = ttl.make_dataflow_buffer_like(up, shape=(1, 1), block_count=2)
        silu_cb = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=2)
        product_out_cb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, product_per_core)
        )
        product_resident_cb = ttl.make_dataflow_buffer_like(
            gate, shape=(1, i_tiles), block_count=max(1, output_per_core)
        )
        a_cb = ttl.make_dataflow_buffer_like(
            lora_a,
            shape=(rank_tiles, rank_k_tiles_per_phase),
            block_count=2,
        )
        h_partial_cb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=2
        )
        h_mcast_cb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )
        h_resident_cb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=max(1, output_per_core)
        )
        h_saved_cb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, rank_tiles), block_count=1
        )
        w_cb = ttl.make_dataflow_buffer_like(
            frozen_weight_t, shape=(base_phase, 1), block_count=2
        )
        base_partial_cb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        base_cb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, output_per_core)
        )
        b_cb = ttl.make_dataflow_buffer_like(
            lora_b, shape=(1, rank_tiles), block_count=2
        )
        delta_cb = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=2)
        scaled_cb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, output_per_core)
        )
        output_cb = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=2)
        training_sigmoid_cb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )

        @ttl.datamovement()
        def read():
            # ABI probes only. All three generated kernels are replaced.
            with gate_cb.reserve() as block:
                tx = ttl.copy(gate[0:1, 0:1], block)
                tx.wait()
            with gate_cb.reserve() as block:
                tx = ttl.copy(up[0:1, 0:1], block)
                tx.wait()
            with gate_cb.reserve() as block:
                tx = ttl.copy(frozen_weight_t[0:1, 0:1], block)
                tx.wait()
            with gate_cb.reserve() as block:
                tx = ttl.copy(lora_a[0:1, 0:1], block)
                tx.wait()
            with gate_cb.reserve() as block:
                tx = ttl.copy(lora_b[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(5):
                with gate_cb.wait() as value:
                    with output_cb.reserve() as result:
                        result.store(value)
            node_col, node_row = ttl.node(dims=2)
            impossible = node_row * 11 + node_col < 0
            if impossible:
                with up_cb.wait() as _value:
                    pass
                with silu_cb.wait() as _value:
                    pass
                with product_out_cb.wait() as _value:
                    pass
                with product_resident_cb.wait() as _value:
                    pass
                with a_cb.wait() as _value:
                    pass
                with h_partial_cb.wait() as _value:
                    pass
                with h_mcast_cb.wait() as _value:
                    pass
                with h_resident_cb.wait() as _value:
                    pass
                with h_saved_cb.wait() as _value:
                    pass
                with w_cb.wait() as _value:
                    pass
                with base_partial_cb.wait() as _value:
                    pass
                with base_cb.wait() as _value:
                    pass
                with b_cb.wait() as _value:
                    pass
                with delta_cb.wait() as _value:
                    pass
                with scaled_cb.wait() as _value:
                    pass
                with training_sigmoid_cb.wait() as _value:
                    pass

        @ttl.datamovement()
        def write():
            for _ in range(5):
                with output_cb.wait() as value:
                    tx = ttl.copy(value, arena[0:1, 0:1])
                    tx.wait()

    variant = hashlib.sha256(
        (
            f"{scale_bf16:.17g}:{tokens}:{hidden_features}:"
            f"{intermediate_features}:{physical_rank}:{base_phase}:"
            f"rank_phase={rank_k_tiles_per_phase}:"
            f"fp32_activation={bool(fp32_activation)}:"
            f"decomposed_activation={bool(decomposed_activation)}:"
            f"save_rank_hidden={bool(save_rank_hidden)}:"
            f"packed_gate_up={bool(packed_gate_up_arena)}:"
            f"packed_row={packed_gate_up_row_tiles}:"
            "resident-swiglu-down-forward-v11-restored-staged-exact"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.resident_swiglu_down.{variant}",
        arg_roles=("in", "in", "in", "in", "in", "out"),
        version_tag=f"resident-swiglu-down-{variant}",
    )
    def resident_forward(gate, up, frozen_weight_t, lora_a, lora_b, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(gate, up, frozen_weight_t, lora_a, lora_b, arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 18:
                    raise RuntimeError(
                        "resident SwiGLU/down expected 18 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                compiled.cb_configs[6].tensor = _FP32_DTYPE_ONLY_TENSOR
                compiled.cb_configs[11].tensor = _FP32_DTYPE_ONLY_TENSOR
                compiled.num_pipe_sync_semaphores = 2

                reader_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, kind) in enumerate(compiled.kernel_paths)
                    if kind != "compute" and "read" in Path(path).name
                ]
                writer_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, kind) in enumerate(compiled.kernel_paths)
                    if kind != "compute" and "write" in Path(path).name
                ]
                if len(reader_indices) != 1 or sorted(reader_indices[0]) != list(range(5)):
                    raise RuntimeError(f"resident MLP reader ABI changed: {reader_indices!r}")
                if writer_indices != [[5]]:
                    raise RuntimeError(f"resident MLP writer ABI changed: {writer_indices!r}")

                replacements = {
                    "@M_TILES@": str(m_tiles),
                    "@H_TILES@": str(h_tiles),
                    "@I_TILES@": str(i_tiles),
                    "@RANK_TILES@": str(rank_tiles),
                    "@PRODUCT_JOBS@": str(product_jobs),
                    "@PRODUCT_JOBS_PER_CORE@": str(product_per_core),
                    "@OUTPUT_JOBS@": str(output_jobs),
                    "@OUTPUT_JOBS_PER_CORE@": str(output_per_core),
                    "@BASE_K_BLOCK_TILES@": str(base_phase),
                    "@NUM_BASE_K_BLOCKS@": str(i_tiles // base_phase),
                    "@H_K_BLOCK_TILES@": str(rank_k_tiles_per_phase),
                    "@NUM_H_K_BLOCKS@": str(
                        i_tiles // rank_k_tiles_per_phase
                    ),
                    "@SCALE_BITS@": str(scale_bits),
                    "@FP32_ACTIVATION@": (
                        "true" if fp32_activation else "false"
                    ),
                    "@DECOMPOSED_ACTIVATION@": (
                        "true" if decomposed_activation else "false"
                    ),
                    "@GATE_PAGE_ID@": gate_page_id,
                    "@UP_PAGE_ID@": up_page_id,
                }
                cpp_dir = Path(__file__).with_name("cpp")
                sources = {
                    "compute": (cpp_dir / "resident_swiglu_down_compute.cpp").read_text(),
                    "read": (cpp_dir / "resident_swiglu_down_reader.cpp").read_text(),
                    "write": (cpp_dir / "resident_swiglu_down_writer.cpp").read_text(),
                }
                if not save_rank_hidden:
                    saved_write = """        if (h == 0) {
            h_saved.wait_front(rank_tiles);
            noc1.async_write(
                CoreLocalMem<uint32_t>(h_saved.get_read_ptr()), arena,
                rank_tiles * kTileBytes, {},
                {.page_id = m * arena_cols + h_tiles});
            noc1.async_write_barrier();
            h_saved.pop_front(rank_tiles);
        }"""
                    discard = """        if (h == 0) {
            // The full backward owner recomputes H in L1. Drain the
            // producer CB without exporting a rank-space activation.
            h_saved.wait_front(rank_tiles);
            h_saved.pop_front(rank_tiles);
        }"""
                    if sources["write"].count(saved_write) != 1:
                        raise RuntimeError(
                            "resident MLP saved-H writer marker changed"
                        )
                    sources["write"] = sources["write"].replace(
                        saved_write, discard, 1
                    )
                reader_position = {
                    tensor_index: position
                    for position, tensor_index in enumerate(reader_indices[0])
                }
                sources["read"] = re.sub(
                    r"get_common_arg_val<uint32_t>\((\d+)\)",
                    lambda match: (
                        "get_common_arg_val<uint32_t>("
                        f"{reader_position[int(match.group(1))]})"
                    ),
                    sources["read"],
                )
                sources["read"] = re.sub(
                    r"get_tensor_accessor_args_cta_offset<(\d+), 18>\(\),\s*(\d+)",
                    lambda match: (
                        "get_tensor_accessor_args_cta_offset<"
                        f"{match.group(1)}, 18>(), "
                        f"{reader_position[int(match.group(1))]}"
                    ),
                    sources["read"],
                )
                for key, source in tuple(sources.items()):
                    for placeholder, value in replacements.items():
                        source = source.replace(placeholder, value)
                    unresolved = [p for p in replacements if p in source]
                    if unresolved:
                        raise RuntimeError(f"unexpanded resident MLP {key}: {unresolved}")
                    sources[key] = source
                for path, kind in compiled.kernel_paths:
                    source_kind = "compute" if kind == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[source_kind], encoding="utf-8")

            break
        return result

    resident_forward.output_shape = resident_swiglu_down_output_shape(
        tokens=tokens,
        hidden_features=hidden_features,
        # Preserve the accepted [output | rank-space] boundary shape even
        # when H itself is drained instead of written.  Returning a slice from
        # this padded arena is numerically observable in downstream backward.
        physical_rank=physical_rank,
    )
    # Permit exact full-layer owners to serialize this physical program as a
    # compiler-private child while retaining the standalone public ABI.
    resident_forward._ttl_program = _program
    return resident_forward
