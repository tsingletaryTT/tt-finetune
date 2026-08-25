"""One-call exact resident backward for an MLP down projection.

The program owns frozen-base dProduct, LoRA dH/dB/dA, LoRA dProduct and the
final BF16 accumulation.  dH is multicast into worker-local L1 and never
appears as a PJRT output or a DRAM intermediate.
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
)
from .tt_lang_lora import TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR


ttl = None
tt_lang_operation = None
_BF16_TILE_BYTES = TILE_SIZE * TILE_SIZE * 2
_FP32_TILE_BYTES = TILE_SIZE * TILE_SIZE * 4


def resident_down_backward_output_tile_shape(
    *,
    tokens: int,
    input_features: int,
    output_features: int,
    activation_backward: bool = False,
) -> tuple[int, int]:
    """Arena layout for the selected final edge.

    The compatibility form is ``dProduct, dA, dB``. The fused-activation form
    replaces dProduct with ``dGate, dUp`` and keeps only final tensors visible
    outside the program.
    """

    return (
        (2 if activation_backward else 1) * (tokens // TILE_SIZE)
        + 1
        + output_features // TILE_SIZE,
        input_features // TILE_SIZE,
    )


def estimate_resident_down_backward_l1_bytes(
    *,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    base_reduction_tiles_per_phase: int = 1,
) -> int:
    dimensions = (
        tokens,
        input_features,
        output_features,
        physical_rank,
    )
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("resident down dimensions must be positive /32")
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    jobs_per_node = -(
        -(m_tiles * k_tiles) // BLACKHOLE_P150A_CORE_COUNT
    )
    da_k_tiles_per_node = -(
        -k_tiles // BLACKHOLE_P150A_CORE_COUNT
    )
    # BF16 CBs 0,1,3..6,8..12,14..19,21..24 plus FP32 CBs 2,7,13,20.
    bf16_tiles = (
        4 * base_reduction_tiles_per_phase
        + max(2, jobs_per_node)
        + 3 * n_tiles
        + max(2, jobs_per_node)
        + 4 * m_tiles
        + 2
        + m_tiles
        + jobs_per_node
        + 2 * m_tiles * da_k_tiles_per_node
        + 1
        + max(2, da_k_tiles_per_node)
        + 2 * max(2, jobs_per_node)
    )
    return bf16_tiles * _BF16_TILE_BYTES + 4 * _FP32_TILE_BYTES


def resident_down_backward_support_error(
    *,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    base_reduction_tiles_per_phase: int = 1,
) -> str | None:
    dimensions = (
        tokens,
        input_features,
        output_features,
        physical_rank,
    )
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        return "resident down dimensions must be positive /32"
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    if rank_tiles != 1 or m_tiles % 2 or n_tiles % 2:
        return "resident exact down path requires rank 32 and even M/N"
    if m_tiles > 8:
        return "resident down semaphore schedule currently supports M <= 8"
    if (
        base_reduction_tiles_per_phase <= 0
        or n_tiles % base_reduction_tiles_per_phase
    ):
        return "base reduction phase must divide N tiles"
    jobs_per_node = -(
        -(m_tiles * k_tiles) // BLACKHOLE_P150A_CORE_COUNT
    )
    da_k_tiles_per_node = -(
        -k_tiles // BLACKHOLE_P150A_CORE_COUNT
    )
    n_tiles_per_node = -(
        -n_tiles // BLACKHOLE_P150A_CORE_COUNT
    )
    if max(jobs_per_node, da_k_tiles_per_node, n_tiles_per_node) > 32:
        return "resident down schedule exceeds the 32-block CB limit"
    required = estimate_resident_down_backward_l1_bytes(
        tokens=tokens,
        input_features=input_features,
        output_features=output_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=base_reduction_tiles_per_phase,
    )
    budget = int(1.46 * 1024 * 1024 * 0.75)
    if required > budget:
        return (
            "resident down backward exceeds 75% worker L1: "
            f"{required} > {budget}"
        )
    return None


@lru_cache(maxsize=32)
def build_tt_lang_resident_down_backward(
    *,
    scale: float,
    tokens: int,
    input_features: int,
    output_features: int,
    physical_rank: int,
    base_reduction_tiles_per_phase: int = 1,
    activation_backward: bool = False,
    activation_diagnostic_stage: int = 6,
) -> Callable:
    error = resident_down_backward_support_error(
        tokens=tokens,
        input_features=input_features,
        output_features=output_features,
        physical_rank=physical_rank,
        base_reduction_tiles_per_phase=base_reduction_tiles_per_phase,
    )
    if error is not None:
        raise ValueError(error)
    if not 0 <= activation_diagnostic_stage <= 7:
        raise ValueError("activation diagnostic stage must be in [0, 7]")
    if activation_backward:
        raise RuntimeError(
            "single-compute-kernel down/activation fusion is numerically "
            "incompatible with the exact FP32-destination down program"
        )
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    jobs = m_tiles * k_tiles
    jobs_per_node = -(-jobs // BLACKHOLE_P150A_CORE_COUNT)
    da_k_tiles_per_node = -(
        -k_tiles // BLACKHOLE_P150A_CORE_COUNT
    )
    n_tiles_per_node = -(
        -n_tiles // BLACKHOLE_P150A_CORE_COUNT
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
                "resident down backward requires TT-XLA"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    scale_bf16 = float(
        torch.tensor(scale, dtype=torch.bfloat16).float().item()
    )
    scale_bits = struct.unpack("<I", struct.pack("<f", scale_bf16))[0]
    variant = hashlib.sha256(
        (
            f"{scale_bf16:.17g}:{tokens}:{input_features}:"
            f"{output_features}:{physical_rank}:"
            f"{base_reduction_tiles_per_phase}:resident-down-backward-v1"
        ).encode("ascii")
    ).hexdigest()[:16]

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _program(dy, weight, b, hidden, a, inputs, arena):
        base_dy = ttl.make_dataflow_buffer_like(
            dy,
            shape=(1, base_reduction_tiles_per_phase),
            block_count=2,
        )
        base_weight = ttl.make_dataflow_buffer_like(
            weight,
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
            # ABI probes; all generated bodies are replaced before launch.
            with base_dy.reserve() as block:
                tx = ttl.copy(dy[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(weight[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(b[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(hidden[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(a[0:1, 0:1], block)
                tx.wait()
            with base_dy.reserve() as block:
                tx = ttl.copy(inputs[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(6):
                with base_dy.wait() as value:
                    with final_dx.reserve() as output:
                        output.store(
                            ttl.block.fill(0, shape=output.shape)
                        )
            node_col, node_row = ttl.node(dims=2)
            impossible = node_row * 11 + node_col < 0
            if impossible:
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
            for _ in range(6):
                with final_dx.wait() as value:
                    tx = ttl.copy(value, arena[0:1, 0:1])
                    tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.resident_down_backward.{variant}",
        arg_roles=("in", "in", "in", "in", "in", "in", "out"),
        version_tag=f"resident-down-backward-{variant}",
    )
    def resident_backward(dy, weight, b, hidden, a, inputs, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(dy, weight, b, hidden, a, inputs, arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 25:
                    raise RuntimeError(
                        "resident down backward expected 25 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                for index in (2, 7, 13, 20):
                    compiled.cb_configs[index].tensor = (
                        _FP32_DTYPE_ONLY_TENSOR
                    )
                compiled.num_pipe_sync_semaphores = 2 * m_tiles + 1

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
                    or sorted(reader_indices[0]) != list(range(6))
                ):
                    raise RuntimeError(
                        f"resident down reader ABI changed: {reader_indices!r}"
                    )
                if writer_indices != [[6]]:
                    raise RuntimeError(
                        f"resident down writer ABI changed: {writer_indices!r}"
                    )

                replacements = {
                    "@M_TILES@": str(m_tiles),
                    "@K_TILES@": str(k_tiles),
                    "@N_TILES@": str(n_tiles),
                    "@BASE_REDUCTION_TILES@": str(
                        base_reduction_tiles_per_phase
                    ),
                    "@JOBS_PER_NODE@": str(jobs_per_node),
                    "@DA_K_TILES_PER_NODE@": str(da_k_tiles_per_node),
                    "@N_TILES_PER_NODE@": str(n_tiles_per_node),
                    "@TOTAL_JOBS@": str(jobs),
                    "@REDUCTION_TILES_PER_PHASE@": str(
                        base_reduction_tiles_per_phase
                    ),
                    "@GRID_COLS@": "11",
                    "@NUM_N_BLOCKS@": str(n_tiles // 2),
                    "@M_TILES_PER_NODE@": "1",
                    "@TOTAL_M_TILES@": str(m_tiles),
                    "@TOTAL_N_TILES@": str(n_tiles),
                    "@REDUCTION_TILES@": str(m_tiles),
                    "@SCALE_BITS@": str(scale_bits),
                }
                cpp_dir = Path(__file__).with_name("cpp")
                lora_source = _shift_compile_time_cb_indices(
                    (
                        cpp_dir / "lora_backward_arena_exact_compute.cpp"
                    ).read_text(encoding="utf-8"),
                    4,
                ).replace(
                    "void kernel_main()",
                    "FORCE_INLINE void lora_backward_phase()",
                    1,
                )
                base_source = (
                    cpp_dir / "frozen_linear_backward_base_phase.cpp"
                ).read_text(encoding="utf-8").replace(
                    "void kernel_main()",
                    "FORCE_INLINE void resident_down_phase_one()",
                    1,
                )
                phase_source = lora_source + "\n" + base_source

                helper_source = (
                    cpp_dir
                    / "grouped2_frozen_lora_backward_tail_exact_compute.cpp"
                ).read_text(encoding="utf-8").split(
                    "\nvoid kernel_main()", 1
                )[0]
                helper_source = helper_source.replace(
                    "            dh.wait_front(reduction_block_tiles);\n",
                    "            // Complete dH window is already resident.\n",
                )
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
                        "                    block * reduction_block_tiles + inner,\n"
                    ),
                )
                helper_source = helper_source.replace(
                    "            dh.pop_front(reduction_block_tiles);\n",
                    "            // Keep dH resident for LoRA dProduct.\n",
                )
                compute_source = (
                    phase_source
                    + "\n"
                    + helper_source
                    + "\n"
                    + (cpp_dir / "resident_down_backward_tail.cpp").read_text(
                        encoding="utf-8"
                    )
                )
                if activation_backward:
                    activation_source = (
                        cpp_dir / "swiglu_backward_exact_compute.cpp"
                    ).read_text(encoding="utf-8")
                    activation_source = activation_source.replace(
                        "constexpr uint32_t cb_dproduct = 0;",
                        "constexpr uint32_t cb_dproduct = 24;",
                    ).replace(
                        "constexpr uint32_t cb_gate = 1;",
                        "constexpr uint32_t cb_gate = 25;",
                    ).replace(
                        "constexpr uint32_t cb_up = 2;",
                        "constexpr uint32_t cb_up = 26;",
                    ).replace(
                        "constexpr uint32_t cb_pregradient = 3;",
                        "constexpr uint32_t cb_pregradient = 3;",
                    ).replace(
                        "constexpr uint32_t cb_sigmoid = 4;",
                        "constexpr uint32_t cb_sigmoid = 6;",
                    ).replace(
                        "constexpr uint32_t cb_one_minus_sigmoid = 5;",
                        "constexpr uint32_t cb_one_minus_sigmoid = 11;",
                    ).replace(
                        "constexpr uint32_t cb_gate_times_one_minus = 6;",
                        "constexpr uint32_t cb_gate_times_one_minus = 12;",
                    ).replace(
                        "constexpr uint32_t cb_derivative_inner = 7;",
                        "constexpr uint32_t cb_derivative_inner = 14;",
                    ).replace(
                        "constexpr uint32_t cb_derivative = 8;",
                        "constexpr uint32_t cb_derivative = 19;",
                    ).replace(
                        "constexpr uint32_t cb_silu = 9;",
                        "constexpr uint32_t cb_silu = 21;",
                    ).replace(
                        "constexpr uint32_t cb_dgate = 10;",
                        "constexpr uint32_t cb_dgate = 27;",
                    ).replace(
                        "constexpr uint32_t cb_dup = 11;",
                        "constexpr uint32_t cb_dup = 28;",
                    ).replace(
                        "constexpr uint32_t cb_one = 12;",
                        "constexpr uint32_t cb_one = 23;",
                    )
                    activation_source = activation_source.replace(
                        "void kernel_main()",
                        "FORCE_INLINE void resident_down_activation_phase()",
                        1,
                    ).replace(
                        "constexpr uint32_t diagnostic_stage = @DIAGNOSTIC_STAGE@;",
                        "constexpr uint32_t diagnostic_stage = "
                        f"{activation_diagnostic_stage};",
                    )
                    activation_source = activation_source.replace(
                        "        up.wait_front(1);\n",
                        "        up.wait_front(1);\n"
                        "        if constexpr (diagnostic_stage == 7) {\n"
                        "            copy_one(cb_dproduct, cb_dgate, true);\n"
                        "        }\n",
                        1,
                    )
                    compute_source = compute_source.replace(
                        "void kernel_main() {",
                        "FORCE_INLINE void resident_down_compute_phases() {",
                        1,
                    )
                    compute_source = (
                        compute_source
                        + "\n"
                        + activation_source
                        + "\nvoid kernel_main() {\n"
                        + "    resident_down_compute_phases();\n"
                        + "    disable_fp32_dest_acc();\n"
                        + "    resident_down_activation_phase();\n"
                        + "    enable_fp32_dest_acc();\n"
                        + "}\n"
                    )
                reader_source = (
                    cpp_dir / "resident_down_backward_reader.cpp"
                ).read_text(encoding="utf-8")
                if activation_backward:
                    reader_source = reader_source.replace(
                        "// Tensor order: dY, W, B, H, A, X.",
                        "// Tensor order: dY, W, B, H, A, X, gate, up.",
                    ).replace(
                        "    const uint32_t x_addr = get_common_arg_val<uint32_t>(5);",
                        "    const uint32_t x_addr = get_common_arg_val<uint32_t>(5);\n"
                        "    const uint32_t gate_addr = get_common_arg_val<uint32_t>(6);\n"
                        "    const uint32_t up_addr = get_common_arg_val<uint32_t>(7);",
                    ).replace(
                        "    // Frozen-base dProduct input stream.",
                        "    const auto gate = TensorAccessor(\n"
                        "        TensorAccessorArgs<\n"
                        "            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 25>(),\n"
                        "            6>(), gate_addr, kTileBytes);\n"
                        "    const auto up = TensorAccessor(\n"
                        "        TensorAccessorArgs<\n"
                        "            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<7, 25>(),\n"
                        "            7>(), up_addr, kTileBytes);\n\n"
                        "    // Frozen-base dProduct input stream.",
                    )
                    activation_reader = """

    // Dedicated depth-two checkpoint streams. Other activation temporaries
    // reuse CB storage only after their previous consumers have finished.
    CircularBuffer activation_gate(get_compile_time_arg_val(25));
    CircularBuffer activation_up(get_compile_time_arg_val(26));
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= total_jobs) continue;
        activation_gate.reserve_back(1);
        activation_up.reserve_back(1);
        noc0.async_read(
            gate, CoreLocalMem<uint32_t>(activation_gate.get_write_ptr()),
            kTileBytes, {.page_id = job}, {});
        noc0.async_read(
            up, CoreLocalMem<uint32_t>(activation_up.get_write_ptr()),
            kTileBytes, {.page_id = job}, {});
        noc0.async_read_barrier();
        activation_up.push_back(1);
        activation_gate.push_back(1);
    }
"""
                    close = reader_source.rfind("\n}")
                    reader_source = (
                        reader_source[:close]
                        + activation_reader
                        + reader_source[close:]
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
                        r"(\d+), 25>\(\),\s*(\d+)"
                    ),
                    lambda match: (
                        "get_tensor_accessor_args_cta_offset<"
                        f"{match.group(1)}, 25>(), "
                        f"{reader_common_position[int(match.group(1))]}"
                    ),
                    reader_source,
                )
                writer_name = (
                    "resident_down_activation_backward_writer.cpp"
                    if activation_backward
                    else "resident_down_backward_writer.cpp"
                )
                writer_source = (cpp_dir / writer_name).read_text(
                    encoding="utf-8"
                )
                sources = {
                    "compute": compute_source,
                    "read": reader_source,
                    "write": writer_source,
                }
                for kind, source in tuple(sources.items()):
                    for placeholder, value in replacements.items():
                        source = source.replace(placeholder, value)
                    unresolved = re.findall(r"@[A-Z0-9_]+@", source)
                    if unresolved:
                        raise RuntimeError(
                            f"unexpanded resident down {kind}: {unresolved}"
                        )
                    sources[kind] = source
                for path, thread_type in compiled.kernel_paths:
                    kind = (
                        "compute"
                        if thread_type == "compute"
                        else (
                            "read" if "read" in Path(path).name else "write"
                        )
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")
            break
        return result

    resident_backward.output_tile_shape = (
        resident_down_backward_output_tile_shape(
            tokens=tokens,
            input_features=input_features,
            output_features=output_features,
            activation_backward=activation_backward,
        )
    )
    return resident_backward
