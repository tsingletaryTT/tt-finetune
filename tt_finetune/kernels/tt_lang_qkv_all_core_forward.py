"""All-110-worker exact Q/K/V projection phase for resident attention."""

from __future__ import annotations

import hashlib
import inspect
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .full_attention_forward_planner import plan_full_attention_forward
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_attention_forward_qkv_resident import _projection_compute_source
from .tt_lang_attention_forward_resident import _ALL_WORKERS_RANGE
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None
_CBS = 15


@lru_cache(maxsize=8)
def build_tt_lang_qkv_all_core_forward(
    *, sequence_length: int, hidden_features: int, kv_features: int,
    query_heads: int, kv_heads: int, head_dim: int, physical_rank: int,
    lora_scale: float, base_k_tiles_per_phase: int = 1,
    lora_k_tiles_per_phase: int = 2,
    has_bias: bool = False,
    phase_count: int = 3,
) -> Callable:
    """Build the all-core Q/K/V phase used by the phased attention owner."""

    plan = plan_full_attention_forward(
        tokens=sequence_length,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        qkv_bias=has_bias,
    )
    if physical_rank != 32:
        raise ValueError("all-core QKV phase requires physical rank 32")
    seq_tiles = sequence_length // 32
    hidden_tiles = hidden_features // 32
    kv_tiles = kv_features // 32
    if (
        lora_k_tiles_per_phase <= 0
        or hidden_tiles % lora_k_tiles_per_phase
    ):
        raise ValueError("invalid QKV LoRA K phase")
    if hidden_tiles % base_k_tiles_per_phase:
        raise ValueError("invalid QKV frozen-base K phase")
    if phase_count not in (1, 2, 3):
        raise ValueError("all-core QKV phase count must be 1, 2 or 3")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("all-core QKV phase requires TT-XLA") from exc
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
    def _program(
        hidden,
        q_weight, q_bias, q_a, q_b,
        k_weight, k_bias, k_a, k_b,
        v_weight, v_bias, v_a, v_b,
        arena,
    ):
        b0 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b1 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b2 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b3 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b4 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b5 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b6 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b7 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b8 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b9 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b10 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b11 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b12 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b13 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b14 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=16)

        @ttl.datamovement()
        def read():
            with b0.reserve() as block:
                tx = ttl.copy(hidden[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_bias[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_bias[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_bias[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_b[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(13):
                with b0.wait() as value:
                    with b14.reserve() as result:
                        result.store(value)
            x, y = ttl.node(dims=2)
            if x + y < 0:
                with b1.wait() as _: pass
                with b2.wait() as _: pass
                with b3.wait() as _: pass
                with b4.wait() as _: pass
                with b5.wait() as _: pass
                with b6.wait() as _: pass
                with b7.wait() as _: pass
                with b8.wait() as _: pass
                with b9.wait() as _: pass
                with b10.wait() as _: pass
                with b11.wait() as _: pass
                with b12.wait() as _: pass
                with b13.wait() as _: pass

        @ttl.datamovement()
        def write():
            for _ in range(13):
                with b14.wait() as value:
                    tx = ttl.copy(value, arena[0:1, 0:1])
                    tx.wait()

    scale_bf16 = float(
        torch.tensor(lora_scale, dtype=torch.bfloat16).float().item()
    )
    scale_bits = struct.unpack("<I", struct.pack("<f", scale_bf16))[0]
    variant = hashlib.sha256(
        (
            f"qkv-all-core-v5-phase-width-reader-abi:{sequence_length}:{hidden_features}:"
            f"{kv_features}:{query_heads}:{kv_heads}:{head_dim}:"
            f"{physical_rank}:{scale_bf16:.17g}:"
            f"{base_k_tiles_per_phase}:{lora_k_tiles_per_phase}:"
            f"{int(has_bias)}"
            f":{phase_count}"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.qkv_all_core_forward.{variant}",
        arg_roles=("in",) * 13 + ("out",),
        version_tag=f"qkv-all-core-forward-{variant}",
    )
    def operation(
        hidden,
        q_weight, q_bias, q_a, q_b,
        k_weight, k_bias, k_a, k_b,
        v_weight, v_bias, v_a, v_b,
        arena,
    ):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(
            hidden,
            q_weight, q_bias, q_a, q_b,
            k_weight, k_bias, k_a, k_b,
            v_weight, v_bias, v_a, v_b,
            arena,
        )
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != _CBS:
                    raise RuntimeError(
                        f"all-core QKV skeleton expected {_CBS} CBs, "
                        f"got {len(compiled.cb_configs)}"
                    )
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles),
                    "@HIDDEN_TILES@": str(hidden_tiles),
                    "@KV_TILES@": str(kv_tiles),
                    "@BASE_PHASE@": str(base_k_tiles_per_phase),
                    "@LORA_PHASE@": str(lora_k_tiles_per_phase),
                    "@HAS_BIAS@": "1" if has_bias else "0",
                    "@PHASE_COUNT@": str(phase_count),
                    "@Q_BASE@": str(plan.arena.slice("query").page_offset),
                    "@K_BASE@": str(plan.arena.slice("key").page_offset),
                    "@V_BASE@": str(plan.arena.slice("value").page_offset),
                    "@HQ_BASE@": str(
                        plan.arena.slice("q_rank_hidden").page_offset
                    ),
                    "@HK_BASE@": str(
                        plan.arena.slice("k_rank_hidden").page_offset
                    ),
                    "@HV_BASE@": str(
                        plan.arena.slice("v_rank_hidden").page_offset
                    ),
                }
                cpp = Path(__file__).with_name("cpp")
                sources = (
                    ("qkv_all_core_forward_reader.cpp", "noc", (
                        cpp / "qkv_all_core_forward_reader.cpp"
                    ).read_text()),
                    ("qkv_all_core_forward_compute.cpp", "compute",
                     _projection_compute_source(
                         sequence_tiles=seq_tiles,
                         hidden_tiles=hidden_tiles,
                         q_tiles=hidden_tiles,
                         kv_tiles=kv_tiles,
                         base_phase=base_k_tiles_per_phase,
                         lora_phase=lora_k_tiles_per_phase,
                         scale_bits=scale_bits,
                         has_bias=has_bias,
                         projection_workers=110,
                         projection_grid_cols=11,
                         include_o_projection=False,
                         force_inline_projection_phases=False,
                         qkv_phase_count=phase_count,
                         direct_single_projection_kernel=(phase_count == 1),
                     )),
                    ("qkv_all_core_forward_writer.cpp", "noc", (
                        cpp / "qkv_all_core_forward_writer.cpp"
                    ).read_text()),
                )
                generated = Path(compiled.kernel_paths[0][0]).parent
                paths = []
                for filename, thread, source in sources:
                    for marker, value in replacements.items():
                        source = source.replace(marker, value)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(
                            f"unexpanded all-core QKV marker in {filename}"
                        )
                    path = generated / filename
                    path.write_text(source, encoding="utf-8")
                    paths.append((str(path), thread))

                from ttl.dataflow_buffer import CompilerAllocatedDFBConfig
                q_jobs = seq_tiles * hidden_tiles
                local_jobs = max(2, -(-q_jobs // 110))
                capacities = (
                    2 * lora_k_tiles_per_phase,
                    2 * lora_k_tiles_per_phase, 2, 2, 1,
                    2 * base_k_tiles_per_phase,
                    2 * base_k_tiles_per_phase,
                    2, local_jobs, 2, 2, local_jobs, 2,
                    local_jobs, local_jobs,
                )
                for index, blocks in enumerate(capacities):
                    config = CompilerAllocatedDFBConfig(
                        dfb_index=index,
                        num_tiles=1,
                        data_format=(
                            "float32" if index in (2, 7) else "bfloat16"
                        ),
                        block_count=blocks,
                    )
                    config._cb_index = index
                    compiled.cb_configs[index] = config

                import ttnn

                compiled.kernel_paths = paths
                compiled.kernel_configs = [
                    ttnn.ReaderConfigDescriptor(),
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                    ),
                    ttnn.WriterConfigDescriptor(),
                ]
                # Match the compact runtime-address order generated by each
                # independently qualified standalone projection.  The C++
                # reader still uses logical tensor accessor metadata offsets,
                # but obtains addresses from this per-kernel common-argument
                # order.  Keeping the two index spaces explicit avoids an ABI
                # change merely because Q/K/V share one program.
                compiled.kernel_tensor_indices = [
                    [2, 1, 0, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12],
                    [],
                    [13],
                ]
                # This diagnostic has one uniform all-worker range for every
                # kernel and CB and no user semaphore.  Leave the optional
                # multi-range fields absent so the installed TT-MLIR consumes
                # the format-v1 artifact.  Populating redundant range fields
                # upgrades the serializer to format v2, which this PJRT build
                # correctly rejects before reaching silicon.
                compiled.kernel_core_ranges = None
                compiled.kernel_noc_roles = None
                compiled.cb_core_ranges = None
                compiled.semaphore_configs = None
                compiled.num_pipe_sync_semaphores = 0
                compiled.kernel_arg_specs = [[], [], []]
            break
        return result

    operation.output_shape = plan.arena.physical_shape
    operation.forward_plan = plan
    operation.intermediate_dram_bytes = 0
    operation.experimental_backend = "qkv_all_core_forward_v1"
    return operation
