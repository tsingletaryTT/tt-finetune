"""One-call, query-row-sharded resident GQA attention backward."""

from __future__ import annotations

import hashlib
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .attention_block_planner import plan_attention_block
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=8)
def build_tt_lang_attention_resident_backward(
    *, sequence_length: int, query_heads: int, kv_heads: int,
    head_dim: int, scale: float, diagnostic_ds: bool = False,
    diagnostic_dv: bool = False,
    diagnostic_pt: bool = False,
    diagnostic_dq: bool = False,
    diagnostic_dk: bool = False,
    diagnostic_dp: bool = False,
    diagnostic_product: bool = False,
    diagnostic_centered: bool = False,
    diagnostic_rowsum: bool = False,
    diagnostic_unscaled_ds: bool = False,
    probability_bf16_input: bool = False,
    dtype_name: str = "float32",
) -> Callable:
    """Build the exact experimental dQ/dK/dV owner for B=1, S=128."""

    plan = plan_attention_block(
        batch_size=1, sequence_length=sequence_length,
        query_heads=query_heads, kv_heads=kv_heads, head_dim=head_dim,
    )
    if sequence_length != 128 or plan.query_chunks != 1:
        raise ValueError("resident attention backward currently requires S=128")
    if plan.reference_dpdqdv_workers > 110:
        raise ValueError("resident attention backward exceeds the 110-worker grid")
    group = query_heads // kv_heads
    if group % 2 != 1:
        raise ValueError("exact ordered GQA reduction currently requires an odd group")
    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    if dtype_name not in {"float32", "bfloat16"}:
        raise ValueError("resident attention dtype must be float32 or bfloat16")
    element_bytes = 4 if dtype_name == "float32" else 2
    tile_bytes = 32 * 32 * element_bytes
    probability_tile_bytes = (
        tile_bytes
        if probability_bf16_input
        else (32 * 32 * 4 if dtype_name == "bfloat16" else tile_bytes)
    )
    # TTNN's Blackhole BF16 BMM contract also uses FP32 destination
    # accumulation, one K tile per block, and an FP32 partial spill/reload.
    fp32_destination = True
    matmul_phase_tiles = 1
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native attention requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=fp32_destination,
        options="--ttl-matmul-full-fp32",
    )
    def _program(probability, output_gradient, query, key, value, arena):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=max(1, blocks)
            )

        p_row = dfb(probability, seq_tiles)
        p_column = dfb(probability, seq_tiles)
        dout_row = dfb(output_gradient, dim_tiles)
        dout_full = dfb(output_gradient, seq_tiles * dim_tiles)
        q_full = dfb(query, seq_tiles * dim_tiles)
        k = dfb(key, seq_tiles * dim_tiles)
        v = dfb(value, seq_tiles * dim_tiles)
        scaler = dfb(arena, 1)
        pt = dfb(arena, seq_tiles)
        dv = dfb(arena, dim_tiles)
        vt = dfb(arena, seq_tiles * dim_tiles)
        dp = dfb(arena, seq_tiles)
        product = dfb(probability, seq_tiles)
        rowsum = dfb(arena, 1)
        centered = dfb(arena, seq_tiles)
        ds = dfb(arena, seq_tiles)
        ds_scaled = dfb(arena, seq_tiles)
        dq = dfb(arena, dim_tiles)
        partial = dfb(arena, seq_tiles)
        ds_handoff = dfb(arena, seq_tiles)
        ds_column = dfb(arena, seq_tiles)
        # Reused by the exact dK tail as Q.T (dim_tiles x seq_tiles).
        ds_t = dfb(arena, seq_tiles * dim_tiles)
        dk = dfb(arena, dim_tiles)
        peer_dk = dfb(arena, (group - 1) * dim_tiles)
        peer_dv = dfb(arena, (group - 1) * dim_tiles)
        # TTNN's FP32 GQA sum is transpose(group, H) -> FPU H-reduce ->
        # transpose back.  These four CBs form a one-tile streaming version of
        # that exact schedule: packed group rows, reduced row, COL scaler, and
        # a reserved ABI/scratch slot.
        sum_dk_a = dfb(
            arena,
            2 * dim_tiles * group if dtype_name == "bfloat16" else 1,
        )
        sum_dk_b = dfb(arena, 1)
        sum_dv_a = dfb(arena, 1)
        # Pre-transpose dK result (dim_tiles x one key-row tile).
        sum_dv_b = dfb(arena, dim_tiles)
        final_dk = dfb(arena, dim_tiles)
        final_dv = dfb(arena, dim_tiles)
        # BF16 forward packs the FP32 softmax output only for the value
        # matmul. Backward retains the FP32 probability and reuses this final
        # CB slot as the exact BF16 boundary for dV and scaled dS.
        p_bf16 = dfb(arena, seq_tiles)

        @ttl.datamovement()
        def read():
            # ABI probes. The generated body is replaced after lowering.
            with p_row.reserve() as block:
                tx = ttl.copy(probability[0:1, 0:1], block); tx.wait()
            with dout_row.reserve() as block:
                tx = ttl.copy(output_gradient[0:1, 0:1], block); tx.wait()
            with q_full.reserve() as block:
                tx = ttl.copy(query[0:1, 0:1], block); tx.wait()
            with k.reserve() as block:
                tx = ttl.copy(key[0:1, 0:1], block); tx.wait()
            with v.reserve() as block:
                tx = ttl.copy(value[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            with p_row.wait() as item:
                with product.reserve() as result:
                    result.store(item)
            with dout_row.wait() as item:
                with dq.reserve() as result:
                    result.store(item)
            with q_full.wait() as item:
                with dq.reserve() as result:
                    result.store(item)
            with k.wait() as item:
                with dq.reserve() as result:
                    result.store(item)
            with v.wait() as item:
                with dq.reserve() as result:
                    result.store(item)
            x, y = ttl.node(dims=2)
            if y * 11 + x < 0:
                with p_column.wait() as _: pass
                with dout_row.wait() as _: pass
                with dout_full.wait() as _: pass
                with q_full.wait() as _: pass
                with k.wait() as _: pass
                with v.wait() as _: pass
                with scaler.wait() as _: pass
                with pt.wait() as _: pass
                with dv.wait() as _: pass
                with vt.wait() as _: pass
                with dp.wait() as _: pass
                with product.wait() as _: pass
                with rowsum.wait() as _: pass
                with centered.wait() as _: pass
                with ds.wait() as _: pass
                with ds_scaled.wait() as _: pass
                with partial.wait() as _: pass
                with ds_handoff.wait() as _: pass
                with ds_column.wait() as _: pass
                with ds_t.wait() as _: pass
                with dk.wait() as _: pass
                with peer_dk.wait() as _: pass
                with peer_dv.wait() as _: pass
                with sum_dk_a.wait() as _: pass
                with sum_dk_b.wait() as _: pass
                with sum_dv_a.wait() as _: pass
                with sum_dv_b.wait() as _: pass
                with final_dk.wait() as _: pass
                with final_dv.wait() as _: pass
                with p_bf16.wait() as _: pass

        @ttl.datamovement()
        def write():
            for _ in range(4):
                with dq.wait() as item:
                    tx = ttl.copy(item, arena[0:1, 0:1])
                    tx.wait()

    variant = hashlib.sha256(
        (
            f"attention-resident-bwd-v50-autograd-owner-zones:{sequence_length}:{query_heads}:"
            f"{kv_heads}:{head_dim}:{scale:.17g}:{dtype_name}:"
            f"diag-ds={int(diagnostic_ds)}:diag-dp={int(diagnostic_dp)}:"
            f"diag-dv={int(diagnostic_dv)}:"
            f"diag-pt={int(diagnostic_pt)}:"
            f"diag-dq={int(diagnostic_dq)}:"
            f"diag-dk={int(diagnostic_dk)}:"
            f"diag-product={int(diagnostic_product)}"
            f":diag-centered={int(diagnostic_centered)}"
            f":diag-rowsum={int(diagnostic_rowsum)}"
            f":diag-unscaled-ds={int(diagnostic_unscaled_ds)}"
            f":p-bf16-input={int(probability_bf16_input)}"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.attention_resident_backward.{variant}",
        arg_roles=("in", "in", "in", "in", "in", "out"),
        version_tag=f"attention-resident-backward-v50-{variant}",
    )
    def operation(probability, output_gradient, query, key, value, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(probability, output_gradient, query, key, value, arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 32:
                    raise RuntimeError(
                        f"resident attention expected 32 CBs, got {len(compiled.cb_configs)}"
                    )
                reader_indices = [
                    compiled.kernel_tensor_indices[i]
                    for i, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "read" in Path(path).name
                ]
                writer_indices = [
                    compiled.kernel_tensor_indices[i]
                    for i, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "write" in Path(path).name
                ]
                if len(reader_indices) != 1 or sorted(reader_indices[0]) != list(range(5)):
                    raise RuntimeError(f"resident attention reader ABI changed: {reader_indices!r}")
                if writer_indices != [[5]]:
                    raise RuntimeError(f"resident attention writer ABI changed: {writer_indices!r}")
                positions = {tensor: pos for pos, tensor in enumerate(reader_indices[0])}
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles),
                    "@DIM_TILES@": str(dim_tiles),
                    "@QUERY_HEADS@": str(query_heads),
                    "@KV_HEADS@": str(kv_heads),
                    "@SCALE_BITS@": str(scale_bits),
                    "@DIAGNOSTIC_DS@": str(int(diagnostic_ds)),
                    "@DIAGNOSTIC_DV@": str(int(diagnostic_dv)),
                    "@DIAGNOSTIC_PT@": str(int(diagnostic_pt)),
                    "@DIAGNOSTIC_DQ@": str(int(diagnostic_dq)),
                    "@DIAGNOSTIC_DK@": str(int(diagnostic_dk)),
                    "@DIAGNOSTIC_DP@": str(int(diagnostic_dp)),
                    "@DIAGNOSTIC_PRODUCT@": str(int(diagnostic_product)),
                    "@DIAGNOSTIC_CENTERED@": str(int(diagnostic_centered)),
                    "@DIAGNOSTIC_ROWSUM@": str(int(diagnostic_rowsum)),
                    "@DIAGNOSTIC_UNSCALED_DS@": str(int(diagnostic_unscaled_ds)),
                    "@DIAGNOSTIC_DOUT_HANDOFF@": "0",
                    "@P_INPUT_BF16@": str(int(probability_bf16_input)),
                    "@TILE_BYTES@": str(tile_bytes),
                    "@P_TILE_BYTES@": str(probability_tile_bytes),
                    "@ELEMENT_BYTES@": str(element_bytes),
                    "@BF16@": str(int(dtype_name == "bfloat16")),
                    "@FP32_DEST@": str(int(fp32_destination)),
                    "@MATMUL_PHASE_TILES@": str(matmul_phase_tiles),
                }
                cpp = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp / "attention_resident_backward_reader.cpp").read_text(),
                    "compute": (cpp / "attention_resident_backward_compute.cpp").read_text(),
                    "write": (cpp / "attention_resident_backward_writer.cpp").read_text(),
                }
                sources["read"] = re.sub(
                    r"get_common_arg_val<uint32_t>\((\d+)\)",
                    lambda m: f"get_common_arg_val<uint32_t>({positions[int(m.group(1))]})",
                    sources["read"],
                )
                sources["read"] = re.sub(
                    r"get_tensor_accessor_args_cta_offset<(\d+), 32>\(\),\s*(\d+)",
                    lambda m: (
                        f"get_tensor_accessor_args_cta_offset<{m.group(1)}, 32>(), "
                        f"{positions[int(m.group(1))]}"
                    ),
                    sources["read"],
                )
                for kind, source in tuple(sources.items()):
                    for placeholder, value_text in replacements.items():
                        source = source.replace(placeholder, value_text)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(f"unexpanded resident attention {kind} placeholder")
                    sources[kind] = source
                for path, thread in compiled.kernel_paths:
                    kind = "compute" if thread == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")
                if dtype_name == "bfloat16":
                    from ttl.dataflow_buffer import CompilerAllocatedDFBConfig

                    fp32_buffers = (
                        (12, seq_tiles),
                        (13, 1),
                        (14, seq_tiles),
                        (15, seq_tiles),
                        (18, seq_tiles),
                    )
                    for cb_index, block_count in fp32_buffers:
                        partial_config = CompilerAllocatedDFBConfig(
                            dfb_index=cb_index,
                            num_tiles=1,
                            data_format="float32",
                            block_count=block_count,
                        )
                        partial_config._cb_index = cb_index
                        compiled.cb_configs[cb_index] = partial_config
                compiled.num_pipe_sync_semaphores = 12
                for config in compiled.kernel_configs:
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, "HiFi4")
                    if hasattr(config, "math_approx_mode"):
                        # TTNN's production large-block BMM uses APPROX=true.
                        # The softmax prefix contains only exact mul/sub/typecast
                        # SFPU operations, so it does not require an approximate
                        # transcendental; qualify both contracts together.
                        config.math_approx_mode = True
                    if hasattr(config, "fp32_dest_acc_en"):
                        config.fp32_dest_acc_en = fp32_destination
                    if hasattr(config, "unpack_to_dest_mode"):
                        import ttnn
                        if dtype_name == "bfloat16":
                            # The production graph keeps softmax probability
                            # in FP32.  TTNN's elementwise product descriptor
                            # unpacks both physical FP32 operands as Float32,
                            # not TF32.  Preserve that contract through the
                            # entire FP32 softmax-backward prefix while leaving
                            # every BF16 matmul operand on its qualified path.
                            # CB12 is the product consumed by REDUCE_ROW and
                            # must retain TTNN's normal TF32 reduction unpack.
                            # CB18 is the matmul partial-reload buffer. TTNN's
                            # BF16 BMM intentionally reloads it through the
                            # default TF32 path; marking it full-FP32 changes
                            # dV/dQ/dK bits. CB15 temporarily stages full-FP32
                            # dP before being reused for dS.
                            full_fp32 = {13, 14, 15}
                            if not probability_bf16_input:
                                full_fp32.update({0, 1})
                        else:
                            full_fp32 = {
                                # dS (CB15) and the raw vector result (CB16)
                                # return to TTNN's normal unpack path at their
                                # respective matmul/scalar boundaries.
                                0, 11, 13, 14, 15,
                            }
                        if (
                            len(config.unpack_to_dest_mode) == 0
                            and not probability_bf16_input
                        ):
                            for cb_index in range(64):
                                config.unpack_to_dest_mode.append(
                                    ttnn.UnpackToDestMode.UnpackToDestFp32
                                    if cb_index in full_fp32
                                    else ttnn.UnpackToDestMode.Default
                                )
            break
        return result

    operation.plan = plan
    operation.group_size = group
    operation.dtype_name = dtype_name
    operation.output_tile_shape = (
        (query_heads + 2 * kv_heads) * sequence_length,
        head_dim,
    )
    operation.experimental_backend = "resident_attention_backward_v50_exact_candidate"
    return operation
