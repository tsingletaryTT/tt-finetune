"""One-program exact streamed cross-entropy dLogits for Blackhole."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache

ttl = None
tt_lang_operation = None


def _range(column: int, row_tiles: int) -> dict:
    return {"start": [column, 0], "end": [column, row_tiles - 1]}


@lru_cache(maxsize=8)
def build_tt_lang_causal_lm_streaming_dlogits(
    *, tokens: int, vocabulary_size: int, diagnostic_stage: str = "final",
    scale_source: str = "metadata",
) -> Callable:
    """Build the exact resident logsum/probability/dLogits boundary.

    ``metadata`` is one INT32 tile per 32-token row block.  Logical column 0
    stores target indices; column 1 stores the full FP32 sparse-seed bits and
    column 2 stores the TF32-truncated row-seed-sum bits.
    This compact ABI is temporary until the loss-count reduction is moved
    into the same program, but already avoids any vocabulary-sized one-hot,
    probability or FP32 dLogits operand.
    """

    if tokens <= 0 or tokens % 32:
        raise ValueError("tokens must be positive /32")
    if vocabulary_size <= 0 or vocabulary_size % 32:
        raise ValueError("vocabulary must be positive /32")
    if diagnostic_stage not in {
        "row_seed", "seed", "product", "loss_terms", "loss_numerator",
        "final"
    }:
        raise ValueError(
            "diagnostic_stage must be row_seed, seed, product, loss_terms, "
            "loss_numerator, or final"
        )
    if scale_source not in {"metadata", "device_global"}:
        raise ValueError("scale_source must be metadata or device_global")
    row_tiles = tokens // 32
    vocabulary_tiles = vocabulary_size // 32
    if row_tiles > 10:
        raise ValueError("streaming dLogits prototype supports at most 320 tokens")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=(5, row_tiles), fp32_dest_acc_en=True)
    def _program(logits, metadata, fp32_output, dlogits_output):
        def bf16(blocks=1):
            return ttl.make_dataflow_buffer_like(
                logits, shape=(1, 1), block_count=blocks
            )

        def fp32(blocks=1):
            return ttl.make_dataflow_buffer_like(
                fp32_output, shape=(1, 1), block_count=blocks
            )

        c0, c1 = bf16(2), bf16()
        c2, c3, c4, c5, c6 = fp32(), fp32(), fp32(), fp32(), fp32()
        c7 = bf16(2)
        c8, c9, c10, c11, c12, c13 = (
            fp32(), fp32(), fp32(), fp32(), fp32(), fp32()
        )
        c14, c15, c16, c17, c18, c19, c20, c21 = (
            fp32(), fp32(), fp32(), fp32(),
            fp32(), fp32(), fp32(), fp32(),
        )
        c22, c23, c24, c25, c26, c27, c28, c29, c30 = (
            fp32(), fp32(), fp32(), fp32(), fp32(),
            fp32(), fp32(), fp32(), fp32(),
        )
        c31 = ttl.make_dataflow_buffer_like(
            metadata, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with c0.reserve() as tile:
                tx = ttl.copy(logits[0:1, 0:1], tile)
                tx.wait()
            with c7.reserve() as tile:
                tx = ttl.copy(logits[0:1, 0:1], tile)
                tx.wait()
            with c31.reserve() as tile:
                tx = ttl.copy(metadata[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with c0.wait() as value:
                with c12.reserve() as result:
                    result.store(ttl.math.typecast(value, torch.float32))
            with c7.wait() as value:
                with c1.reserve() as result:
                    result.store(value)
            with c31.wait() as _:
                pass
            node_x, node_y = ttl.node(dims=2)
            if node_x + node_y < 0:
                with c2.wait() as _: pass
                with c3.wait() as _: pass
                with c4.wait() as _: pass
                with c5.wait() as _: pass
                with c6.wait() as _: pass
                with c8.wait() as _: pass
                with c9.wait() as _: pass
                with c10.wait() as _: pass
                with c11.wait() as _: pass
                with c13.wait() as _: pass
                with c14.wait() as _: pass
                with c15.wait() as _: pass
                with c16.wait() as _: pass
                with c17.wait() as _: pass
                with c18.wait() as _: pass
                with c19.wait() as _: pass
                with c20.wait() as _: pass
                with c21.wait() as _: pass
                with c22.wait() as _: pass
                with c23.wait() as _: pass
                with c24.wait() as _: pass
                with c25.wait() as _: pass
                with c26.wait() as _: pass
                with c27.wait() as _: pass
                with c28.wait() as _: pass
                with c29.wait() as _: pass
                with c30.wait() as _: pass

        @ttl.datamovement()
        def write():
            with c12.wait() as value:
                tx = ttl.copy(value, fp32_output[0:1, 0:1])
                tx.wait()
            with c1.wait() as value:
                tx = ttl.copy(value, dlogits_output[0:1, 0:1])
                tx.wait()

    variant = hashlib.sha256(
        f"causal-lm-streaming-dlogits-v2:{tokens}:{vocabulary_size}:"
        f"{diagnostic_stage}:{scale_source}".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.causal_lm_streaming_dlogits.{variant}",
        arg_roles=("in", "in", "out", "out"),
        version_tag=f"causal-lm-streaming-dlogits-{variant}",
    )
    def operation(logits, metadata, fp32_output, dlogits_output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(logits, metadata, fp32_output, dlogits_output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 32:
                    raise RuntimeError(
                        f"streaming dLogits expected 32 CBs, got {len(compiled.cb_configs)}"
                    )
                generated = Path(compiled.kernel_paths[0][0]).parent
                cpp = Path(__file__).with_name("cpp")
                specifications = [
                    ("causal_lm_stream_maxsum_reader.cpp", "noc", 0, [0]),
                    ("causal_lm_stream_maxsum_compute.cpp", "compute", 0, []),
                    ("causal_lm_stream_maxsum_writer.cpp", "noc", 0, []),
                    ("causal_lm_stream_dlogits_reader.cpp", "noc", 1, [0, 1]),
                    ("causal_lm_stream_dlogits_compute.cpp", "compute", 1, []),
                    ("causal_lm_stream_dlogits_writer.cpp", "noc", 1, [3]),
                    ("causal_lm_stream_log_reader.cpp", "noc", 2, []),
                    ("causal_lm_stream_log_compute.cpp", "compute", 2, []),
                    ("causal_lm_stream_logsend_writer.cpp", "noc", 2, []),
                    ("causal_lm_stream_loss_reader.cpp", "noc", 3, []),
                    ("causal_lm_stream_loss_compute.cpp", "compute", 3, []),
                    ("causal_lm_stream_losssend_writer.cpp", "noc", 3, []),
                    ("causal_lm_stream_loss_aggregate_reader.cpp", "noc", 4, [1]),
                    ("causal_lm_stream_loss_aggregate_compute.cpp", "compute", 4, []),
                    ("causal_lm_stream_loss_aggregate_writer.cpp", "noc", 4, [2]),
                ]
                import ttnn
                kernels, configs, indices, ranges, roles = [], [], [], [], []
                for filename, thread, column, tensor_indices in specifications:
                    source = (cpp / filename).read_text().replace(
                        "@VOCAB_TILES@", str(vocabulary_tiles)
                    ).replace(
                        "@VOCAB_SIZE@", str(vocabulary_size)
                    ).replace(
                        "@ROW_TILES@", str(row_tiles)
                    ).replace(
                        "@DEVICE_COUNT@",
                        "true" if scale_source == "device_global" else "false",
                    ).replace(
                        "cta_offset<0, 29>", "cta_offset<0, 32>"
                    ).replace(
                        "@DIAGNOSTIC_ROW_SEED@",
                        "1" if diagnostic_stage == "row_seed" else "0",
                    ).replace(
                        "@DIAGNOSTIC_SEED@",
                        "1" if diagnostic_stage == "seed" else "0",
                    ).replace(
                        "@DIAGNOSTIC_PRODUCT@",
                        "1" if diagnostic_stage == "product" else "0",
                    ).replace(
                        "@DIAGNOSTIC_LOSS_TERMS@",
                        "1" if diagnostic_stage == "loss_terms" else "0",
                    ).replace(
                        "@OUTPUT_LOSS@",
                        "1" if diagnostic_stage == "final" else "0",
                    )
                    path = generated / f"stream_dlogits_{filename}"
                    path.write_text(source, encoding="utf-8")
                    kernels.append((str(path), thread))
                    if thread == "compute":
                        config = ttnn.ComputeConfigDescriptor(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            math_approx_mode=False,
                            fp32_dest_acc_en=True,
                        )
                        while len(config.unpack_to_dest_mode) < 64:
                            config.unpack_to_dest_mode.append(
                                ttnn.UnpackToDestMode.Default
                        )
                        if column == 1:
                            # c16 was write-only in logsum/probability
                            # qualification. dLogits consumes that FP32 exp
                            # tile locally, so it too requires direct FP32
                            # unpack instead of the default BF16 interpretation.
                            for cb in (16, 18, 20, *range(22, 31)):
                                config.unpack_to_dest_mode[cb] = (
                                    ttnn.UnpackToDestMode.UnpackToDestFp32
                                )
                        if column == 2:
                            config.unpack_to_dest_mode[11] = (
                                ttnn.UnpackToDestMode.UnpackToDestFp32
                            )
                        if column == 3:
                            for cb in (
                                (13, 15)
                                if diagnostic_stage == "loss_terms" else ()
                            ):
                                config.unpack_to_dest_mode[cb] = (
                                    ttnn.UnpackToDestMode.UnpackToDestFp32
                                )
                        if column == 4:
                            for cb in (16, 17, 18, 20):
                                config.unpack_to_dest_mode[cb] = (
                                    ttnn.UnpackToDestMode.UnpackToDestFp32
                                )
                    elif "reader" in filename:
                        config = ttnn.ReaderConfigDescriptor()
                    else:
                        config = ttnn.WriterConfigDescriptor()
                    configs.append(config)
                    indices.append(tensor_indices)
                    ranges.append(_range(
                        column, 1 if column == 4 else row_tiles
                    ))
                    roles.append(
                        None if thread == "compute" else (
                            "reader" if "reader" in filename else "writer"
                        )
                    )
                all_cores = {
                    "start": [0, 0], "end": [4, row_tiles - 1]
                }
                compiled.kernel_paths = kernels
                compiled.kernel_configs = configs
                compiled.kernel_tensor_indices = indices
                compiled.kernel_core_ranges = ranges
                compiled.kernel_noc_roles = roles
                compiled.cb_core_ranges = [all_cores for _ in range(32)]
                compiled.semaphore_configs = [
                    {
                        "id": semaphore,
                        "core_type": "worker",
                        "core_range": all_cores,
                        "initial_value": 0,
                    }
                    for semaphore in range(16)
                ]
                compiled.num_pipe_sync_semaphores = 16
                compiled.kernel_arg_specs = [[] for _ in kernels]
            break
        return result

    operation.physical_programs = 1
    operation.public_custom_calls = 1
    operation.safe_default = False
    operation.metadata_contract = (
        "target_i32_only" if scale_source == "device_global" else
        "target_i32_full_seed_fp32_and_row_seed_tf32_bits"
    )
    operation.scale_source = scale_source
    operation.diagnostic_stage = diagnostic_stage
    operation.experimental_backend = "causal_lm_streaming_dlogits_v1"
    operation._ttl_program = _program
    return operation


__all__ = ["build_tt_lang_causal_lm_streaming_dlogits"]
