"""Exact one-program streamed causal-LM log-sum-exp for Blackhole."""

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
def build_tt_lang_causal_lm_streaming_logsumexp(
    *, tokens: int, vocabulary_size: int, diagnostic_stage: str = "final",
) -> Callable:
    if diagnostic_stage not in {
        "maximum", "cast", "shifted", "exponentials", "sum",
        "returned_logsum", "probabilities", "final",
    }:
        raise ValueError(
            "diagnostic_stage must be maximum, cast, shifted, exponentials, "
            "sum, returned_logsum, probabilities, or final"
        )
    if tokens <= 0 or tokens % 32:
        raise ValueError("tokens must be positive /32")
    if vocabulary_size <= 0 or vocabulary_size % 32:
        raise ValueError("vocabulary must be positive /32")
    row_tiles = tokens // 32
    vocabulary_tiles = vocabulary_size // 32
    if row_tiles > 10:
        raise ValueError("streaming prototype supports at most 320 tokens")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=(3, row_tiles), fp32_dest_acc_en=True)
    def _program(logits, output):
        def bf16(blocks=1):
            return ttl.make_dataflow_buffer_like(
                logits, shape=(1, 1), block_count=blocks
            )

        def fp32(blocks=1):
            return ttl.make_dataflow_buffer_like(
                output, shape=(1, 1), block_count=blocks
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
        c22, c23, c24, c25, c26, c27, c28 = (
            fp32(), fp32(), fp32(), fp32(), fp32(), fp32(), fp32(),
        )

        @ttl.datamovement()
        def read():
            with c0.reserve() as tile:
                tx = ttl.copy(logits[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with c0.wait() as value:
                with c12.reserve() as result:
                    result.store(ttl.math.typecast(value, torch.float32))
            node_x, node_y = ttl.node(dims=2)
            if node_x + node_y < 0:
                with c1.wait() as _: pass
                with c2.wait() as _: pass
                with c3.wait() as _: pass
                with c4.wait() as _: pass
                with c5.wait() as _: pass
                with c6.wait() as _: pass
                with c7.wait() as _: pass
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

        @ttl.datamovement()
        def write():
            with c12.wait() as value:
                tx = ttl.copy(value, output[0:1, 0:1])
                tx.wait()

    variant = hashlib.sha256(
        f"causal-lm-streaming-logsumexp-v2:{tokens}:{vocabulary_size}:"
        f"{diagnostic_stage}".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.causal_lm_streaming_logsumexp.{variant}",
        arg_roles=("in", "out"),
        version_tag=f"causal-lm-streaming-logsumexp-{variant}",
    )
    def operation(logits, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(logits, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 29:
                    raise RuntimeError(
                        f"streaming logsumexp expected 29 CBs, got {len(compiled.cb_configs)}"
                    )
                generated = Path(compiled.kernel_paths[0][0]).parent
                cpp = Path(__file__).with_name("cpp")
                specifications = [
                    ("causal_lm_stream_maxsum_reader.cpp", "noc", 0, [0]),
                    ("causal_lm_stream_maxsum_compute.cpp", "compute", 0, []),
                    (("causal_lm_stream_maxsend_writer.cpp"
                      if diagnostic_stage in {"cast", "shifted", "exponentials"} else
                      ("causal_lm_stream_maxsum_writer.cpp"
                       if diagnostic_stage in {
                           "returned_logsum", "probabilities", "final"
                       } else
                       "causal_lm_stream_maxsum_debug_writer.cpp")),
                     "noc", 0,
                     [1] if diagnostic_stage in {"maximum", "sum"} else []),
                ]
                if diagnostic_stage != "maximum":
                    specifications.extend([
                        (("causal_lm_stream_probability_reader.cpp"
                          if diagnostic_stage in {
                              "returned_logsum", "probabilities"
                          } else
                          "causal_lm_stream_exp_reader.cpp"), "noc", 1, [0]),
                        (("causal_lm_stream_probability_compute.cpp"
                          if diagnostic_stage in {
                              "returned_logsum", "probabilities"
                          } else
                          "causal_lm_stream_exp_compute.cpp"), "compute", 1, []),
                        ((("causal_lm_stream_cast_debug_writer.cpp"
                           if diagnostic_stage == "cast" else
                           ("causal_lm_stream_shifted_debug_writer.cpp"
                            if diagnostic_stage == "shifted" else
                            "causal_lm_stream_exp_debug_writer.cpp"))
                          if diagnostic_stage in {"cast", "shifted", "exponentials"} else
                          (("causal_lm_stream_return_debug_writer.cpp"
                            if diagnostic_stage == "returned_logsum" else
                            "causal_lm_stream_probability_writer.cpp")
                           if diagnostic_stage in {
                               "returned_logsum", "probabilities"
                           } else
                           "causal_lm_stream_exp_writer.cpp")),
                         "noc", 1,
                         [1] if diagnostic_stage in {
                             "cast", "shifted", "exponentials",
                             "returned_logsum", "probabilities"
                         } else []),
                    ])
                if diagnostic_stage in {
                    "returned_logsum", "probabilities", "final"
                }:
                    specifications.extend([
                        ("causal_lm_stream_log_reader.cpp", "noc", 2, []),
                        ("causal_lm_stream_log_compute.cpp", "compute", 2, []),
                        (("causal_lm_stream_logsend_writer.cpp"
                          if diagnostic_stage in {
                              "returned_logsum", "probabilities"
                          } else
                          "causal_lm_stream_log_writer.cpp"),
                         "noc", 2,
                         [] if diagnostic_stage in {
                             "returned_logsum", "probabilities"
                         } else [1]),
                    ])
                import ttnn
                kernels, configs, indices, ranges, roles = [], [], [], [], []
                for filename, thread, column, tensor_indices in specifications:
                    source = (cpp / filename).read_text().replace(
                        "@VOCAB_TILES@", str(vocabulary_tiles)
                    ).replace(
                        "@DIAGNOSTIC_MAXIMUM@",
                        "1" if diagnostic_stage == "maximum" else "0",
                    )
                    if diagnostic_stage == "cast" and "exp_compute" in filename:
                        marker = "        sfpu_sub_bcast_cols(cast, maximum, expanded, shifted);\n"
                        position = source.find(marker)
                        if position < 0:
                            raise RuntimeError("streaming cast boundary changed")
                        source = source[:position] + "    }\n    maximum.pop_front(1);\n}\n"
                    elif diagnostic_stage == "shifted" and "exp_compute" in filename:
                        source = source.replace(
                            "        exp_tile_to_cb(shifted, exponential);\n", ""
                        )
                    if diagnostic_stage == "returned_logsum":
                        if "probability_reader" in filename:
                            marker = "    // Pass three: replay logits"
                            position = source.find(marker)
                            if position < 0:
                                raise RuntimeError("returned-logsum reader boundary changed")
                            source = source[:position] + "}\n"
                        elif "probability_compute" in filename:
                            marker = "    for (uint32_t tile = 0; tile < wt; ++tile) {\n"
                            first = source.find(marker)
                            position = source.find(marker, first + len(marker))
                            if position < 0:
                                raise RuntimeError("returned-logsum compute boundary changed")
                            source = (
                                source[:position]
                                + "    copy_tile_to_cb(logsum, probability);\n"
                                + "    maximum.pop_front(1);\n}\n"
                            )
                    if diagnostic_stage in {"maximum", "cast", "shifted", "exponentials"} and "maxsum_compute" in filename:
                        marker = "    compute_kernel_lib::reduce<\n        PoolType::SUM"
                        position = source.find(marker)
                        if position < 0:
                            raise RuntimeError("streaming SUM marker changed")
                        source = source[:position] + "    return;\n}\n"
                    if diagnostic_stage in {"maximum", "cast", "shifted", "exponentials"} and "maxsum_reader" in filename:
                        marker = "    DataflowBuffer exponentials(3);"
                        position = source.find(marker)
                        if position < 0:
                            raise RuntimeError("streaming exp receive marker changed")
                        source = source[:position] + "    return;\n}\n"
                    path = generated / f"stream_{filename}"
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
                            for cb in (22, 23, 24, 25, 26, 27):
                                config.unpack_to_dest_mode[cb] = (
                                    ttnn.UnpackToDestMode.UnpackToDestFp32
                                )
                        if column == 2:
                            config.unpack_to_dest_mode[11] = (
                                ttnn.UnpackToDestMode.UnpackToDestFp32
                            )
                    elif "reader" in filename:
                        config = ttnn.ReaderConfigDescriptor()
                    else:
                        config = ttnn.WriterConfigDescriptor()
                    configs.append(config)
                    indices.append(tensor_indices)
                    ranges.append(_range(column, row_tiles))
                    roles.append(
                        None if thread == "compute" else (
                            "reader" if "reader" in filename else "writer"
                        )
                    )
                last_column = {
                    "maximum": 0, "cast": 1, "shifted": 1, "exponentials": 1,
                    "sum": 1, "returned_logsum": 2,
                    "probabilities": 2, "final": 2
                }[
                    diagnostic_stage
                ]
                all_cores = {
                    "start": [0, 0], "end": [last_column, row_tiles - 1]
                }
                compiled.kernel_paths = kernels
                compiled.kernel_configs = configs
                compiled.kernel_tensor_indices = indices
                compiled.kernel_core_ranges = ranges
                compiled.kernel_noc_roles = roles
                compiled.cb_core_ranges = [all_cores for _ in range(29)]
                compiled.semaphore_configs = [
                    {
                        "id": semaphore,
                        "core_type": "worker",
                        "core_range": all_cores,
                        "initial_value": 0,
                    }
                    for semaphore in range(11)
                ]
                compiled.num_pipe_sync_semaphores = 11
                compiled.kernel_arg_specs = [[] for _ in kernels]
            break
        return result

    operation.physical_programs = 1
    operation.public_custom_calls = 1
    operation.full_exponential_tensor_materialized = False
    operation.safe_default = False
    operation.diagnostic_stage = diagnostic_stage
    operation.experimental_backend = (
        f"causal_lm_streaming_logsumexp_v2_{diagnostic_stage}"
    )
    operation._ttl_program = _program
    return operation


__all__ = ["build_tt_lang_causal_lm_streaming_logsumexp"]
