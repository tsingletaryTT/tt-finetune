"""One-program exact loss/dLogits/frozen-LM-head backward on Blackhole.

This extends the accepted streaming cross-entropy owner with six row-local
head workers.  BF16 dLogits is rounded at the accepted boundary but travels
only through L1/NoC; the public outputs are one FP32 loss tile and the final
BF16 dNormalized tensor.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_lora import _FP32_DTYPE_ONLY_TENSOR

ttl = None
tt_lang_operation = None


def _range(column: int, row_tiles: int) -> dict:
    return {"start": [column, 0], "end": [column, row_tiles - 1]}


@lru_cache(maxsize=8)
def build_tt_lang_causal_lm_resident_head_backward(
    *, tokens: int, vocabulary_size: int, hidden_features: int,
    scale_source: str = "device_global",
    head_reduction_tiles_per_phase: int | None = None,
) -> Callable:
    """Build ``loss + dLogits @ frozen_head`` as one physical program."""

    dimensions = (tokens, vocabulary_size, hidden_features)
    if any(value <= 0 or value % 32 for value in dimensions):
        raise ValueError("tokens, vocabulary and hidden size must be positive /32")
    if scale_source not in {"metadata", "device_global"}:
        raise ValueError("scale_source must be metadata or device_global")
    row_tiles = tokens // 32
    vocabulary_tiles = vocabulary_size // 32
    hidden_tiles = hidden_features // 32
    if head_reduction_tiles_per_phase is None:
        # TTNN's accepted SmolLM2 frozen-head contract spills/reloads after
        # two vocabulary tiles. Other shapes remain diagnostic-only until
        # their generic schedule has been accepted independently.
        head_reduction_tiles_per_phase = (
            2
            if (vocabulary_size, hidden_features) == (49152, 576)
            else 1
        )
    if (
        head_reduction_tiles_per_phase <= 0
        or vocabulary_tiles % head_reduction_tiles_per_phase
    ):
        raise ValueError(
            "head reduction phase must be positive and divide vocabulary tiles"
        )
    if row_tiles > 10:
        raise ValueError("resident head backward supports at most 320 tokens")
    head_workers = min(6, hidden_tiles)
    head_start_x = 5
    max_head_jobs = (hidden_tiles + head_workers - 1) // head_workers

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator

    arena_tiles = hidden_tiles + 1

    @ttl.operation(grid=(11, row_tiles), fp32_dest_acc_en=True)
    def _program(logits, metadata, frozen_weight, output_arena):
        def bf16(blocks=1):
            return ttl.make_dataflow_buffer_like(
                logits, shape=(1, 1), block_count=blocks
            )

        def work(blocks=1):
            return ttl.make_dataflow_buffer_like(
                output_arena, shape=(1, 1), block_count=blocks
            )

        # c0/c1/c2/c7 are deliberately reused by the disjoint head-worker
        # range as dLogits/result/FP32-partial/weight.
        c0 = bf16(max(2, head_reduction_tiles_per_phase))
        c1 = ttl.make_dataflow_buffer_like(
            output_arena, shape=(1, 1), block_count=max_head_jobs
        )
        c2 = work(max_head_jobs)
        c3, c4, c5, c6 = work(), work(), work(), work()
        c7 = bf16(max(2, max_head_jobs))
        c8, c9, c10, c11, c12, c13 = (
            work(), work(), work(), work(), work(), work()
        )
        c14, c15, c16, c17, c18, c19, c20, c21 = (
            work(), work(), work(), work(),
            work(), work(), work(), work(),
        )
        c22, c23, c24, c25, c26, c27, c28, c29, c30 = (
            work(), work(), work(), work(), work(),
            work(), work(), work(), work(),
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
                tx = ttl.copy(frozen_weight[0:1, 0:1], tile)
                tx.wait()
            with c31.reserve() as tile:
                tx = ttl.copy(metadata[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with c0.wait() as value:
                with c12.reserve() as result:
                    result.store(value)
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
            with c1.wait() as value:
                tx = ttl.copy(value, output_arena[0:1, 0:1])
                tx.wait()

    variant = hashlib.sha256(
        f"causal-lm-resident-head-bwd-v3-job-phased:{tokens}:{vocabulary_size}:"
        f"{hidden_features}:{scale_source}:"
        f"{head_reduction_tiles_per_phase}".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.causal_lm_resident_head_backward.{variant}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"causal-lm-resident-head-backward-{variant}",
    )
    def operation(logits, metadata, frozen_weight, output_arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(logits, metadata, frozen_weight, output_arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 32:
                    raise RuntimeError(
                        "resident head backward expected 32 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                generated = Path(compiled.kernel_paths[0][0]).parent
                cpp = Path(__file__).with_name("cpp")
                specifications = [
                    ("causal_lm_stream_maxsum_reader.cpp", "noc", 0, [0]),
                    ("causal_lm_stream_maxsum_compute.cpp", "compute", 0, []),
                    ("causal_lm_stream_maxsum_writer.cpp", "noc", 0, []),
                    ("causal_lm_stream_dlogits_reader.cpp", "noc", 1, [0, 1]),
                    ("causal_lm_stream_dlogits_compute.cpp", "compute", 1, []),
                    ("causal_lm_stream_dlogits_headsend_writer.cpp", "noc", 1, []),
                    ("causal_lm_stream_log_reader.cpp", "noc", 2, []),
                    ("causal_lm_stream_log_compute.cpp", "compute", 2, []),
                    ("causal_lm_stream_logsend_writer.cpp", "noc", 2, []),
                    ("causal_lm_stream_loss_reader.cpp", "noc", 3, []),
                    ("causal_lm_stream_loss_compute.cpp", "compute", 3, []),
                    ("causal_lm_stream_losssend_writer.cpp", "noc", 3, []),
                    ("causal_lm_stream_loss_aggregate_reader.cpp", "noc", 4, [1]),
                    ("causal_lm_stream_loss_aggregate_compute.cpp", "compute", 4, []),
                    ("causal_lm_stream_loss_arena_writer.cpp", "noc", 4, [3]),
                    ("causal_lm_stream_head_reader.cpp", "noc", 5, [2]),
                    ("causal_lm_stream_head_compute.cpp", "compute", 5, []),
                    ("causal_lm_stream_head_writer.cpp", "noc", 5, [3]),
                ]
                import ttnn
                kernels, configs, indices, ranges, roles = [], [], [], [], []
                for filename, thread, column, tensor_indices in specifications:
                    source = (cpp / filename).read_text()
                    replacements = {
                        "@VOCAB_TILES@": str(vocabulary_tiles),
                        "@VOCAB_SIZE@": str(vocabulary_size),
                        "@ROW_TILES@": str(row_tiles),
                        "@HIDDEN_TILES@": str(hidden_tiles),
                        "@HEAD_WORKERS@": str(head_workers),
                        "@HEAD_START_X@": str(head_start_x),
                        "@HEAD_REDUCTION_TILES_PER_PHASE@": str(
                            head_reduction_tiles_per_phase
                        ),
                        "@ARENA_TILES@": str(arena_tiles),
                        "@DEVICE_COUNT@": (
                            "true" if scale_source == "device_global" else "false"
                        ),
                        "@DIAGNOSTIC_ROW_SEED@": "0",
                        "@DIAGNOSTIC_SEED@": "0",
                        "@DIAGNOSTIC_PRODUCT@": "0",
                        "@DIAGNOSTIC_LOSS_TERMS@": "0",
                        "@OUTPUT_LOSS@": "1",
                    }
                    for placeholder, value in replacements.items():
                        source = source.replace(placeholder, value)
                    source = source.replace("cta_offset<0, 29>", "cta_offset<0, 32>")
                    path = generated / f"resident_head_{filename}"
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
                            for cb in (16, 18, 20, *range(22, 31)):
                                config.unpack_to_dest_mode[cb] = (
                                    ttnn.UnpackToDestMode.UnpackToDestFp32
                                )
                        elif column == 2:
                            config.unpack_to_dest_mode[11] = (
                                ttnn.UnpackToDestMode.UnpackToDestFp32
                            )
                        elif column == 4:
                            for cb in (16, 17, 18, 20):
                                config.unpack_to_dest_mode[cb] = (
                                    ttnn.UnpackToDestMode.UnpackToDestFp32
                                )
                        elif column == 5:
                            config.unpack_to_dest_mode[2] = (
                                ttnn.UnpackToDestMode.UnpackToDestFp32
                            )
                    elif "reader" in filename:
                        config = ttnn.ReaderConfigDescriptor()
                    else:
                        config = ttnn.WriterConfigDescriptor()
                    configs.append(config)
                    indices.append(tensor_indices)
                    ranges.append(_range(
                        column,
                        1 if column == 4 else row_tiles,
                    ) if column < 5 else {
                        "start": [head_start_x, 0],
                        "end": [head_start_x + head_workers - 1, row_tiles - 1],
                    })
                    roles.append(
                        None if thread == "compute" else (
                            "reader" if "reader" in filename else "writer"
                        )
                    )
                all_cores = {"start": [0, 0], "end": [10, row_tiles - 1]}
                for cb_index in (*range(2, 7), *range(8, 31)):
                    compiled.cb_configs[cb_index].tensor = (
                        _FP32_DTYPE_ONLY_TENSOR
                    )
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
    operation.resident_compute_islands = 6
    # This owner deliberately remains a qualification probe. The loss side
    # is exact, but its private dLogits -> frozen-head reduction has not yet
    # reproduced the canonical graph's BF16 dHidden contract. Runtime
    # selection must not infer production eligibility from this builder.
    operation.safe_default = False
    operation.qualification_status = "diagnostic_nonexact"
    operation.production_exact_shapes = ()
    operation.scale_source = scale_source
    operation.head_reduction_tiles_per_phase = (
        head_reduction_tiles_per_phase
    )
    operation.experimental_backend = "causal_lm_resident_head_backward_v3"
    operation._ttl_program = _program
    return operation


__all__ = ["build_tt_lang_causal_lm_resident_head_backward"]
