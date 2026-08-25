"""Exact full-width FP32 reduction/log phase for the resident LM edge."""

from __future__ import annotations

import hashlib
import inspect
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=8)
def build_tt_lang_causal_lm_logsum(
    *, tokens: int, vocabulary_size: int, diagnostic_stage: str = "final",
) -> Callable:
    if diagnostic_stage not in {"sum", "final"}:
        raise ValueError("diagnostic_stage must be 'sum' or 'final'")
    if tokens <= 0 or tokens % 32:
        raise ValueError("logsum tokens must be positive /32")
    if vocabulary_size <= 0 or vocabulary_size % 32:
        raise ValueError("logsum vocabulary must be positive /32")
    row_tiles = tokens // 32
    vocabulary_tiles = vocabulary_size // 32
    if row_tiles > 110:
        raise ValueError("logsum row tiles exceed the Blackhole grid")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native logsum requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    options: dict[str, object] = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": True,
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        options["math_approx_mode"] = False

    @ttl.operation(**options)
    def _program(exponentials, output):
        def dfb(tensor, blocks=1):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=blocks
            )

        c0 = dfb(exponentials, 2)
        c1 = dfb(exponentials)
        c2 = dfb(exponentials)
        c3 = dfb(output)
        c4 = dfb(output)

        @ttl.datamovement()
        def read():
            with c0.reserve() as tile:
                transaction = ttl.copy(exponentials[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with c0.wait() as value:
                with c4.reserve() as result:
                    result.store(value)
            node_x, node_y = ttl.node(dims=2)
            if node_x + node_y < 0:
                with c1.wait() as _: pass
                with c2.wait() as _: pass
                with c3.wait() as _: pass

        @ttl.datamovement()
        def write():
            with c4.wait() as value:
                transaction = ttl.copy(value, output[0:1, 0:1])
                transaction.wait()

    variant = hashlib.sha256(
        f"causal-lm-logsum-v3:{tokens}:{vocabulary_size}:{diagnostic_stage}".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.causal_lm_logsum.{variant}",
        arg_roles=("in", "out"),
        version_tag=f"causal-lm-logsum-{variant}",
    )
    def operation(exponentials, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(exponentials, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 5:
                    raise RuntimeError(
                        f"causal-LM logsum expected 5 CBs, got {len(compiled.cb_configs)}"
                    )
                reader_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "read" in Path(path).name
                ]
                writer_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "write" in Path(path).name
                ]
                if reader_indices != [[0]] or writer_indices != [[1]]:
                    raise RuntimeError(
                        "causal-LM logsum tensor ABI changed: "
                        f"reader={reader_indices}, writer={writer_indices}"
                    )
                cpp = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp / "causal_lm_logsum_reader.cpp").read_text(),
                    "compute": (cpp / "causal_lm_logsum_compute.cpp").read_text(),
                    "write": (cpp / "causal_lm_logsum_writer.cpp").read_text(),
                }
                for kind in sources:
                    sources[kind] = (
                        sources[kind]
                        .replace("@VOCAB_TILES@", str(vocabulary_tiles))
                        .replace("@ROW_TILES@", str(row_tiles))
                        .replace(
                            "@FINAL_LOG@",
                            (
                                "DataflowBuffer sum(cb_sum);\n"
                                "    DataflowBuffer logsum(cb_logsum);\n"
                                "    log_tile_to_cb(sum, logsum);"
                                if diagnostic_stage == "final"
                                else ""
                            ),
                        )
                        .replace(
                            "@OUTPUT_CB@",
                            "4" if diagnostic_stage == "final" else "3",
                        )
                    )
                for path, thread in compiled.kernel_paths:
                    kind = "compute" if thread == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")
                for config in compiled.kernel_configs:
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, "HiFi4")
                    if hasattr(config, "math_approx_mode"):
                        config.math_approx_mode = False
                    if hasattr(config, "fp32_dest_acc_en"):
                        config.fp32_dest_acc_en = True
                    if hasattr(config, "unpack_to_dest_mode"):
                        import ttnn
                        while len(config.unpack_to_dest_mode) < 64:
                            config.unpack_to_dest_mode.append(
                                ttnn.UnpackToDestMode.Default
                            )
                        # c0 must retain generic SUM's normal FPU/TF32 input
                        # contract; only the separate unary LOG consumes c3
                        # through the full-FP32 SFPU unpack path.
                        config.unpack_to_dest_mode[3] = (
                            ttnn.UnpackToDestMode.UnpackToDestFp32
                        )
            break
        return result

    operation.output_shape = (tokens, 32)
    operation.physical_programs = 1
    operation.safe_default = False
    operation.experimental_backend = (
        f"causal_lm_logsum_exact_candidate_v3_{diagnostic_stage}"
    )
    operation._ttl_program = _program
    return operation


__all__ = ["build_tt_lang_causal_lm_logsum"]
