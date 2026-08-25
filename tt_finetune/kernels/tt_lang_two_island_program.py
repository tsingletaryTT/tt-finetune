"""PJRT bridge smoke test for one mixed-config TT-Metal program.

This operation is intentionally tiny. It proves that artifact format v2 can
carry the already accepted 100-core FP32 matmul + 10-core BF16 activation
program through Torch/XLA and TT-MLIR without changing the safe trainer path.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None

_MATMUL_RANGE = {"start": [0, 0], "end": [9, 9]}
_ACTIVATION_RANGE = {"start": [10, 0], "end": [10, 9]}
_ALL_WORKERS_RANGE = {"start": [0, 0], "end": [10, 9]}


def _source(name: str, *defines: str) -> str:
    body = (Path(__file__).with_name("cpp") / name).read_text(
        encoding="utf-8"
    )
    prefix = "".join(f"#define {define} 1\n" for define in defines)
    return prefix + body


@lru_cache(maxsize=1)
def build_tt_lang_two_island_matmul_activation() -> Callable:
    """Build the four-operand PJRT smoke operation."""

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("two-island PJRT proof requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _program(lhs, weight, rhs, output):
        lhs_cb = ttl.make_dataflow_buffer_like(
            lhs, shape=(1, 1), block_count=1
        )
        weight_cb = ttl.make_dataflow_buffer_like(
            weight, shape=(1, 1), block_count=1
        )
        rhs_cb = ttl.make_dataflow_buffer_like(
            rhs, shape=(1, 1), block_count=1
        )
        scratch0 = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=1
        )
        scratch1 = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=1
        )
        scratch2 = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=1
        )
        output_cb = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with lhs_cb.reserve() as block:
                tx = ttl.copy(lhs[0:1, 0:1], block)
                tx.wait()
            with weight_cb.reserve() as block:
                tx = ttl.copy(weight[0:1, 0:1], block)
                tx.wait()
            with rhs_cb.reserve() as block:
                tx = ttl.copy(rhs[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            with (
                lhs_cb.wait() as lhs_tile,
                weight_cb.wait() as weight_tile,
                output_cb.reserve() as result,
            ):
                result.store(lhs_tile @ weight_tile)
            node_col, node_row = ttl.node(dims=2)
            if node_col + node_row < 0:
                with rhs_cb.wait() as _value:
                    pass
                with scratch0.wait() as _value:
                    pass
                with scratch1.wait() as _value:
                    pass
                with scratch2.wait() as _value:
                    pass

        @ttl.datamovement()
        def write():
            with output_cb.wait() as value:
                tx = ttl.copy(value, output[0:1, 0:1])
                tx.wait()

    variant = hashlib.sha256(
        b"two-island-pjrt-matmul-activation-v2"
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.two_island_pjrt.{variant}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"two-island-pjrt-{variant}",
    )
    def operation(lhs, weight, rhs, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(lhs, weight, rhs, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 7:
                    raise RuntimeError(
                        "two-island PJRT proof expected seven CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                generated_dir = Path(compiled.kernel_paths[0][0]).parent
                sources = (
                    (
                        "two_island_pjrt_producer_reader.cpp",
                        "noc",
                        _source(
                            "two_island_producer_reader.cpp",
                            "TWO_ISLAND_REAL_MATMUL",
                            "TWO_ISLAND_PJRT_ARTIFACT",
                        ),
                    ),
                    (
                        "two_island_pjrt_producer_compute.cpp",
                        "compute",
                        _source(
                            "two_island_producer_compute.cpp",
                            "TWO_ISLAND_REAL_MATMUL",
                        ),
                    ),
                    (
                        "two_island_pjrt_producer_writer.cpp",
                        "noc",
                        _source(
                            "two_island_producer_writer.cpp",
                            "TWO_ISLAND_REAL_MATMUL",
                        ),
                    ),
                    (
                        "two_island_pjrt_consumer_reader.cpp",
                        "noc",
                        _source(
                            "two_island_consumer_reader.cpp",
                            "TWO_ISLAND_PJRT_ARTIFACT",
                        ),
                    ),
                    (
                        "two_island_pjrt_consumer_compute.cpp",
                        "compute",
                        _source("two_island_consumer_compute.cpp"),
                    ),
                    (
                        "two_island_pjrt_consumer_writer.cpp",
                        "noc",
                        _source(
                            "two_island_consumer_writer.cpp",
                            "TWO_ISLAND_PJRT_ARTIFACT",
                        ),
                    ),
                )
                kernel_paths = []
                for filename, thread_type, source in sources:
                    path = generated_dir / filename
                    path.write_text(source, encoding="utf-8")
                    kernel_paths.append((str(path), thread_type))

                import ttnn

                fp32_compute = ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                )
                bf16_compute = ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                )
                compiled.kernel_paths = kernel_paths
                compiled.kernel_configs = [
                    ttnn.ReaderConfigDescriptor(),
                    fp32_compute,
                    ttnn.WriterConfigDescriptor(),
                    ttnn.ReaderConfigDescriptor(),
                    bf16_compute,
                    ttnn.WriterConfigDescriptor(),
                ]
                compiled.kernel_tensor_indices = [
                    [0, 1],
                    [],
                    [],
                    [2],
                    [],
                    [3],
                ]
                compiled.kernel_core_ranges = [
                    _MATMUL_RANGE,
                    _MATMUL_RANGE,
                    _MATMUL_RANGE,
                    _ACTIVATION_RANGE,
                    _ACTIVATION_RANGE,
                    _ACTIVATION_RANGE,
                ]
                compiled.kernel_noc_roles = [
                    "reader",
                    None,
                    "writer",
                    "reader",
                    None,
                    "writer",
                ]
                compiled.cb_core_ranges = [
                    _ALL_WORKERS_RANGE,
                    _ALL_WORKERS_RANGE,
                    _MATMUL_RANGE,
                    _MATMUL_RANGE,
                    _MATMUL_RANGE,
                    _ACTIVATION_RANGE,
                    _ACTIVATION_RANGE,
                ]
                compiled.semaphore_configs = [
                    {
                        "id": semaphore_id,
                        "core_type": "worker",
                        "core_range": _ALL_WORKERS_RANGE,
                        "initial_value": 0,
                    }
                    for semaphore_id in range(2)
                ]
                compiled.num_pipe_sync_semaphores = 2
                compiled.kernel_arg_specs = [[] for _ in range(6)]
            break
        return result

    return operation
