"""Direct TT-Metal proof for mixed numerical compute islands in one program.

This deliberately sits below TT-Lang's current one-compute-kernel interop
restriction. It is an isolated silicon harness, not yet an XLA custom op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


TILE_BYTES_BF16 = 32 * 32 * 2
GRID_COLS = 11
GRID_ROWS = 10
MATMUL_CORES = 100
ACTIVATION_CORES = 10


def _core_ranges(ttnn: Any) -> tuple[Any, Any, Any]:
    matmul = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
        ]
    )
    activation = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 9))]
    )
    all_workers = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))]
    )
    return matmul, activation, all_workers


def _kernel_path(name: str) -> str:
    return str(Path(__file__).with_name("cpp") / name)


def _kernel(
    ttnn: Any,
    name: str,
    core_ranges: Any,
    config: Any,
    *,
    compile_time_args: list[int] | None = None,
    common_runtime_args: list[int] | None = None,
    defines: list[tuple[str, str]] | None = None,
) -> Any:
    return ttnn.KernelDescriptor(
        kernel_source=_kernel_path(name),
        core_ranges=core_ranges,
        compile_time_args=compile_time_args or [],
        common_runtime_args=common_runtime_args or [],
        defines=defines or [],
        config=config,
    )


def build_two_island_program(
    input_tensor: Any,
    rhs_tensor: Any,
    output_tensor: Any,
    *,
    diagnostic_copy_handoff: bool = False,
    producer_matmul_weight: Any | None = None,
) -> Any:
    """Build one 100-core FP32 + 10-core BF16 TT-Metal program."""

    import ttnn

    matmul, activation, all_workers = _core_ranges(ttnn)
    input_accessor = list(
        ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    )
    producer_matmul = producer_matmul_weight is not None
    producer_accessor = list(input_accessor)
    producer_addresses = [input_tensor.buffer_address()]
    if producer_matmul:
        producer_accessor.extend(
            ttnn.TensorAccessorArgs(
                producer_matmul_weight
            ).get_compile_time_args()
        )
        producer_addresses.append(producer_matmul_weight.buffer_address())
    rhs_accessor = list(
        ttnn.TensorAccessorArgs(rhs_tensor).get_compile_time_args()
    )
    output_accessor = list(
        ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    )
    fp32_compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    producer_defines = (
        [("TWO_ISLAND_REAL_MATMUL", "1")] if producer_matmul else []
    )
    bf16_compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
    )
    kernels = [
        _kernel(
            ttnn,
            "two_island_producer_reader.cpp",
            matmul,
            ttnn.ReaderConfigDescriptor(),
            compile_time_args=producer_accessor,
            common_runtime_args=producer_addresses,
            defines=producer_defines,
        ),
        _kernel(
            ttnn,
            "two_island_producer_compute.cpp",
            matmul,
            fp32_compute,
            defines=producer_defines,
        ),
        _kernel(
            ttnn,
            "two_island_producer_writer.cpp",
            matmul,
            ttnn.WriterConfigDescriptor(),
            defines=producer_defines,
        ),
        _kernel(
            ttnn,
            "two_island_consumer_reader.cpp",
            activation,
            ttnn.ReaderConfigDescriptor(),
            compile_time_args=rhs_accessor,
            common_runtime_args=[rhs_tensor.buffer_address()],
        ),
        _kernel(
            ttnn,
            "two_island_consumer_compute.cpp",
            activation,
            bf16_compute,
            defines=(
                [("TWO_ISLAND_COPY_HANDOFF", "1")]
                if diagnostic_copy_handoff
                else []
            ),
        ),
        _kernel(
            ttnn,
            "two_island_consumer_writer.cpp",
            activation,
            ttnn.WriterConfigDescriptor(),
            compile_time_args=output_accessor,
            common_runtime_args=[output_tensor.buffer_address()],
        ),
    ]
    bf16 = ttnn.bfloat16

    def cb(index: int, ranges: Any) -> Any:
        return ttnn.CBDescriptor(
            total_size=TILE_BYTES_BF16,
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index,
                    data_format=bf16,
                    page_size=TILE_BYTES_BF16,
                )
            ],
        )

    # CB0 spans both islands, which guarantees the same allocated L1 address
    # on producer and consumer. CB1 is intentionally separate on disjoint core
    # ranges even though the local index is reused.
    cbs = [
        cb(0, all_workers),
        cb(1, matmul),
        cb(2, matmul),
        cb(3, matmul),
        cb(4, matmul),
        cb(1, activation),
        cb(6, activation),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(
            semaphore_id,
            core_ranges=all_workers,
            initial_value=0,
        )
        for semaphore_id in range(2)
    ]
    return ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=semaphores,
        cbs=cbs,
    )


def run_two_island_program(
    input_tensor: Any,
    rhs_tensor: Any,
    output_tensor: Any,
    program: Any | None = None,
    *,
    producer_matmul_weight: Any | None = None,
) -> Any:
    """Execute the prototype and return the mutated output tensor."""

    import ttnn

    if program is None:
        program = build_two_island_program(
            input_tensor,
            rhs_tensor,
            output_tensor,
            producer_matmul_weight=producer_matmul_weight,
        )
    tensors = [input_tensor]
    if producer_matmul_weight is not None:
        tensors.append(producer_matmul_weight)
    tensors.extend((rhs_tensor, output_tensor))
    return ttnn.generic_op(tensors, program)
