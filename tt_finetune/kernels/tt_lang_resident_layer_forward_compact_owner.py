"""Exact compact-program Transformer-layer forward owner.

One public custom call flattens the accepted five-program attention block and
the extracted resident MLP continuation.  The post-attention normalized row
is compiler-owned L1 state; it is never a PJRT value or a DRAM round trip.
"""

from __future__ import annotations

from functools import lru_cache
import copy
import os
import re
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import (
    compile_serialized_child,
    compile_serialized_sequence_children,
    make_serialized_page_copy_bitwise,
    make_serialized_strided_page_copy_bitwise,
    remap_serialized_child,
    retarget_tensor_reads_to_packed_pages,
    retarget_tensor_writes_to_packed_pages,
    serialized_child_tensor_ids,
)
from .tt_lang_exact_residual_add import build_tt_lang_exact_residual_add
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID
from .tt_lang_native_swiglu_down import build_tt_lang_native_swiglu_down
from .tt_lang_native_paired_gate_up import build_tt_lang_native_paired_gate_up
from .tt_lang_owner_aligned_copy import build_tt_lang_owner_aligned_copy
from .tt_lang_resident_attention_block_forward_owner import (
    build_tt_lang_resident_attention_block_forward_owner,
)
from .tt_lang_resident_layer_forward_trainable_owner import (
    build_tt_lang_resident_layer_forward_trainable_owner,
)
from .tt_lang_resident_grouped_gate_up_forward import (
    build_tt_lang_resident_grouped_gate_up_forward,
)
from .tt_lang_grouped2_logical_forward import (
    build_tt_lang_grouped2_logical_forward,
)
from .tt_lang_resident_swiglu_down import (
    build_tt_lang_resident_swiglu_down_forward,
)
from .model_workloads import EXACT_BASE_K_PHASES
from .tt_lang_semaphore_scrub import build_tt_lang_semaphore_scrub


ttl = None
tt_lang_operation = None


def _remap_used(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


def _make_pure_semaphore_scrub(program: dict) -> dict:
    """Clear mailboxes without inheriting a preceding child's CB epoch."""

    result = copy.deepcopy(program)
    readers = [
        kernel for kernel in result.get("kernels", [])
        if kernel.get("thread_type") == "noc"
        and "clear every" in kernel.get("cpp_source", "")
    ]
    if len(readers) != 1:
        raise ValueError("compact-layer semaphore scrub ABI changed")
    reader = readers[0]
    reader["tensor_indices"] = []
    reader["cpp_source"] = r'''#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    for (uint32_t semaphore = 0; semaphore < 16; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }
}
'''
    result["kernels"] = [reader]
    result["cb_configs"] = []
    result.pop("operand_indices", None)
    return result


def _make_all_worker_barrier(program: dict) -> dict:
    """Turn the scrub scaffold into a 110-worker device rendezvous."""

    result = copy.deepcopy(program)
    readers = [
        kernel for kernel in result.get("kernels", [])
        if kernel.get("thread_type") == "noc"
    ]
    if len(readers) != 2:
        raise ValueError("all-worker barrier scaffold topology changed")
    reader = next(
        kernel for kernel in readers
        if "async_read(" in kernel.get("cpp_source", "")
    )
    reader["tensor_indices"] = []
    reader["cpp_source"] = r'''#include <cstdint>
#include "api/dataflow/dataflow_api.h"
namespace {
constexpr uint32_t kWorkers = 110;
constexpr uint32_t kColumns = 11;
constexpr uint8_t kPhysicalX[11] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
FORCE_INLINE uint64_t worker_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(
        kPhysicalX[node % kColumns],
        static_cast<uint8_t>(node / kColumns + 2), address);
}
}
void kernel_main() {
    const uint32_t node =
        get_absolute_logical_y() * kColumns + get_absolute_logical_x();
    auto* arrivals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(0));
    auto* release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(1));
    auto* init_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(2));
    auto* init_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(3));
    noc_semaphore_set(arrivals, 0);
    noc_semaphore_set(release, 0);
    noc_semaphore_set(init_ready, 0);
    noc_semaphore_inc(worker_l1(0, get_semaphore(2)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        noc_semaphore_wait_min(init_count, kWorkers);
        noc_semaphore_set(init_count, 0);
        noc_semaphore_set(init_ready, 1);
        for (uint32_t target = 1; target < kWorkers; ++target)
            noc_async_write(
                get_semaphore(3), worker_l1(target, get_semaphore(3)),
                sizeof(uint32_t));
        noc_async_write_barrier();
    }
    noc_semaphore_wait(init_ready, 1);
    noc_semaphore_inc(worker_l1(0, get_semaphore(0)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        noc_semaphore_wait_min(arrivals, kWorkers);
        noc_semaphore_set(arrivals, 0);
        noc_semaphore_set(release, 1);
        for (uint32_t target = 1; target < kWorkers; ++target)
            noc_async_write(
                get_semaphore(1), worker_l1(target, get_semaphore(1)),
                sizeof(uint32_t));
        noc_async_write_barrier();
    }
    noc_semaphore_wait(release, 1);
}
'''
    result["kernels"] = [reader]
    result["cb_configs"] = []
    result.pop("operand_indices", None)
    return result


def _retarget_grouped_mlp_writes_to_arena(
    program: dict,
    *,
    arena_tensor: int,
    gate_page_offset: int,
    up_page_offset: int,
    gate_h_page_offset: int,
    up_h_page_offset: int,
) -> dict:
    """Publish grouped gate/up/H directly to their packed arena regions."""

    result = copy.deepcopy(program)
    writers = [
        kernel for kernel in result.get("kernels", [])
        if kernel.get("thread_type") == "noc"
        and "Final/checkpoint writer" in kernel.get("cpp_source", "")
    ]
    if len(writers) != 1:
        raise ValueError("resident grouped MLP writer ABI changed")
    writer = writers[0]
    source = writer["cpp_source"]
    marker = "    constexpr uint32_t cb_output = 12;\n"
    constants = (
        marker
        + f"    constexpr uint32_t gate_page_offset = {gate_page_offset};\n"
        + f"    constexpr uint32_t up_page_offset = {up_page_offset};\n"
        + f"    constexpr uint32_t gate_h_page_offset = {gate_h_page_offset};\n"
        + f"    constexpr uint32_t up_h_page_offset = {up_h_page_offset};\n"
    )
    if source.count(marker) != 1:
        raise ValueError("resident grouped MLP writer constants changed")
    source = source.replace(marker, constants, 1)
    h_page = ".page_id = row * arena_row_tiles + total_n_tiles + member"
    if source.count(h_page) != 1:
        raise ValueError("resident grouped MLP H publication changed")
    source = source.replace(
        h_page,
        ".page_id = (member ? up_h_page_offset : gate_h_page_offset) + row",
        1,
    )
    output_page = ".page_id = row * arena_row_tiles + n"
    if source.count(output_page) != 1:
        raise ValueError("resident grouped MLP output publication changed")
    source = source.replace(
        output_page,
        ".page_id = (n < intermediate_tiles ? gate_page_offset : "
        "up_page_offset) + row * intermediate_tiles + "
        "(n % intermediate_tiles)",
        1,
    )
    writer["cpp_source"] = source
    used = serialized_child_tensor_ids(result)
    return remap_serialized_child(
        result,
        {tensor: arena_tensor if tensor == 6 else tensor for tensor in used},
    )


def _split_grouped_mlp_writes(
    program: dict,
    *,
    gate_page_offset: int,
    up_page_offset: int,
    gate_h_page_offset: int,
    up_h_page_offset: int,
) -> dict:
    """Write gate/up directly to two tensors and checkpoint the same pages."""

    result = copy.deepcopy(program)
    writer = next(
        (
            kernel for kernel in result.get("kernels", [])
            if kernel.get("thread_type") == "noc"
            and "Final/checkpoint writer" in kernel.get("cpp_source", "")
        ),
        None,
    )
    if writer is None or writer.get("tensor_indices") != [6]:
        raise ValueError("resident grouped split-writer ABI changed")
    source = writer["cpp_source"]
    marker = "    constexpr uint32_t cb_output = 12;\n"
    if source.count(marker) != 1:
        raise ValueError("resident grouped split-writer constants changed")
    source = source.replace(
        marker,
        marker
        + f"    constexpr uint32_t gate_page_offset = {gate_page_offset};\n"
        + f"    constexpr uint32_t up_page_offset = {up_page_offset};\n"
        + f"    constexpr uint32_t gate_h_page_offset = {gate_h_page_offset};\n"
        + f"    constexpr uint32_t up_h_page_offset = {up_h_page_offset};\n",
        1,
    )
    accessor = '''    const auto arena = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 15>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
'''
    if source.count(accessor) != 1:
        raise ValueError("resident grouped split-writer accessor changed")
    source = source.replace(
        accessor,
        '''    const auto gate = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 15>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto up = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<7, 15>(), 1>(),
        get_common_arg_val<uint32_t>(1), kTileBytes);
    const auto arena = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<8, 15>(), 2>(),
        get_common_arg_val<uint32_t>(2), kTileBytes);
''',
        1,
    )
    h_page = ".page_id = row * arena_row_tiles + total_n_tiles + member"
    if source.count(h_page) != 1:
        raise ValueError("resident grouped split-writer H page changed")
    source = source.replace(
        h_page,
        ".page_id = (member ? up_h_page_offset : gate_h_page_offset) + row",
        1,
    )
    output_write = '''        noc1.async_write(
            CoreLocalMem<uint32_t>(output.get_read_ptr()), arena,
            kTileBytes, {}, {.page_id = row * arena_row_tiles + n});
        noc1.async_write_barrier();
'''
    if source.count(output_write) != 1:
        raise ValueError("resident grouped split-writer output changed")
    source = source.replace(
        output_write,
        '''        const bool up_member = n >= intermediate_tiles;
        const uint32_t member_n = n % intermediate_tiles;
        if (up_member) {
            noc1.async_write(
                CoreLocalMem<uint32_t>(output.get_read_ptr()), up,
                kTileBytes, {},
                {.page_id = row * intermediate_tiles + member_n});
        } else {
            noc1.async_write(
                CoreLocalMem<uint32_t>(output.get_read_ptr()), gate,
                kTileBytes, {},
                {.page_id = row * intermediate_tiles + member_n});
        }
        noc1.async_write(
            CoreLocalMem<uint32_t>(output.get_read_ptr()), arena,
            kTileBytes, {},
            {.page_id = (up_member ? up_page_offset : gate_page_offset)
                + row * intermediate_tiles + member_n});
        noc1.async_write_barrier();
''',
        1,
    )
    writer["cpp_source"] = source
    writer["tensor_indices"] = [6, 7, 8]
    return result


def _fuse_grouped2_checkpoint_publication(
    program: dict,
    *,
    sequence_tiles: int,
    intermediate_tiles: int,
    rank_tiles: int,
    gate_page_offset: int,
    up_page_offset: int,
    gate_h_page_offset: int,
    up_h_page_offset: int,
) -> dict:
    """Duplicate grouped2 producer tiles directly into the packed checkpoint.

    The exact grouped2 program must continue publishing gate/up/H into its
    compiler-owned L1 arena because the resident SwiGLU/down child consumes
    that representation.  Previously four later physical programs copied the
    same bytes from that arena into the persistent forward checkpoint.  Give
    the existing writer a second destination instead: each tile is sent to
    both destinations before its producer CB is popped.  This changes only
    NoC destinations; the accepted compute and BF16 packing schedule is left
    untouched.
    """

    dimensions = (sequence_tiles, intermediate_tiles, rank_tiles)
    if any(value <= 0 for value in dimensions):
        raise ValueError("grouped2 publication dimensions must be positive")
    offsets = (
        gate_page_offset,
        up_page_offset,
        gate_h_page_offset,
        up_h_page_offset,
    )
    if any(value < 0 for value in offsets):
        raise ValueError("grouped2 publication offsets must be non-negative")

    result = copy.deepcopy(program)
    writer_pattern = re.compile(
        r"get_tensor_accessor_args_cta_offset<\s*6\s*,\s*16\s*>\(\)"
    )
    writers = [
        kernel
        for kernel in result.get("kernels", [])
        if kernel.get("thread_type") == "noc"
        and writer_pattern.search(kernel.get("cpp_source", ""))
        and "get_compile_time_arg_val(12)" in kernel.get("cpp_source", "")
        and "get_compile_time_arg_val(4)" in kernel.get("cpp_source", "")
        and "async_write(" in kernel.get("cpp_source", "")
    ]
    if len(writers) != 1:
        raise ValueError(
            "grouped2 checkpoint publication expected one qualified writer, "
            f"found {len(writers)}"
        )
    writer = writers[0]
    if [int(value) for value in writer.get("tensor_indices", [])] != [6]:
        raise ValueError("grouped2 writer tensor ABI changed")

    total_n_tiles = 2 * intermediate_tiles
    arena_row_tiles = total_n_tiles + 2 * rank_tiles
    total_jobs = sequence_tiles * total_n_tiles
    jobs_per_node = -(-total_jobs // 110)
    writer["cpp_source"] = f'''#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {{
    constexpr uint32_t kGridColumns = 11;
    constexpr uint32_t kCoreCount = 110;
    constexpr uint32_t kTileBytes = 2048;
    constexpr uint32_t kSequenceTiles = {sequence_tiles};
    constexpr uint32_t kIntermediateTiles = {intermediate_tiles};
    constexpr uint32_t kRankTiles = {rank_tiles};
    constexpr uint32_t kTotalNTiles = {total_n_tiles};
    constexpr uint32_t kArenaRowTiles = {arena_row_tiles};
    constexpr uint32_t kTotalJobs = {total_jobs};
    constexpr uint32_t kJobsPerNode = {jobs_per_node};
    constexpr uint32_t kGatePageOffset = {gate_page_offset};
    constexpr uint32_t kUpPageOffset = {up_page_offset};
    constexpr uint32_t kGateHPageOffset = {gate_h_page_offset};
    constexpr uint32_t kUpHPageOffset = {up_h_page_offset};

    CircularBuffer output(get_compile_time_arg_val(12));
    CircularBuffer saved_h(get_compile_time_arg_val(4));
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                6, 16>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto checkpoint = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                7, 16>(), 1>(),
        get_common_arg_val<uint32_t>(1), kTileBytes);
    Noc noc1(1);
    const uint32_t node =
        get_absolute_logical_y() * kGridColumns + get_absolute_logical_x();

    for (uint32_t local = 0; local < kJobsPerNode; ++local) {{
        const uint32_t job = node * kJobsPerNode + local;
        if (job >= kTotalJobs) continue;
        const uint32_t row = job / kTotalNTiles;
        const uint32_t n = job - row * kTotalNTiles;
        const uint32_t member = n / kIntermediateTiles;
        const uint32_t member_n = n - member * kIntermediateTiles;

        output.wait_front(1);
        const uint32_t output_address = output.get_read_ptr();
        noc1.async_write(
            CoreLocalMem<uint32_t>(output_address), arena, kTileBytes, {{}},
            {{.page_id = row * kArenaRowTiles + n}});
        noc1.async_write_barrier();
        noc1.async_write(
            CoreLocalMem<uint32_t>(output_address), checkpoint, kTileBytes, {{}},
            {{.page_id = (member ? kUpPageOffset : kGatePageOffset)
                + row * kIntermediateTiles + member_n}});
        noc1.async_write_barrier();
        output.pop_front(1);

        if (member_n == 0) {{
            saved_h.wait_front(kRankTiles);
            for (uint32_t rank = 0; rank < kRankTiles; ++rank) {{
                const uint32_t h_address =
                    saved_h.get_read_ptr() + rank * kTileBytes;
                noc1.async_write(
                    CoreLocalMem<uint32_t>(h_address), arena, kTileBytes, {{}},
                    {{.page_id = row * kArenaRowTiles + kTotalNTiles
                        + member * kRankTiles + rank}});
                noc1.async_write_barrier();
                noc1.async_write(
                    CoreLocalMem<uint32_t>(h_address), checkpoint, kTileBytes, {{}},
                    {{.page_id = (member ? kUpHPageOffset : kGateHPageOffset)
                        + row * kRankTiles + rank}});
                noc1.async_write_barrier();
            }}
            saved_h.pop_front(kRankTiles);
        }}
    }}
}}
'''
    writer["tensor_indices"] = [6, 7]
    result["num_tensors"] = max(int(result.get("num_tensors", 0)), 8)
    return result


def _fuse_residual_add_publication(
    program: dict,
    *,
    row_tiles: int,
    column_tiles: int,
    destination_page_offsets: tuple[int, ...],
    output_tensor_index: int = 2,
    checkpoint_tensor_index: int = 3,
) -> dict:
    """Publish an exact residual result to L1 and packed DRAM in one writer."""

    if row_tiles <= 0 or column_tiles <= 0:
        raise ValueError("residual publication dimensions must be positive")
    offsets = tuple(dict.fromkeys(int(value) for value in destination_page_offsets))
    if not offsets or any(value < 0 for value in offsets):
        raise ValueError("residual publication requires non-negative destinations")

    result = copy.deepcopy(program)
    writer_pattern = re.compile(
        rf"get_tensor_accessor_args_cta_offset<\s*"
        rf"{output_tensor_index}\s*,\s*3\s*>\(\)"
    )
    writers = [
        kernel
        for kernel in result.get("kernels", [])
        if kernel.get("thread_type") == "noc"
        and writer_pattern.search(kernel.get("cpp_source", ""))
        and "get_compile_time_arg_val(2)" in kernel.get("cpp_source", "")
        and "async_write(" in kernel.get("cpp_source", "")
    ]
    if len(writers) != 1:
        raise ValueError(
            "residual publication expected one qualified writer, "
            f"found {len(writers)}"
        )
    writer = writers[0]
    if [int(value) for value in writer.get("tensor_indices", [])] != [
        output_tensor_index
    ]:
        raise ValueError("residual writer tensor ABI changed")

    total_tiles = row_tiles * column_tiles
    publication_writes = "\n".join(
        f'''        noc1.async_write(
            CoreLocalMem<uint32_t>(address), checkpoint, kTileBytes, {{}},
            {{.page_id = {offset} + tile_id}});
        noc1.async_write_barrier();'''
        for offset in offsets
    )
    writer["cpp_source"] = f'''#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {{
    constexpr uint32_t kGridColumns = 11;
    constexpr uint32_t kTileBytes = 2048;
    constexpr uint32_t kColumnTiles = {column_tiles};
    constexpr uint32_t kTotalTiles = {total_tiles};
    CircularBuffer result(get_compile_time_arg_val(2));
    const auto local_output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                {output_tensor_index}, 3>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto checkpoint = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                {checkpoint_tensor_index}, 3>(), 1>(),
        get_common_arg_val<uint32_t>(1), kTileBytes);
    Noc noc1(1);
    const uint32_t node =
        get_absolute_logical_y() * kGridColumns + get_absolute_logical_x();
    for (uint32_t column = 0; column < kColumnTiles; ++column) {{
        const uint32_t tile_id = node * kColumnTiles + column;
        if (tile_id >= kTotalTiles) continue;
        result.wait_front(1);
        const uint32_t address = result.get_read_ptr();
        noc1.async_write(
            CoreLocalMem<uint32_t>(address), local_output, kTileBytes, {{}},
            {{.page_id = tile_id}});
        noc1.async_write_barrier();
{publication_writes}
        result.pop_front(1);
    }}
}}
'''
    writer["tensor_indices"] = [
        output_tensor_index, checkpoint_tensor_index,
    ]
    result["num_tensors"] = max(int(result.get("num_tensors", 0)), 4)
    return result


def _fuse_rmsnorm_checkpoint_publication(
    program: dict,
    *,
    row_tiles: int,
    column_tiles: int,
    destination_page_offset: int,
    output_tensor_index: int,
    checkpoint_tensor_index: int,
) -> dict:
    """Duplicate exact RMSNorm writer tiles into the backward checkpoint."""

    if row_tiles <= 0 or column_tiles <= 0:
        raise ValueError("RMSNorm publication dimensions must be positive")
    if destination_page_offset < 0:
        raise ValueError("RMSNorm publication destination must be non-negative")
    result = copy.deepcopy(program)
    output_pattern = re.compile(
        rf"get_tensor_accessor_args_cta_offset<\s*"
        rf"{output_tensor_index}\s*,\s*total_cbs\s*>\(\)"
    )
    writers = [
        kernel
        for kernel in result.get("kernels", [])
        if kernel.get("thread_type") == "noc"
        and output_pattern.search(kernel.get("cpp_source", ""))
        and "Final BF16 writer for the exact resident RMSNorm" in kernel.get(
            "cpp_source", ""
        )
    ]
    if len(writers) != 1:
        raise ValueError(
            "RMSNorm publication expected one qualified writer, "
            f"found {len(writers)}"
        )
    writer = writers[0]
    if [int(value) for value in writer.get("tensor_indices", [])] != [
        output_tensor_index
    ]:
        raise ValueError("RMSNorm writer tensor ABI changed")
    source = writer.get("cpp_source", "")
    cb_match = re.search(r"cb_output\s*=\s*(\d+)", source)
    block_match = re.search(r"published_block_tiles\s*=\s*(\d+)", source)
    total_cbs_match = re.search(r"total_cbs\s*=\s*(\d+)", source)
    if not (cb_match and block_match and total_cbs_match):
        raise ValueError("RMSNorm writer schedule metadata changed")
    if "replicate_scalar = false" not in source:
        raise ValueError("RMSNorm publication only supports full-width output")
    cb_output = int(cb_match.group(1))
    published_block_tiles = int(block_match.group(1))
    total_cbs = int(total_cbs_match.group(1))
    writer["cpp_source"] = f'''// Final BF16 writer plus exact packed checkpoint publication.
#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {{
    constexpr uint32_t cb_output = {cb_output};
    constexpr uint32_t output_tiles = {column_tiles};
    constexpr uint32_t published_block_tiles = {published_block_tiles};
    constexpr uint32_t token_tiles = {row_tiles};
    constexpr uint32_t total_cbs = {total_cbs};
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t checkpoint_page_offset = {destination_page_offset};
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= token_tiles) return;

    const auto output_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                {output_tensor_index}, total_cbs>(), 0>(),
        get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto checkpoint = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                {checkpoint_tensor_index}, total_cbs>(), 1>(),
        get_common_arg_val<uint32_t>(1), tile_bytes);
    Noc noc1(1);
    CircularBuffer output(cb_output);
    uint32_t page = node * output_tiles;
    for (uint32_t start = 0; start < output_tiles;
         start += published_block_tiles) {{
        const uint32_t valid =
            output_tiles - start < published_block_tiles
                ? output_tiles - start
                : published_block_tiles;
        output.wait_front(published_block_tiles);
        for (uint32_t tile = 0; tile < valid; ++tile) {{
            const uint32_t address =
                output.get_read_ptr() + tile * tile_bytes;
            noc1.async_write(
                CoreLocalMem<uint32_t>(address), output_tensor, tile_bytes,
                {{}}, {{.page_id = page}});
            noc1.async_write(
                CoreLocalMem<uint32_t>(address), checkpoint, tile_bytes,
                {{}}, {{.page_id = checkpoint_page_offset + page}});
            ++page;
        }}
        noc1.async_write_barrier();
        output.pop_front(published_block_tiles);
    }}
}}
'''
    writer["tensor_indices"] = [
        output_tensor_index, checkpoint_tensor_index,
    ]
    result["num_tensors"] = int(result.get("num_tensors", 0)) + 1
    return result


def _offset_hardcoded_cbs(program: dict, *, offset: int) -> dict:
    """Move a custom resident child into a disjoint physical CB epoch."""

    result = copy.deepcopy(program)
    for config in result.get("cb_configs", []):
        old = int(config["buffer_index"])
        if old + offset >= 32:
            raise ValueError("CB offset exceeds the Tensix CB table")
        config["buffer_index"] = old + offset
    declaration = re.compile(r"(\bcb_[A-Za-z0-9_]+\s*=\s*)(\d+)(\s*;)")
    total_rewritten = 0
    for kernel in result.get("kernels", []):
        source = kernel.get("cpp_source", "")
        source, count = declaration.subn(
            lambda match: (
                match.group(1)
                + str(int(match.group(2)) + offset)
                + match.group(3)
            ),
            source,
        )
        total_rewritten += count
        kernel["cpp_source"] = source
    if total_rewritten == 0:
        raise ValueError("resident grouped child has no hardcoded CB IDs")
    return result


def _placeholder_like(tensor, shape: tuple[int, int]):
    """Construct a shape-only compile operand without allocating a tensor."""

    try:
        return type(tensor)(
            shape=shape,
            dtype=tensor.dtype,
            layout=tensor.layout,
            memory_config=tensor.memory_config(),
            device=tensor.device(),
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "compact layer requires shape-only PJRT compile operands"
        ) from exc


@lru_cache(maxsize=8)
def build_tt_lang_resident_layer_forward_compact_owner(
    *,
    sequence_length: int,
    hidden_features: int,
    intermediate_features: int,
    kv_features: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_rank: int,
    lora_scale: float,
    attention_scale: float,
    has_bias: bool,
    active_lora_roles: tuple[str, ...] = (
        "q", "k", "v", "o", "gate", "up", "down",
    ),
    input_rmsnorm_epsilon: float,
    post_attention_rmsnorm_epsilon: float,
    hidden_page_base: int = 0,
    publish_layer_output_at_page_zero: bool = True,
    debug_external_mlp_residual: bool = False,
    o_lora_k_tiles_per_phase: int | None = None,
    gate_up_base_k_tiles_per_phase: int | None = None,
    canonical_native_o_path: bool = False,
    exact_attention_tail: bool = False,
    canonical_bf16_attention: bool = False,
    canonical_grouped_qkv_rank: bool = False,
    canonical_separate_qkv_rank: bool = False,
    logical_rank: int | None = None,
) -> Callable:
    """Build the compact exact forward owner for one Transformer layer."""

    if input_rmsnorm_epsilon != post_attention_rmsnorm_epsilon:
        raise ValueError("compact layer currently requires equal RMS epsilons")
    canonical_roles = {"q", "k", "v", "o", "gate", "up", "down"}
    active_lora_roles = tuple(dict.fromkeys(active_lora_roles))
    if not active_lora_roles or not set(active_lora_roles) <= canonical_roles:
        raise ValueError("active LoRA roles must be a non-empty canonical subset")
    attention = build_tt_lang_resident_attention_block_forward_owner(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        rmsnorm_epsilon=input_rmsnorm_epsilon,
        has_bias=has_bias,
        active_lora_roles=active_lora_roles,
        grouped_qkv_lora=bool(exact_attention_tail and has_bias),
        o_lora_k_tiles_per_phase=o_lora_k_tiles_per_phase,
        canonical_native_o_path=canonical_native_o_path,
        logical_rank=logical_rank,
        exact_attention_tail=exact_attention_tail,
        canonical_bf16_attention=canonical_bf16_attention,
        canonical_grouped_qkv_rank=canonical_grouped_qkv_rank,
        canonical_separate_qkv_rank=canonical_separate_qkv_rank,
    )
    monolithic = build_tt_lang_resident_layer_forward_trainable_owner(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        has_bias=has_bias,
        input_rmsnorm_epsilon=input_rmsnorm_epsilon,
        post_attention_rmsnorm_epsilon=post_attention_rmsnorm_epsilon,
        hidden_page_base=hidden_page_base,
        publish_layer_output_at_page_zero=publish_layer_output_at_page_zero,
        o_lora_k_tiles_per_phase=o_lora_k_tiles_per_phase,
    )
    checkpoint_copy = build_tt_lang_owner_aligned_copy(
        rows=sequence_length, columns=hidden_features
    )
    semaphore_scrub = build_tt_lang_semaphore_scrub()
    canonical_mlp = bool(exact_attention_tail)
    # Rank-scaled owners keep the same exact programs but use the warm DRAM
    # tier for transient tensors once the complete attention+MLP allocation
    # would overlap static circular buffers in a full autograd graph.
    spill_rank_scaled_intermediates = bool(
        canonical_mlp
        and (
            (logical_rank is not None and logical_rank >= 32)
            or os.environ.get("TT_FINETUNE_TRACE_SAFE_SPILL", "0") == "1"
        )
    )

    def planned_internal_spec(spec: dict[str, object]) -> dict[str, object]:
        planned = dict(spec)
        force_l1 = bool(planned.pop("force_l1", False))
        if force_l1:
            planned["memory"] = "l1"
        elif spill_rank_scaled_intermediates:
            planned["memory"] = "clone"
        return planned
    # Attention arithmetic fidelity and the MLP schedule are independent.
    # Rank-scaled owners use the canonical BF16 attention core, but must keep
    # the already-qualified grouped2 + decomposed-SwiGLU MLP path as well.
    production_exact_mlp = bool(canonical_mlp)
    precomputed_mlp_rank_delta = bool(
        production_exact_mlp and physical_rank > 32
    )
    production_native_gate_up = bool(
        production_exact_mlp and physical_rank > 64
    )
    production_ttnn_down = bool(
        production_exact_mlp
        and (
            physical_rank > 32
            or os.environ.get("TT_FINETUNE_TTNN_DOWN", "0") == "1"
        )
    )
    canonical_mlp_internal_specs: tuple[dict[str, object], ...] = ()
    debug_publish_down = os.environ.get(
        "TT_FINETUNE_DEBUG_PUBLISH_LAYER_DOWN", "0"
    ) == "1"
    debug_mlp_stage = os.environ.get("TT_FINETUNE_DEBUG_LAYER_MLP_STAGE", "")
    # These publication fusions are part of the exact resident schedule, not
    # optional numerical kernels.  They only retarget the already-packed BF16
    # writer tiles to their persistent checkpoint destinations before the
    # producer CB is popped.  The isolated 20-step evolving-state authority
    # proves equality for every loss, gradient, trainable and AdamW state.
    # Keep a disable value for matched reference/diagnostic runs, but compile
    # the non-materializing schedule by default whenever the exact owner is
    # selected.
    fused_mlp_checkpoint_publication = bool(
        production_exact_mlp
        and os.environ.get(
            "TT_FINETUNE_FUSED_MLP_CHECKPOINT_PUBLICATION", "1"
        ) != "0"
    )
    fused_mlp_output_publication = bool(
        production_exact_mlp
        and os.environ.get(
            "TT_FINETUNE_FUSED_MLP_OUTPUT_PUBLICATION", "1"
        ) != "0"
    )
    fused_attention_checkpoint_publication = bool(
        production_exact_mlp
        and os.environ.get(
            "TT_FINETUNE_FUSED_ATTENTION_CHECKPOINT_PUBLICATION", "1"
        ) != "0"
    )
    output_publication_programs = 1 + int(
        publish_layer_output_at_page_zero
        and monolithic.layer_output_page_offset != 0
    )
    if canonical_mlp:
        if False and production_native_gate_up:
            hidden_tiles = hidden_features // 32
            intermediate_tiles = intermediate_features // 32
            sequence_tiles = sequence_length // 32
            rank_tiles = physical_rank // 32
            mlp_private_base = 32 + attention.compiler_owned_l1_tensors
            gate_weight_id = mlp_private_base
            up_weight_id = mlp_private_base + 1
            gate_id = mlp_private_base + 2
            up_id = mlp_private_base + 3
            pair_internal_base = mlp_private_base + 4
            down_weight_id = pair_internal_base + grouped_mlp.compiler_owned_l1_tensors
            down_id = down_weight_id + 1
            down_internal_base = down_id + 1
            layer_output_id = down_internal_base + down_mlp.compiler_owned_l1_tensors

            grouped_weight_offset = hidden_tiles + 16
            grouped_weight_pages = hidden_tiles * 2 * intermediate_tiles
            down_weight_offset = (
                grouped_weight_offset
                + grouped_weight_pages
                + 2 * rank_tiles * hidden_tiles
                + 2 * rank_tiles * intermediate_tiles
            )
            grouped_weight = _placeholder_like(
                t00, (hidden_features, 2 * intermediate_features)
            )
            projection_weight = _placeholder_like(
                t00, (hidden_features, intermediate_features)
            )
            for section_offset, private_id in (
                (0, gate_weight_id),
                (intermediate_tiles, up_weight_id),
            ):
                weight_copy = build_tt_lang_owner_aligned_copy(
                    rows=hidden_features, columns=intermediate_features
                )
                weight_child = compile_serialized_child(
                    weight_copy, (grouped_weight, projection_weight)
                )
                weight_child = make_serialized_strided_page_copy_bitwise(
                    weight_child,
                    rows_tiles=hidden_tiles,
                    section_column_tiles=intermediate_tiles,
                    source_row_tiles=2 * intermediate_tiles,
                    source_section_offset=section_offset,
                    source_page_offset=grouped_weight_offset,
                )
                children.append(_remap_used(
                    weight_child, {0: 20, 1: private_id}
                ))

            gate_seed = _placeholder_like(
                t00, (sequence_length, intermediate_features)
            )
            up_seed = _placeholder_like(
                t00, (sequence_length, intermediate_features)
            )
            pair_children = compile_serialized_sequence_children(
                grouped_mlp,
                (
                    t00, projection_weight, t21, t22,
                    projection_weight, t23, t24,
                    gate_seed, up_seed,
                ),
            )
            pair_map = {
                0: 31,
                1: gate_weight_id,
                2: 21,
                3: 22,
                4: up_weight_id,
                5: 23,
                6: 24,
                7: gate_id,
                8: up_id,
                **{
                    9 + index: pair_internal_base + index
                    for index in range(grouped_mlp.compiler_owned_l1_tensors)
                },
            }
            children.extend(
                _remap_used(child, pair_map) for child in pair_children
            )

            for source_id, columns, destination_offset in (
                (gate_id, intermediate_features, monolithic.mlp_gate_page_offset),
                (up_id, intermediate_features, monolithic.mlp_up_page_offset),
                (pair_internal_base, physical_rank, monolithic.mlp_gate_h_page_offset),
                (pair_internal_base + 6, physical_rank, monolithic.mlp_up_h_page_offset),
            ):
                checkpoint = build_tt_lang_owner_aligned_copy(
                    rows=sequence_length, columns=columns
                )
                checkpoint_seed = _placeholder_like(
                    t00, (sequence_length, columns)
                )
                checkpoint_child = compile_serialized_child(
                    checkpoint, (checkpoint_seed, t30)
                )
                checkpoint_child = make_serialized_page_copy_bitwise(
                    checkpoint_child,
                    total_pages=sequence_tiles * (columns // 32),
                    column_tiles=columns // 32,
                    source_page_offset=0,
                )
                checkpoint_child = retarget_tensor_writes_to_packed_pages(
                    checkpoint_child,
                    packed_tensor=1,
                    tensor_page_offsets={1: destination_offset},
                )
                children.append(_remap_used(
                    checkpoint_child, {0: source_id, 1: 30}
                ))

            down_weight = _placeholder_like(
                t00, (intermediate_features, hidden_features)
            )
            down_weight_destination = _placeholder_like(
                t00, (intermediate_features, hidden_features)
            )
            down_weight_copy = build_tt_lang_owner_aligned_copy(
                rows=intermediate_features, columns=hidden_features
            )
            down_weight_child = compile_serialized_child(
                down_weight_copy, (down_weight, down_weight_destination)
            )
            down_weight_child = make_serialized_page_copy_bitwise(
                down_weight_child,
                total_pages=intermediate_tiles * hidden_tiles,
                column_tiles=hidden_tiles,
                source_page_offset=down_weight_offset,
            )
            children.append(_remap_used(
                down_weight_child, {0: 20, 1: down_weight_id}
            ))

            down_seed = _placeholder_like(
                t00, (sequence_length, hidden_features)
            )
            down_children = compile_serialized_sequence_children(
                down_mlp,
                (gate_seed, up_seed, down_weight, t25, t26, down_seed),
            )
            down_map = {
                0: gate_id, 1: up_id, 2: down_weight_id,
                3: 25, 4: 26, 5: down_id,
                **{
                    6 + index: down_internal_base + index
                    for index in range(down_mlp.compiler_owned_l1_tensors)
                },
            }
            children.extend(
                _remap_used(child, down_map) for child in down_children
            )

            layer_output = _placeholder_like(
                t00, (sequence_length, hidden_features)
            )
            add_child = compile_serialized_child(
                mlp_add, (down_seed, t00, layer_output)
            )
            children.append(_remap_used(add_child, {
                0: down_id,
                1: 29 if debug_external_mlp_residual else 32,
                2: layer_output_id,
            }))
            publication_offsets = [monolithic.layer_output_page_offset]
            if publish_layer_output_at_page_zero and 0 not in publication_offsets:
                publication_offsets.append(0)
            for destination_offset in publication_offsets:
                publish = compile_serialized_child(
                    checkpoint_copy, (layer_output, t30)
                )
                publish = retarget_tensor_writes_to_packed_pages(
                    publish,
                    packed_tensor=1,
                    tensor_page_offsets={1: destination_offset},
                )
                children.append(_remap_used(
                    publish, {0: layer_output_id, 1: 30}
                ))
        elif production_exact_mlp:
            grouped_mlp = (
                build_tt_lang_native_paired_gate_up(
                    scale=lora_scale,
                    tokens=sequence_length,
                    hidden_features=hidden_features,
                    intermediate_features=intermediate_features,
                    physical_rank=physical_rank,
                    logical_rank=logical_rank,
                )
                if production_native_gate_up else
                build_tt_lang_grouped2_logical_forward(
                scale=lora_scale,
                tokens=sequence_length,
                input_features=hidden_features,
                member_output_features=intermediate_features,
                physical_rank=physical_rank,
                base_k_tiles_per_phase=(
                    EXACT_BASE_K_PHASES[
                        (hidden_features, intermediate_features)
                    ]
                    if gate_up_base_k_tiles_per_phase is None
                    else int(gate_up_base_k_tiles_per_phase)
                ),
                save_input_checkpoint=False,
                precomputed_rank_delta=precomputed_mlp_rank_delta,
                )
            )
            if production_ttnn_down:
                down_mlp = build_tt_lang_native_swiglu_down(
                    scale=lora_scale,
                    tokens=sequence_length,
                    hidden_features=hidden_features,
                    intermediate_features=intermediate_features,
                    physical_rank=physical_rank,
                    logical_rank=logical_rank,
                )
            else:
                down_mlp = build_tt_lang_resident_swiglu_down_forward(
                    scale=lora_scale,
                    tokens=sequence_length,
                    hidden_features=hidden_features,
                    intermediate_features=intermediate_features,
                    physical_rank=physical_rank,
                    base_k_tiles_per_phase=(
                        int(os.environ["TT_FINETUNE_DOWN_BASE_K_PHASE"])
                        if "TT_FINETUNE_DOWN_BASE_K_PHASE" in os.environ
                        else None
                    ),
                    rank_k_tiles_per_phase=int(os.environ.get(
                        "TT_FINETUNE_DOWN_RANK_K_PHASE", "2"
                    )),
                    allow_unvalidated_base_phase=(
                        "TT_FINETUNE_DOWN_BASE_K_PHASE" in os.environ
                    ),
                    fp32_activation=False,
                    # Preserve the frozen resident runtime numerical contract at the
                    # R8/R16 golden fingerprints.  Those owners were
                    # qualified against the framework's decomposed BF16
                    # gate*sigmoid boundary.  Rank-scaled R32+ graphs use the
                    # newer single-SiLU boundary because their generic oracle
                    # changes rounding there.  Applying the R32 contract to
                    # the frozen ranks changes the first layer's down output
                    # even though every attention and gate/up boundary stays
                    # exact.
                    decomposed_activation=bool(
                        logical_rank is None or logical_rank < 32
                    ),
                    save_rank_hidden=True,
                    packed_gate_up_arena=True,
                    packed_gate_up_row_tiles=(
                        2 * (intermediate_features // 32)
                        + 2 * (physical_rank // 32)
                    ),
                )
        else:
            grouped_mlp = build_tt_lang_resident_grouped_gate_up_forward(
                scale=lora_scale,
                tokens=sequence_length,
                hidden_features=hidden_features,
                intermediate_features=intermediate_features,
                base_k_tiles_per_phase=2,
                worker_count=110,
            )
            down_mlp = build_tt_lang_native_swiglu_down(
                scale=lora_scale,
                tokens=sequence_length,
                hidden_features=hidden_features,
                intermediate_features=intermediate_features,
                physical_rank=physical_rank,
            )
        mlp_add = build_tt_lang_exact_residual_add(
            rows=sequence_length,
            columns=hidden_features,
            owner_aligned=True,
        )
        canonical_mlp_internal_specs = (
            *((
                {
                    "source_operand": 30,
                    "memory": "l1",
                    "shape": [sequence_length, physical_rank],
                    "data_format": "BFloat16",
                    "layout": "tile",
                },
                {
                    "source_operand": 30,
                    "memory": "l1",
                    "shape": [sequence_length, physical_rank],
                    "data_format": "BFloat16",
                    "layout": "tile",
                },
                {
                    "source_operand": 30,
                    "memory": "l1",
                    "shape": [sequence_length, intermediate_features],
                    "data_format": "BFloat16",
                    "layout": "tile",
                },
                {
                    "source_operand": 30,
                    "memory": "l1",
                    "shape": [sequence_length, intermediate_features],
                    "data_format": "BFloat16",
                    "layout": "tile",
                },
            ) if precomputed_mlp_rank_delta else ()),
            *(
                (
                    {
                        "source_operand": 30,
                        "memory": "l1",
                        "shape": [
                            sequence_length,
                            2 * intermediate_features + 2 * physical_rank,
                        ],
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 30,
                        "memory": "l1",
                        "shape": [
                            sequence_length,
                            hidden_features
                            + (0 if production_ttnn_down else physical_rank),
                        ],
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 30,
                        "memory": "l1",
                        "shape": [sequence_length, hidden_features],
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                    *(
                        (
                            {
                                "source_operand": 30,
                                "memory": "l1",
                                "shape": [sequence_length, intermediate_features],
                                "data_format": "BFloat16",
                                "layout": "tile",
                            },
                            {
                                "source_operand": 30,
                                "memory": "l1",
                                "shape": [sequence_length, intermediate_features],
                                "data_format": "BFloat16",
                                "layout": "tile",
                            },
                            {
                                "source_operand": 30,
                                "memory": "l1",
                                "shape": [intermediate_features, hidden_features],
                                "data_format": "BFloat16",
                                "layout": "tile",
                            },
                            *down_mlp.internal_tensor_specs,
                        )
                        if production_ttnn_down else ()
                    ),
                ) if production_exact_mlp else ()
            ),
            *(() if production_exact_mlp else (
            {
                # Keep the first MLP-private ID out of the attention owner's
                # final allocation slot.  Artifact-v4 currently retains one
                # allocator guard entry at this boundary.
                "source_operand": 30,
                "memory": "l1",
                "shape": [1, 1],
                "data_format": "BFloat16",
                "layout": "tile",
            },
            *(
                {
                    "source_operand": 30,
                    "memory": "l1",
                    "shape": [sequence_length, intermediate_features],
                    "data_format": "BFloat16",
                    "layout": "tile",
                }
                for _ in range(2)
            ),
            {
                "source_operand": 30,
                "memory": "clone",
                "shape": [intermediate_features, hidden_features],
                "data_format": "BFloat16",
                "layout": "tile",
            },
            *(
                {
                    "source_operand": 30,
                    "memory": "l1",
                    "shape": shape,
                    "data_format": "BFloat16",
                    "layout": "tile",
                }
                for shape in (
                    [sequence_length, hidden_features],
                    [sequence_length, hidden_features],
                )
            ),
            *(
                {**dict(spec), "source_operand": 30}
                for spec in down_mlp.internal_tensor_specs
            ),
            )),
        )
        if production_native_gate_up:
            def mlp_spec(shape: list[int]) -> dict[str, object]:
                return {
                    "source_operand": 30,
                    "memory": "l1",
                    "shape": shape,
                    "data_format": "BFloat16",
                    "layout": "tile",
                }

            canonical_mlp_internal_specs = (
                mlp_spec([hidden_features, intermediate_features]),
                mlp_spec([hidden_features, intermediate_features]),
                mlp_spec([sequence_length, intermediate_features]),
                mlp_spec([sequence_length, intermediate_features]),
                *(
                    {**dict(spec), "source_operand": 30}
                    for spec in grouped_mlp.internal_tensor_specs
                ),
                mlp_spec([intermediate_features, hidden_features]),
                mlp_spec([sequence_length, hidden_features]),
                *(
                    {**dict(spec), "source_operand": 30}
                    for spec in down_mlp.internal_tensor_specs
                ),
                mlp_spec([sequence_length, hidden_features]),
            )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("compact layer forward requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30,
    ):
        source = ttl.make_dataflow_buffer_like(t00, shape=(1, 1), block_count=1)
        output = ttl.make_dataflow_buffer_like(t30, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(t00[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                tx = ttl.copy(tile, t30[0:1, 0:1])
                tx.wait()

    variant = (
        f"{sequence_length}x{hidden_features}.i{intermediate_features}."
        f"q{query_heads}.k{kv_heads}.d{head_dim}.r{physical_rank}."
        f"bias{int(has_bias)}.hidden{hidden_page_base}."
        f"targets{'-'.join(active_lora_roles)}."
        f"page0{int(publish_layer_output_at_page_zero)}."
        f"extres{int(debug_external_mlp_residual)}."
        f"exactattn{int(exact_attention_tail)}."
        f"bf16{int(canonical_bf16_attention)}."
        f"groupedrank{int(canonical_grouped_qkv_rank)}."
        f"separaterank{int(canonical_separate_qkv_rank)}."
        f"rank{logical_rank}.oloraphase{o_lora_k_tiles_per_phase}."
        f"gatephase{gate_up_base_k_tiles_per_phase}."
        f"ttnndown{int(production_ttnn_down)}."
        f"spill{int(spill_rank_scaled_intermediates)}."
        f"predeltamlp{int(precomputed_mlp_rank_delta)}."
        f"{'fusedmlpchk1.' if fused_mlp_checkpoint_publication else ''}"
        f"{'fusedmlpout1.' if fused_mlp_output_publication else ''}"
        f"{'fusedattnchk1.' if fused_attention_checkpoint_publication else ''}"
        f"debugmlp{debug_mlp_stage or 'down'}.v85"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.resident_layer_forward_compact.{variant}",
        arg_roles=("in",) * 30 + ("out",),
        version_tag=f"resident-layer-forward-compact-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
            t30,
        )
        # Local 31 is post-attention normalized X. Local 32/33 are the
        # compact attention owner's two private ping-pong tensors.
        attention_arguments = (
            *tensors[:21], t27, t28, t30, t29,
        )
        attention_children = compile_serialized_sequence_children(
            attention, attention_arguments
        )
        if len(attention_children) != attention.physical_programs:
            raise RuntimeError("compact attention physical count changed")
        attention_map = {
            **{index: index for index in range(21)},
            21: 27,
            22: 28,
            23: 30,
            24: 31,
        }
        attention_map.update({
            25 + index: 32 + index
            for index in range(attention.compiler_owned_l1_tensors)
        })
        children = [
            _remap_used(child, attention_map)
            for child in attention_children
        ]

        residual_checkpoint_offset = monolithic.forward_plan.arena.slice(
            "post_attention_residual"
        ).page_offset
        normalized_checkpoint_offset = monolithic.forward_plan.arena.slice(
            "post_attention_normalized"
        ).page_offset
        if fused_attention_checkpoint_publication:
            if len(children) < 2:
                raise RuntimeError(
                    "compact attention has no residual/RMSNorm publication tail"
                )
            children[-2] = _fuse_residual_add_publication(
                children[-2],
                row_tiles=sequence_length // 32,
                column_tiles=hidden_features // 32,
                destination_page_offsets=(residual_checkpoint_offset,),
                output_tensor_index=32,
                checkpoint_tensor_index=30,
            )
            children[-1] = _fuse_rmsnorm_checkpoint_publication(
                children[-1],
                row_tiles=sequence_length // 32,
                column_tiles=hidden_features // 32,
                destination_page_offset=normalized_checkpoint_offset,
                output_tensor_index=31,
                checkpoint_tensor_index=30,
            )

        # Tensors 32 and 31 remain in L1 for the immediate MLP handoff, but
        # they are also long-lived checkpoints consumed after the complete
        # model forward. Publish them once into the packed arena without
        # replacing the private handoff. Previously those arena regions kept
        # their zero initializer, so real-model backward consumed zeros even
        # though the compact forward/MLP prefix itself executed correctly.
        if not fused_attention_checkpoint_publication:
            residual_copy = compile_serialized_child(
                checkpoint_copy, (t00, t30)
            )
            residual_copy = retarget_tensor_writes_to_packed_pages(
                residual_copy,
                packed_tensor=1,
                tensor_page_offsets={1: residual_checkpoint_offset},
            )
            normalized_copy = compile_serialized_child(
                checkpoint_copy, (t00, t30)
            )
            normalized_copy = retarget_tensor_writes_to_packed_pages(
                normalized_copy,
                packed_tensor=1,
                tensor_page_offsets={1: normalized_checkpoint_offset},
            )
            children.extend((
                _remap_used(residual_copy, {0: 32, 1: 30}),
                _remap_used(normalized_copy, {0: 31, 1: 30}),
            ))

        # Reset both semaphore mailboxes and the program's CB epoch before
        # the following all-core MLP producer.  A noc-only clear was enough
        # for the smaller Smol shape but left part of Qwen's 110-core group
        # in the preceding attention allocation epoch.
        children.append(_make_pure_semaphore_scrub(
            compile_serialized_child(semaphore_scrub, (t00, t29))
        ))

        if production_native_gate_up:
            hidden_tiles = hidden_features // 32
            intermediate_tiles = intermediate_features // 32
            sequence_tiles = sequence_length // 32
            rank_tiles = physical_rank // 32
            base = 32 + attention.compiler_owned_l1_tensors
            gate_weight_id, up_weight_id = base, base + 1
            gate_id, up_id = base + 2, base + 3
            pair_internal = base + 4
            down_weight_id = pair_internal + grouped_mlp.compiler_owned_l1_tensors
            down_id = down_weight_id + 1
            down_internal = down_id + 1
            layer_output_id = down_internal + down_mlp.compiler_owned_l1_tensors
            grouped_weight_offset = hidden_tiles + 16
            down_weight_offset = (
                grouped_weight_offset
                + hidden_tiles * 2 * intermediate_tiles
                + 2 * rank_tiles * hidden_tiles
                + 2 * rank_tiles * intermediate_tiles
            )
            grouped_weight = _placeholder_like(
                t00, (hidden_features, 2 * intermediate_features)
            )
            projection_weight = _placeholder_like(
                t00, (hidden_features, intermediate_features)
            )
            for section, destination in (
                (0, gate_weight_id),
                (intermediate_tiles, up_weight_id),
            ):
                child = compile_serialized_child(
                    build_tt_lang_owner_aligned_copy(
                        rows=hidden_features, columns=intermediate_features
                    ),
                    (grouped_weight, projection_weight),
                )
                child = make_serialized_strided_page_copy_bitwise(
                    child,
                    rows_tiles=hidden_tiles,
                    section_column_tiles=intermediate_tiles,
                    source_row_tiles=2 * intermediate_tiles,
                    source_section_offset=section,
                    source_page_offset=grouped_weight_offset,
                )
                children.append(_remap_used(
                    child, {0: 20, 1: destination}
                ))
            gate_seed = _placeholder_like(
                t00, (sequence_length, intermediate_features)
            )
            up_seed = _placeholder_like(
                t00, (sequence_length, intermediate_features)
            )
            pair_map = {
                0: 31, 1: gate_weight_id, 2: 21, 3: 22,
                4: up_weight_id, 5: 23, 6: 24,
                7: gate_id, 8: up_id,
                **{
                    9 + i: pair_internal + i
                    for i in range(grouped_mlp.compiler_owned_l1_tensors)
                },
            }
            mapped_pair = [
                _remap_used(child, pair_map)
                for child in compile_serialized_sequence_children(
                    grouped_mlp,
                    (
                        t00, projection_weight, t21, t22,
                        projection_weight, t23, t24,
                        gate_seed, up_seed,
                    ),
                )
            ]
            for index, child in enumerate(mapped_pair):
                children.append(child)
                h_checkpoint = {
                    0: (pair_internal, monolithic.mlp_gate_h_page_offset),
                    6: (pair_internal + 6, monolithic.mlp_up_h_page_offset),
                }.get(index)
                if h_checkpoint is not None:
                    source, destination = h_checkpoint
                    h_child = compile_serialized_child(
                        build_tt_lang_owner_aligned_copy(
                            rows=sequence_length, columns=physical_rank
                        ),
                        (
                            _placeholder_like(
                                t00, (sequence_length, physical_rank)
                            ),
                            t30,
                        ),
                    )
                    h_child = make_serialized_page_copy_bitwise(
                        h_child,
                        total_pages=sequence_tiles * rank_tiles,
                        column_tiles=rank_tiles,
                        source_page_offset=0,
                    )
                    h_child = retarget_tensor_writes_to_packed_pages(
                        h_child, packed_tensor=1,
                        tensor_page_offsets={1: destination},
                    )
                    children.append(_remap_used(
                        h_child, {0: source, 1: 30}
                    ))
            for source, columns, destination in (
                (gate_id, intermediate_features, monolithic.mlp_gate_page_offset),
                (up_id, intermediate_features, monolithic.mlp_up_page_offset),
            ):
                child = compile_serialized_child(
                    build_tt_lang_owner_aligned_copy(
                        rows=sequence_length, columns=columns
                    ),
                    (_placeholder_like(t00, (sequence_length, columns)), t30),
                )
                child = make_serialized_page_copy_bitwise(
                    child,
                    total_pages=sequence_tiles * (columns // 32),
                    column_tiles=columns // 32,
                    source_page_offset=0,
                )
                child = retarget_tensor_writes_to_packed_pages(
                    child, packed_tensor=1,
                    tensor_page_offsets={1: destination},
                )
                children.append(_remap_used(child, {0: source, 1: 30}))
            down_weight = _placeholder_like(
                t00, (intermediate_features, hidden_features)
            )
            down_weight_destination = _placeholder_like(
                t00, (intermediate_features, hidden_features)
            )
            child = compile_serialized_child(
                build_tt_lang_owner_aligned_copy(
                    rows=intermediate_features, columns=hidden_features
                ),
                (down_weight, down_weight_destination),
            )
            child = make_serialized_page_copy_bitwise(
                child,
                total_pages=intermediate_tiles * hidden_tiles,
                column_tiles=hidden_tiles,
                source_page_offset=down_weight_offset,
            )
            children.append(_remap_used(
                child, {0: 20, 1: down_weight_id}
            ))
            down_seed = _placeholder_like(
                t00, (sequence_length, hidden_features)
            )
            down_map = {
                0: gate_id, 1: up_id, 2: down_weight_id,
                3: 25, 4: 26, 5: down_id,
                **{
                    6 + i: down_internal + i
                    for i in range(down_mlp.compiler_owned_l1_tensors)
                },
            }
            children.extend(
                _remap_used(child, down_map)
                for child in compile_serialized_sequence_children(
                    down_mlp,
                    (gate_seed, up_seed, down_weight, t25, t26, down_seed),
                )
            )
            layer_output = _placeholder_like(
                t00, (sequence_length, hidden_features)
            )
            child = compile_serialized_child(
                mlp_add, (down_seed, t00, layer_output)
            )
            children.append(_remap_used(child, {
                0: down_id,
                1: 29 if debug_external_mlp_residual else 32,
                2: layer_output_id,
            }))
            destinations = [monolithic.layer_output_page_offset]
            if publish_layer_output_at_page_zero and 0 not in destinations:
                destinations.append(0)
            for destination in destinations:
                child = compile_serialized_child(
                    checkpoint_copy, (layer_output, t30)
                )
                child = retarget_tensor_writes_to_packed_pages(
                    child, packed_tensor=1,
                    tensor_page_offsets={1: destination},
                )
                children.append(_remap_used(
                    child, {0: layer_output_id, 1: 30}
                ))
        elif production_exact_mlp:
            hidden_tiles = hidden_features // 32
            intermediate_tiles = intermediate_features // 32
            sequence_tiles = sequence_length // 32
            rank_tiles = physical_rank // 32
            grouped_row_tiles = 2 * intermediate_tiles + 2 * rank_tiles
            mlp_private_base = 32 + attention.compiler_owned_l1_tensors
            group_id = mlp_private_base + (
                4 if precomputed_mlp_rank_delta else 0
            )
            down_id = group_id + 1
            layer_output_id = group_id + 2

            grouped_arena = _placeholder_like(
                t00, (sequence_length, grouped_row_tiles * 32)
            )
            grouped_weight = _placeholder_like(
                t00, (hidden_features, 2 * intermediate_features)
            )
            if precomputed_mlp_rank_delta:
                gate_h_seed = _placeholder_like(
                    t00, (sequence_length, physical_rank)
                )
                up_h_seed = _placeholder_like(
                    t00, (sequence_length, physical_rank)
                )
                gate_delta_seed = _placeholder_like(
                    t00, (sequence_length, intermediate_features)
                )
                up_delta_seed = _placeholder_like(
                    t00, (sequence_length, intermediate_features)
                )
                rank_compute = {
                    "math_fidelity": "HiFi4",
                    "math_approx_mode": True,
                    "fp32_dest_acc_en": True,
                    "packer_l1_acc": False,
                    "dst_full_sync_en": False,
                }
                delta_compute = {
                    "math_fidelity": "HiFi4",
                    "math_approx_mode": False,
                    "fp32_dest_acc_en": True,
                    "packer_l1_acc": False,
                    "dst_full_sync_en": False,
                }
                if physical_rank > 64:
                    delta_compute = {
                        "math_fidelity": "HiFi4",
                        "fp32_dest_acc_en": True,
                    }
                # The accepted grouped2 oracle reduces each rank output tile
                # independently in two-K-tile phases.  Auto-selected TTNN
                # widens the N block for R64+, changing the partial-sum spill
                # schedule by one BF16 ULP.  Keep the same all-core 2-D
                # schedule as the oracle for every physical rank tile.
                rank_program = {
                    "family": "2d", "grid": [11, 10],
                    "in0_block_w": 2,
                    "out_subblock_h": 1, "out_subblock_w": 1,
                    "out_block_h": 1, "out_block_w": 1,
                    "per_core_m": 1, "per_core_n": 1,
                    "transpose_mcast": False, "fuse_batch": False,
                }
                delta_n_block = (
                    intermediate_tiles + 11 - 1
                ) // 11
                wide_delta_program = {
                    "family": "2d",
                    "grid": [
                        (intermediate_tiles + delta_n_block - 1)
                        // delta_n_block,
                        sequence_tiles,
                    ],
                    "in0_block_w": 1,
                    "out_subblock_h": 1,
                    "out_subblock_w": 1,
                    "out_block_h": 1,
                    "out_block_w": delta_n_block,
                    "per_core_m": 1,
                    "per_core_n": delta_n_block,
                    "transpose_mcast": False,
                    "fuse_batch": False,
                }
                rank_children = [
                    {
                        "operation": "ttnn.matmul",
                        "operand_indices": [31, 21, mlp_private_base],
                        "transpose_a": False,
                        "transpose_b": True,
                        "compute_config": rank_compute,
                    },
                    {
                        "operation": "ttnn.matmul",
                        "operand_indices": [31, 23, mlp_private_base + 1],
                        "transpose_a": False,
                        "transpose_b": True,
                        "compute_config": rank_compute,
                    },
                ]
                # R64's safe graph uses the independent one-N-tile schedule;
                # wider ranks select a different TTNN geometry and must be
                # left to the shape-aware selector.
                if physical_rank == 64:
                    for rank_child in rank_children:
                        rank_child["program_config"] = rank_program
                children.extend(rank_children)
                children.extend((
                    {
                        "operation": "ttnn.matmul",
                        "operand_indices": [
                            mlp_private_base, 22, mlp_private_base + 2,
                        ],
                        "transpose_a": False,
                        "transpose_b": True,
                        **({"program_config": wide_delta_program}
                           if physical_rank > 64 else {}),
                        "compute_config": delta_compute,
                    },
                    {
                        "operation": "ttnn.matmul",
                        "operand_indices": [
                            mlp_private_base + 1, 24,
                            mlp_private_base + 3,
                        ],
                        "transpose_a": False,
                        "transpose_b": True,
                        **({"program_config": wide_delta_program}
                           if physical_rank > 64 else {}),
                        "compute_config": delta_compute,
                    },
                ))
                group_operands = (
                    t00, grouped_weight,
                    t21, gate_delta_seed,
                    t23, up_delta_seed,
                    grouped_arena,
                )
            else:
                group_operands = (
                    t00, grouped_weight, t21, t22, t23, t24, grouped_arena
                )
            group_child = compile_serialized_child(
                grouped_mlp, group_operands,
            )
            grouped_weight_offset = hidden_tiles + 16
            grouped_weight_pages = hidden_tiles * 2 * intermediate_tiles
            down_weight_offset = (
                grouped_weight_offset
                + grouped_weight_pages
                + 2 * rank_tiles * hidden_tiles
                + 2 * rank_tiles * intermediate_tiles
            )
            group_child = retarget_tensor_reads_to_packed_pages(
                group_child,
                packed_tensor=20,
                tensor_page_offsets={1: grouped_weight_offset},
            )
            if fused_mlp_checkpoint_publication:
                group_child = _fuse_grouped2_checkpoint_publication(
                    group_child,
                    sequence_tiles=sequence_tiles,
                    intermediate_tiles=intermediate_tiles,
                    rank_tiles=rank_tiles,
                    gate_page_offset=monolithic.mlp_gate_page_offset,
                    up_page_offset=monolithic.mlp_up_page_offset,
                    gate_h_page_offset=monolithic.mlp_gate_h_page_offset,
                    up_h_page_offset=monolithic.mlp_up_h_page_offset,
                )
            group_mapping = {
                0: 31,
                20: 20,
                2: (
                    21
                ),
                3: (
                    mlp_private_base + 2
                    if precomputed_mlp_rank_delta else 22
                ),
                4: (
                    23
                ),
                5: (
                    mlp_private_base + 3
                    if precomputed_mlp_rank_delta else 24
                ),
                6: group_id,
                **({7: 30} if fused_mlp_checkpoint_publication else {}),
            }
            children.append(_remap_used(group_child, group_mapping))

            checkpoint_sections = (
                (intermediate_features, 0, monolithic.mlp_gate_page_offset),
                (
                    intermediate_features,
                    intermediate_tiles,
                    monolithic.mlp_up_page_offset,
                ),
                (
                    physical_rank,
                    2 * intermediate_tiles,
                    monolithic.mlp_gate_h_page_offset,
                ),
                (
                    physical_rank,
                    2 * intermediate_tiles + rank_tiles,
                    monolithic.mlp_up_h_page_offset,
                ),
            )
            if not fused_mlp_checkpoint_publication:
                for (
                    columns,
                    section_offset,
                    destination_offset,
                ) in checkpoint_sections:
                    section_copy = build_tt_lang_owner_aligned_copy(
                        rows=sequence_length, columns=columns
                    )
                    copy_child = compile_serialized_child(
                        section_copy, (grouped_arena, t30)
                    )
                    copy_child = make_serialized_strided_page_copy_bitwise(
                        copy_child,
                        rows_tiles=sequence_tiles,
                        section_column_tiles=columns // 32,
                        source_row_tiles=grouped_row_tiles,
                        source_section_offset=section_offset,
                    )
                    copy_child = retarget_tensor_writes_to_packed_pages(
                        copy_child,
                        packed_tensor=1,
                        tensor_page_offsets={1: destination_offset},
                    )
                    children.append(_remap_used(
                        copy_child, {0: group_id, 1: 30}
                    ))

            down_weight = _placeholder_like(
                t00, (intermediate_features, hidden_features)
            )
            down_a = _placeholder_like(
                t00, (physical_rank, intermediate_features)
            )
            down_b = _placeholder_like(t00, (hidden_features, physical_rank))
            down_arena = _placeholder_like(
                t00,
                (
                    sequence_length,
                    hidden_features
                    + (0 if production_ttnn_down else physical_rank),
                ),
            )
            if production_ttnn_down:
                gate_id = group_id + 3
                up_id = group_id + 4
                down_weight_id = group_id + 5
                down_internal_base = group_id + 6
                for section_offset, private_id in (
                    (0, gate_id),
                    (intermediate_tiles, up_id),
                ):
                    private_projection = _placeholder_like(
                        t00, (sequence_length, intermediate_features)
                    )
                    section_copy = build_tt_lang_owner_aligned_copy(
                        rows=sequence_length, columns=intermediate_features
                    )
                    copy_child = compile_serialized_child(
                        section_copy, (grouped_arena, private_projection)
                    )
                    copy_child = make_serialized_strided_page_copy_bitwise(
                        copy_child,
                        rows_tiles=sequence_tiles,
                        section_column_tiles=intermediate_tiles,
                        source_row_tiles=grouped_row_tiles,
                        source_section_offset=section_offset,
                    )
                    children.append(_remap_used(
                        copy_child, {0: group_id, 1: private_id}
                    ))

                weight_copy = build_tt_lang_owner_aligned_copy(
                    rows=intermediate_features, columns=hidden_features
                )
                down_weight_l1 = _placeholder_like(
                    t00, (intermediate_features, hidden_features)
                )
                weight_child = compile_serialized_child(
                    weight_copy, (down_weight, down_weight_l1)
                )
                weight_child = retarget_tensor_reads_to_packed_pages(
                    weight_child,
                    packed_tensor=0,
                    tensor_page_offsets={0: down_weight_offset},
                )
                children.append(_remap_used(
                    weight_child, {0: 20, 1: down_weight_id}
                ))

                down_children = compile_serialized_sequence_children(
                    down_mlp,
                    (
                        _placeholder_like(
                            t00, (sequence_length, intermediate_features)
                        ),
                        _placeholder_like(
                            t00, (sequence_length, intermediate_features)
                        ),
                        down_weight_l1,
                        down_a,
                        down_b,
                        down_arena,
                    ),
                )
                down_map = {
                    0: gate_id,
                    1: up_id,
                    2: down_weight_id,
                    3: 25,
                    4: 26,
                    5: down_id,
                    **{
                        6 + index: down_internal_base + index
                        for index in range(down_mlp.compiler_owned_l1_tensors)
                    },
                }
                mapped_down_children = [
                    _remap_used(child, down_map) for child in down_children
                ]
                if debug_publish_down and debug_mlp_stage in (
                    "product", "base", "delta", "scaled", "down"
                ):
                    debug_sources = {
                        "product": (down_internal_base + 1, intermediate_features, 1),
                        "delta": (down_internal_base + 4, hidden_features, 3),
                        "scaled": (down_internal_base + 6, hidden_features, 5),
                        "base": (down_internal_base + 3, hidden_features, 6),
                        "down": (down_id, hidden_features, 7),
                    }
                    debug_id, debug_columns, producer_index = debug_sources[
                        debug_mlp_stage
                    ]
                    debug_copy = build_tt_lang_owner_aligned_copy(
                        rows=sequence_length, columns=debug_columns
                    )
                    debug_seed = _placeholder_like(
                        t00, (sequence_length, debug_columns)
                    )
                    debug_child = compile_serialized_child(
                        debug_copy, (debug_seed, t30)
                    )
                    debug_child = make_serialized_page_copy_bitwise(
                        debug_child,
                        total_pages=(
                            sequence_tiles * (debug_columns // 32)
                        ),
                        column_tiles=debug_columns // 32,
                        source_page_offset=0,
                    )
                    debug_child = retarget_tensor_writes_to_packed_pages(
                        debug_child,
                        packed_tensor=1,
                        tensor_page_offsets={
                            1: monolithic.forward_plan.arena.slice(
                                "post_attention_residual"
                            ).page_offset,
                        },
                    )
                    mapped_down_children.insert(
                        producer_index + 1,
                        _remap_used(debug_child, {0: debug_id, 1: 30}),
                    )
                children.extend(mapped_down_children)
            else:
                grouped_arena_up = _placeholder_like(
                    t00, (sequence_length, grouped_row_tiles * 32)
                )
                down_child = compile_serialized_child(
                    down_mlp,
                    (
                        grouped_arena,
                        grouped_arena_up,
                        down_weight,
                        down_a,
                        down_b,
                        down_arena,
                    ),
                )
                down_child = retarget_tensor_reads_to_packed_pages(
                    down_child,
                    packed_tensor=20,
                    tensor_page_offsets={2: down_weight_offset},
                )
                children.append(_remap_used(down_child, {
                    0: group_id,
                    1: group_id,
                    20: 20,
                    3: 25,
                    4: 26,
                    5: down_id,
                }))

            layer_output = _placeholder_like(
                t00, (sequence_length, hidden_features)
            )
            publication_offsets = [monolithic.layer_output_page_offset]
            if publish_layer_output_at_page_zero and 0 not in publication_offsets:
                publication_offsets.append(0)
            add_child = compile_serialized_child(
                mlp_add, (down_arena, t00, layer_output)
            )
            if fused_mlp_output_publication:
                add_child = _fuse_residual_add_publication(
                    add_child,
                    row_tiles=sequence_tiles,
                    column_tiles=hidden_tiles,
                    destination_page_offsets=tuple(publication_offsets),
                )
            children.append(_remap_used(add_child, {
                0: down_id,
                1: 29 if debug_external_mlp_residual else 32,
                2: layer_output_id,
                **({3: 30} if fused_mlp_output_publication else {}),
            }))

            if not fused_mlp_output_publication:
                for destination_offset in publication_offsets:
                    publish = compile_serialized_child(
                        checkpoint_copy, (layer_output, t30)
                    )
                    publish = retarget_tensor_writes_to_packed_pages(
                        publish,
                        packed_tensor=1,
                        tensor_page_offsets={1: destination_offset},
                    )
                    children.append(_remap_used(
                        publish, {0: layer_output_id, 1: 30}
                    ))
            if debug_publish_down and debug_mlp_stage in (
                "gate_private", "up_private"
            ):
                # Keep the grouped L1 arena live through final publication,
                # then expose one hidden-width prefix for diagnosis.
                debug_projection = build_tt_lang_owner_aligned_copy(
                    rows=sequence_length, columns=hidden_features
                )
                debug_child = compile_serialized_child(
                    debug_projection, (grouped_arena, t30)
                )
                debug_child = make_serialized_strided_page_copy_bitwise(
                    debug_child,
                    rows_tiles=sequence_tiles,
                    section_column_tiles=hidden_tiles,
                    source_row_tiles=grouped_row_tiles,
                    source_section_offset=(
                        0
                        if debug_mlp_stage == "gate_private"
                        else intermediate_tiles
                    ),
                )
                debug_child = retarget_tensor_writes_to_packed_pages(
                    debug_child,
                    packed_tensor=1,
                    tensor_page_offsets={
                        1: monolithic.forward_plan.arena.slice(
                            "post_attention_residual"
                        ).page_offset,
                    },
                )
                children.append(_remap_used(
                    debug_child, {0: group_id, 1: 30}
                ))
        elif canonical_mlp:
            hidden_tiles = hidden_features // 32
            intermediate_tiles = intermediate_features // 32
            guard_id = 32 + attention.compiler_owned_l1_tensors
            gate_id = guard_id + 1
            up_id = guard_id + 2
            down_weight_id = guard_id + 3
            down_id = down_weight_id + 1
            layer_output_id = down_id + 1
            down_internal_base = layer_output_id + 1

            grouped_weight = _placeholder_like(
                t00, (hidden_features, 2 * intermediate_features)
            )

            post_gamma_pages = hidden_tiles
            zero_pages = 16
            grouped_weight_offset = post_gamma_pages + zero_pages
            grouped_weight_pages = hidden_tiles * 2 * intermediate_tiles
            down_weight_offset = (
                grouped_weight_offset
                + grouped_weight_pages
                + 2 * hidden_tiles
                + 2 * intermediate_tiles
            )

            grouped_child = compile_serialized_child(
                grouped_mlp,
                (t00, grouped_weight, t21, t22, t23, t24, t30),
            )
            grouped_child = retarget_tensor_reads_to_packed_pages(
                grouped_child,
                packed_tensor=20,
                tensor_page_offsets={1: grouped_weight_offset},
            )
            grouped_child = _split_grouped_mlp_writes(
                grouped_child,
                gate_page_offset=monolithic.mlp_gate_page_offset,
                up_page_offset=monolithic.mlp_up_page_offset,
                gate_h_page_offset=monolithic.mlp_gate_h_page_offset,
                up_h_page_offset=monolithic.mlp_up_h_page_offset,
            )
            grouped_child = _offset_hardcoded_cbs(grouped_child, offset=16)
            children.append(_remap_used(grouped_child, {
                0: 31,
                20: 20,
                2: 21,
                3: 22,
                4: 23,
                5: 24,
                6: gate_id,
                7: up_id,
                8: 30,
            }))
            children.append(
                _make_all_worker_barrier(
                    compile_serialized_child(semaphore_scrub, (t00, t29))
                )
            )

            down_weight_t = _placeholder_like(
                t00, (intermediate_features, hidden_features)
            )
            down_output = _placeholder_like(
                t00, (sequence_length, hidden_features)
            )
            weight_copy = build_tt_lang_owner_aligned_copy(
                rows=intermediate_features, columns=hidden_features
            )
            down_weight_l1 = _placeholder_like(
                t00, (intermediate_features, hidden_features)
            )
            weight_child = compile_serialized_child(
                weight_copy, (down_weight_t, down_weight_l1)
            )
            weight_child = retarget_tensor_reads_to_packed_pages(
                weight_child,
                packed_tensor=0,
                tensor_page_offsets={0: down_weight_offset},
            )
            children.append(_remap_used(weight_child, {
                0: 20,
                1: down_weight_id,
            }))

            down_children = compile_serialized_sequence_children(
                down_mlp,
                (
                    _placeholder_like(
                        t00, (sequence_length, intermediate_features)
                    ),
                    _placeholder_like(
                        t00, (sequence_length, intermediate_features)
                    ),
                    down_weight_t,
                    t25,
                    t26,
                    down_output,
                ),
            )
            down_map = {
                0: gate_id,
                1: up_id,
                2: down_weight_id,
                3: 25,
                4: 26,
                5: down_id,
                **{
                    6 + index: down_internal_base + index
                    for index in range(down_mlp.compiler_owned_l1_tensors)
                },
            }
            mapped_down_children = [
                _remap_used(child, down_map) for child in down_children
            ]

            if debug_publish_down:
                debug_sources = {
                    "gate_private": gate_id,
                    "up_private": up_id,
                    "weight": down_weight_id,
                    "product": down_internal_base + 1,
                    "base": down_internal_base + 3,
                    "delta": down_internal_base + 4,
                    "scaled": down_internal_base + 6,
                    "down": down_id,
                }
                debug_source_id = debug_sources.get(
                    debug_mlp_stage or "down", down_id
                )
                debug_source_shape = (
                    (sequence_length, intermediate_features)
                    if debug_mlp_stage in (
                        "gate_private", "up_private", "product"
                    )
                    else (
                        (intermediate_features, hidden_features)
                        if debug_mlp_stage == "weight"
                        else (sequence_length, hidden_features)
                    )
                )
                publish_down = compile_serialized_child(
                    checkpoint_copy,
                    (_placeholder_like(t00, debug_source_shape), t30),
                )
                publish_down = retarget_tensor_writes_to_packed_pages(
                    publish_down,
                    packed_tensor=1,
                    tensor_page_offsets={
                        1: monolithic.forward_plan.arena.slice(
                            "post_attention_residual"
                        ).page_offset,
                    },
                )
                debug_publish_child = _remap_used(
                    publish_down, {0: debug_source_id, 1: 30}
                )
                # Publish the selected value immediately after its producer.
                # Internal TTNN tensors are intentionally eligible for reuse
                # once their last real consumer has run, so a diagnostic copy
                # appended after the whole sequence can observe a later value.
                if debug_mlp_stage in ("gate_private", "up_private"):
                    children.append(debug_publish_child)
                elif debug_mlp_stage == "weight":
                    children.append(debug_publish_child)
                elif debug_mlp_stage == "product":
                    mapped_down_children.insert(2, debug_publish_child)
                elif debug_mlp_stage == "delta":
                    mapped_down_children.insert(4, debug_publish_child)
                elif debug_mlp_stage == "base":
                    mapped_down_children.insert(7, debug_publish_child)
                else:
                    mapped_down_children.append(debug_publish_child)

            children.extend(mapped_down_children)

            layer_output = _placeholder_like(
                t00, (sequence_length, hidden_features)
            )
            add_child = compile_serialized_child(
                mlp_add, (down_output, t00, layer_output)
            )
            children.append(_remap_used(add_child, {
                0: down_id,
                1: 29 if debug_external_mlp_residual else 32,
                2: layer_output_id,
            }))

            publication_offsets = [monolithic.layer_output_page_offset]
            if publish_layer_output_at_page_zero and 0 not in publication_offsets:
                publication_offsets.append(0)
            for destination_offset in publication_offsets:
                publish = compile_serialized_child(
                    checkpoint_copy, (layer_output, t30)
                )
                publish = retarget_tensor_writes_to_packed_pages(
                    publish,
                    packed_tensor=1,
                    tensor_page_offsets={1: destination_offset},
                )
                children.append(_remap_used(
                    publish, {0: layer_output_id, 1: 30}
                ))
        else:
            from .tt_lang_program_composition import (
                extract_resident_full_mlp_forward_child,
            )
            mlp_child = compile_serialized_child(
                monolithic, (*tensors[:27], t30)
            )
            mlp_child = extract_resident_full_mlp_forward_child(
                mlp_child, normalized_tensor=0, residual_tensor=28
            )
            mlp_map = {
                0: 31,
                **{index: index for index in range(1, 27)},
                27: 30,
                28: 29 if debug_external_mlp_residual else 32,
            }
            children.append(_remap_used(mlp_child, mlp_map))

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
                if exact_attention_tail:
                    compiled.internal_tensors = [
                        {
                            "source_operand": 0,
                            "memory": (
                                "clone"
                                if spill_rank_scaled_intermediates else "l1"
                            ),
                            "shape": [sequence_length, hidden_features],
                            "data_format": "BFloat16",
                            "layout": "tile",
                        },
                        *(
                            planned_internal_spec({
                                **spec,
                                "source_operand": (
                                    30
                                    if spec.get("source_operand") == 23
                                    else spec.get("source_operand", 0)
                                ),
                            })
                            for spec in attention.internal_tensor_specs
                        ),
                        *(
                            planned_internal_spec(spec)
                            for spec in canonical_mlp_internal_specs
                        ),
                    ]
                else:
                    compiled.internal_tensors = [
                        {"source_operand": 0, "memory": "l1"},
                        {"source_operand": 0, "memory": "l1"},
                        {"source_operand": 0, "memory": "l1"},
                    ]
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = (
        attention.physical_programs
        - (2 if fused_attention_checkpoint_publication else 0)
        + (
            (
                (21 if production_ttnn_down else 11)
                if production_exact_mlp else 16
            )
            + (4 if precomputed_mlp_rank_delta else 0)
            - (4 if fused_mlp_checkpoint_publication else 0)
            - (
                output_publication_programs
                if fused_mlp_output_publication and production_exact_mlp
                else 0
            )
            + int(debug_publish_down)
            + int(
                publish_layer_output_at_page_zero
                and monolithic.layer_output_page_offset != 0
            )
            if canonical_mlp else 4
        )
    )
    operation.compiler_owned_l1_tensors = (
        1 + attention.compiler_owned_l1_tensors + (
            len(canonical_mlp_internal_specs) if canonical_mlp else 0
        )
        if exact_attention_tail else 3
    )
    operation.output_shape = monolithic.output_shape
    operation.forward_plan = monolithic.forward_plan
    operation.layer_output_page_offset = monolithic.layer_output_page_offset
    operation.hidden_page_base = hidden_page_base
    operation.publishes_layer_output_at_page_zero = bool(
        publish_layer_output_at_page_zero
    )
    operation.fused_mlp_checkpoint_publication = bool(
        fused_mlp_checkpoint_publication
    )
    operation.fused_mlp_output_publication = bool(
        fused_mlp_output_publication
    )
    operation.fused_attention_checkpoint_publication = bool(
        fused_attention_checkpoint_publication
    )
    operation.layer_used_pages = monolithic.layer_used_pages
    operation.layer_allocated_pages = monolithic.layer_allocated_pages
    operation.mlp_gate_page_offset = monolithic.mlp_gate_page_offset
    operation.mlp_up_page_offset = monolithic.mlp_up_page_offset
    operation.mlp_gate_h_page_offset = monolithic.mlp_gate_h_page_offset
    operation.mlp_up_h_page_offset = monolithic.mlp_up_h_page_offset
    operation.resident_gate_up_intermediate_features = (
        monolithic.resident_gate_up_intermediate_features
    )
    operation.intermediate_dram_bytes = 0
    operation.checkpoint_dram_bytes = monolithic.checkpoint_dram_bytes
    operation.projection_cb_bytes = monolithic.projection_cb_bytes
    operation.reverse_handoff_noc_bytes = monolithic.reverse_handoff_noc_bytes
    operation.trainable_mlp_parameters = True
    operation.safe_default = False
    operation.experimental_backend = (
        "full_layer_forward_compact_exact_v4"
        if canonical_mlp
        else "full_layer_forward_compact_exact_v2"
    )
    operation.canonical_mlp = canonical_mlp
    if exact_attention_tail:
        operation.internal_tensor_specs = [
            {
                "source_operand": 0,
                "memory": (
                    "clone" if spill_rank_scaled_intermediates else "l1"
                ),
                "shape": [sequence_length, hidden_features],
                "data_format": "BFloat16",
                "layout": "tile",
            },
            *(
                planned_internal_spec({
                    **spec,
                    "source_operand": (
                        30
                        if spec.get("source_operand") == 23
                        else spec.get("source_operand", 0)
                    ),
                })
                for spec in attention.internal_tensor_specs
            ),
            *(
                planned_internal_spec(spec)
                for spec in canonical_mlp_internal_specs
            ),
        ]
    operation.exact_attention_tail = bool(exact_attention_tail)
    operation.canonical_bf16_attention = bool(canonical_bf16_attention)
    operation.canonical_grouped_qkv_rank = bool(canonical_grouped_qkv_rank)
    operation.canonical_separate_qkv_rank = bool(
        canonical_separate_qkv_rank
    )
    # internal_tensor_specs prepends the compact owner's hidden staging
    # tensor before the attention owner's private tensors.
    operation.qwen_grouped_prefix_internal_indices = tuple(
        1 + int(index) for index in getattr(
            attention, "qwen_grouped_prefix_internal_indices", ()
        )
    )
    operation.debug_external_mlp_residual = bool(
        debug_external_mlp_residual
    )
    operation._ttl_program = _probe
    return operation
