"""One-call exact resident backward for a complete SwiGLU LoRA MLP.

The program owns the down projection, exact SwiGLU derivative, and the
grouped gate/up projections.  The ten BF16 activation workers retain dGate
and dUp in banked L1 while the 100 FP32 matmul workers are repartitioned into
two 50-core groups.  Only the seven final gradients are written to ``arena``.

This module deliberately composes the already silicon-validated schedules.
It does not change their reduction order or BF16 pack boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from copy import copy
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .full_mlp_backward_planner import (
    full_mlp_backward_support_error,
    plan_full_mlp_backward,
)
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID
from .tt_lang_frozen_linear_lora_backward import (
    _shift_compile_time_cb_indices,
)
from .tt_lang_lora import TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR
from .tt_lang_resident_down_activation_backward import (
    _ACTIVATION_RANGE,
    _ALL_WORKERS_RANGE,
    _MATMUL_RANGE,
    _activation_compute_source,
    _banked_activation_compute_source,
    _cpp,
    _down_compute_source,
    _down_reader_source,
    _replace_all,
)

ttl = None
tt_lang_operation = None

_DOWN_CBS = 25
_ACTIVATION_CBS = 13
_MATMUL_NODES = 100
_ACTIVATION_NODES = 10
_MEMBER_NODES = 50


def _streamed_grouped_dh_cb_depths(
    *, group_jobs_per_core: int
) -> tuple[int, int]:
    """Return the minimum safe depths for grouped B and scaled-dY.

    The later LoRA-dX ``A`` stream owns a separate CB.  Both phase-one FIFOs
    therefore retain the exact two-tile ring geometry; a depth-three ring
    cannot safely serve repeated ``reserve_back(2)`` operations.
    """

    del group_jobs_per_core
    return 2, 2


def full_mlp_backward_output_shape(
    *, tokens: int, hidden_features: int, intermediate_features: int
) -> tuple[int, int]:
    plan = plan_full_mlp_backward(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        physical_rank=TILE_SIZE,
    )
    return (
        plan.output_tile_shape[0] * TILE_SIZE,
        plan.output_tile_shape[1] * TILE_SIZE,
    )


def _scale_bits(value: float) -> tuple[float, int]:
    bf16 = float(torch.tensor(value, dtype=torch.bfloat16).float().item())
    return bf16, struct.unpack("<I", struct.pack("<f", bf16))[0]


def _grouped_sources(
    *, replacements: dict[str, str], total_cbs: int
) -> tuple[str, str, str]:
    """Return the accepted 50+50 grouped reader/compute/writer sources."""

    lora = _shift_compile_time_cb_indices(
        _cpp("lora_backward_arena_exact_compute.cpp"), 4
    ).replace("void kernel_main()", "FORCE_INLINE void lora_backward_phase()", 1)
    base = _cpp("frozen_linear_backward_base_phase.cpp").replace(
        "void kernel_main()", "FORCE_INLINE void resident_phase_one()", 1
    )
    old_partition = """const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();"""
    local_partition = """const uint32_t global_node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();
    constexpr uint32_t member_nodes = @MEMBER_NODES@;
    const uint32_t node_index = global_node_index % member_nodes;"""
    phase = (lora + "\n" + base).replace(old_partition, local_partition)

    helper = _cpp("grouped2_frozen_lora_backward_tail_exact_compute.cpp").split(
        "\nvoid kernel_main()", 1
    )[0]
    helper = (
        helper.replace(
            "            dh.wait_front(reduction_block_tiles);\n",
            "            // Complete dH window is already resident.\n",
        )
        .replace(
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
        .replace(
            "            dh.pop_front(reduction_block_tiles);\n",
            "            // Keep dH resident for LoRA dX.\n",
        )
    )
    tail = (
        _cpp("grouped2_resident_backward_tail.cpp")
        .replace(
            "constexpr uint32_t grid_cols = 11;",
            "constexpr uint32_t grid_cols = 10;",
        )
        .replace(
            "constexpr uint32_t member_nodes = 55;",
            "constexpr uint32_t member_nodes = 50;",
        )
    )
    compute = phase + "\n" + helper + "\n" + tail

    reader = (
        _cpp("grouped2_resident_backward_reader.cpp")
        .replace(
            "constexpr uint32_t kGridCols = 11;\n"
            "constexpr uint32_t kMemberNodes = 55;",
            "constexpr uint32_t kGridCols = 10;\n"
            "constexpr uint32_t kMemberNodes = 50;",
        )
        .replace(
            "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
            "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};",
            "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
            "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};",
        )
    )
    writer = (
        _cpp("grouped2_resident_backward_writer.cpp")
        .replace(
            "constexpr uint32_t kGridCols = 11;\n"
            "constexpr uint32_t kMemberNodes = 55;",
            "constexpr uint32_t kGridCols = 10;\n"
            "constexpr uint32_t kMemberNodes = 50;",
        )
        .replace(
            "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
            "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};",
            "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
            "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};",
        )
    )

    sources = [compute, reader, writer]
    for index, source in enumerate(sources):
        source = source.replace(">(), 19>()", f">(), {total_cbs}>()")
        source = re.sub(
            r"get_tensor_accessor_args_cta_offset<(\d+), 19>",
            rf"get_tensor_accessor_args_cta_offset<\1, {total_cbs}>",
            source,
        )
        sources[index] = _replace_all(source, replacements)
    return sources[0], sources[1], sources[2]


def _prefix_grouped_compute(source: str) -> str:
    """Avoid helper-name collisions with the preceding down phase."""

    names = (
        "lora_backward_phase",
        "resident_phase_one",
        "add_tiles_bf16",
        "commit_bf16",
        "pack_reconfig_bf16",
        "pack_bf16",
        "release_bf16",
        "run_lora_dx",
        "run_bf16_add",
        "run_da",
        "run_resident_lora_dx",
        "resident_tail",
    )
    for name in names:
        source = re.sub(rf"\b{name}\b", f"full_grouped_{name}", source)
    return source.replace(
        "void kernel_main()", "FORCE_INLINE void full_grouped_compute_phase()", 1
    )


def _grouped_reader_from_activation_l1(
    source: str,
    *,
    down_jobs_per_node: int,
    down_total_jobs: int,
    activation_banks: int,
    scaled_gradient_banks: bool = False,
) -> str:
    """Replace every grouped dY DRAM read with an activation-L1 read."""

    source = source.replace("kGridCols", "kGroupedGridCols")
    source = source.replace("kMemberNodes", "kGroupedMemberNodes")
    source = source.replace("kPhysicalX", "kGroupedPhysicalX")
    source = source.replace("kTileBytes", "kGroupedTileBytes")
    source = re.sub(r"\bphysical_y\b", "grouped_physical_y", source)
    source = source.replace(
        "void kernel_main()", "FORCE_INLINE void full_grouped_reader_phase()", 1
    )

    marker = (
        "    const uint32_t group_y = member * 5;\n"
        "    volatile tt_l1_ptr uint32_t* go_sem ="
    )
    prelude = f"""    // Snapshot the activation-bank addresses before IDs 0..10 are
    // reused by the grouped dH and peer-sum schedule.
    volatile tt_l1_ptr uint32_t* activation_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    noc_semaphore_wait(activation_ready, 1);
    const uint32_t dgate_base0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(0));
    const uint32_t dup_base0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(1));
    uint32_t dgate_base1 = 0;
    uint32_t dup_base1 = 0;
    if constexpr ({activation_banks} == 2) {{
        dgate_base1 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(2));
        dup_base1 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(3));
    }}

    // Every matmul core must snapshot the bank mailboxes before any core
    // reuses IDs 0..10 for grouped dH and peer-sum synchronization.
    constexpr uint32_t snapshot_count_sem = 12;
    constexpr uint32_t snapshot_go_sem = 13;
    noc_semaphore_inc(
        get_noc_addr(kGroupedPhysicalX[0], grouped_physical_y(0),
                     get_semaphore(snapshot_count_sem)), 1);
    noc_async_atomic_barrier();
    if (global_node == 0) {{
        volatile tt_l1_ptr uint32_t* count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(snapshot_count_sem));
        noc_semaphore_wait_min(count, 100);
        noc_semaphore_set(count, 0);
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(snapshot_go_sem));
        noc_semaphore_set(go, 1);
        for (uint32_t target = 1; target < 100; ++target) {{
            noc_async_write(
                get_semaphore(snapshot_go_sem),
                get_noc_addr(
                    kGroupedPhysicalX[target % 10],
                    grouped_physical_y(target / 10),
                    get_semaphore(snapshot_go_sem)),
                sizeof(uint32_t));
        }}
        noc_async_write_barrier();
    }}
    volatile tt_l1_ptr uint32_t* snapshot_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(snapshot_go_sem));
    noc_semaphore_wait(snapshot_go, 1);
    noc_semaphore_set(snapshot_go, 0);

#define FULL_READ_RESIDENT_DY(page_expr, destination_expr) do {{ \\
    const uint32_t full_page = (page_expr); \\
    const uint32_t producer = full_page / {down_jobs_per_node}; \\
    const uint32_t activation_node = producer % 10; \\
    uint32_t local_index = full_page - producer * {down_jobs_per_node}; \\
    for (uint32_t prior = activation_node; prior < producer; prior += 10) {{ \\
        const uint32_t first = prior * {down_jobs_per_node}; \\
        const uint32_t remaining = {down_total_jobs} - first; \\
        local_index += remaining < {down_jobs_per_node} \\
            ? remaining : {down_jobs_per_node}; \\
    }} \\
    const uint32_t bank = local_index / 32; \\
    const uint32_t slot = local_index - bank * 32; \\
    const uint32_t dgate_base = bank == 0 ? dgate_base0 : dgate_base1; \\
    const uint32_t dup_base = bank == 0 ? dup_base0 : dup_base1; \\
    const uint32_t base = member == 0 ? dgate_base : dup_base; \\
    noc_async_read( \\
        get_noc_addr(13, grouped_physical_y(activation_node), \\
        base + slot * kGroupedTileBytes), \\
        (destination_expr), kGroupedTileBytes); \\
}} while (0)

"""
    if scaled_gradient_banks:
        scaled_snapshot = f"""    // Exact BF16(dY * scale) is produced by the activation island.
    // Snapshot those bank addresses before semaphore IDs 7..10 are reused.
    const uint32_t scaled_dgate_base0 =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7));
    const uint32_t scaled_dup_base0 =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8));
    uint32_t scaled_dgate_base1 = 0;
    uint32_t scaled_dup_base1 = 0;
    if constexpr ({activation_banks} == 2) {{
        scaled_dgate_base1 =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9));
        scaled_dup_base1 =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
    }}

"""
        snapshot_marker = (
            "    // Every matmul core must snapshot the bank mailboxes"
        )
        if prelude.count(snapshot_marker) != 1:
            raise RuntimeError("activation address snapshot marker changed")
        prelude = prelude.replace(
            snapshot_marker, scaled_snapshot + snapshot_marker, 1
        )
        prelude += f"""#define FULL_READ_RESIDENT_SCALED_DY(page_expr, destination_expr) do {{ \\
    const uint32_t full_page = (page_expr); \\
    const uint32_t producer = full_page / {down_jobs_per_node}; \\
    const uint32_t activation_node = producer % 10; \\
    uint32_t local_index = full_page - producer * {down_jobs_per_node}; \\
    for (uint32_t prior = activation_node; prior < producer; prior += 10) {{ \\
        const uint32_t first = prior * {down_jobs_per_node}; \\
        const uint32_t remaining = {down_total_jobs} - first; \\
        local_index += remaining < {down_jobs_per_node} \\
            ? remaining : {down_jobs_per_node}; \\
    }} \\
    const uint32_t bank = local_index / 32; \\
    const uint32_t slot = local_index - bank * 32; \\
    const uint32_t dgate_base = bank == 0 \\
        ? scaled_dgate_base0 : scaled_dgate_base1; \\
    const uint32_t dup_base = bank == 0 \\
        ? scaled_dup_base0 : scaled_dup_base1; \\
    const uint32_t base = member == 0 ? dgate_base : dup_base; \\
    noc_async_read( \\
        get_noc_addr(13, grouped_physical_y(activation_node), \\
        base + slot * kGroupedTileBytes), \\
        (destination_expr), kGroupedTileBytes); \\
}} while (0)

"""
    if source.count(marker) != 1:
        raise RuntimeError("grouped reader ownership marker changed")
    source = source.replace(
        marker,
        "    const uint32_t group_y = member * 5;\n"
        + prelude
        + "    volatile tt_l1_ptr uint32_t* go_sem =",
        1,
    )

    patterns = (
        (
            """noc0.async_read(
                    dy,
                    CoreLocalMem<uint32_t>(
                        base_dy.get_write_ptr() + inner * kGroupedTileBytes),
                    kGroupedTileBytes,
                    {.page_id = m * n_tiles + n0 + inner},
                    {});""",
            """FULL_READ_RESIDENT_DY(
                    m * n_tiles + n0 + inner,
                    base_dy.get_write_ptr() + inner * kGroupedTileBytes);""",
        ),
        (
            """noc0.async_read(
                dy,
                CoreLocalMem<uint32_t>(
                    dh_dy.get_write_ptr() + n * kGroupedTileBytes),
                kGroupedTileBytes,
                {.page_id = node * n_tiles + n},
                {});""",
            """FULL_READ_RESIDENT_DY(
                node * n_tiles + n,
                dh_dy.get_write_ptr() + n * kGroupedTileBytes);""",
        ),
        (
            """noc0.async_read(
                dy,
                CoreLocalMem<uint32_t>(
                    db_dy.get_write_ptr() + m * kGroupedTileBytes),
                kGroupedTileBytes,
                {.page_id = m * n_tiles + n_tile},
                {});""",
            """FULL_READ_RESIDENT_DY(
                m * n_tiles + n_tile,
                db_dy.get_write_ptr() + m * kGroupedTileBytes);""",
        ),
    )
    for old, new in patterns:
        if source.count(old) != 1:
            raise RuntimeError("grouped dY read pattern changed")
        source = source.replace(old, new, 1)

    reset = """    for (uint32_t semaphore = 0; semaphore < 11; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }"""
    if source.count(reset) != 1:
        raise RuntimeError("grouped semaphore reset changed")
    # The reset remains in the same place, after the address snapshot.
    acknowledgement = """
    noc_async_read_barrier();
    noc_semaphore_inc(
        get_noc_addr(13, grouped_physical_y(0), get_semaphore(5)), 1);
    noc_async_atomic_barrier();
#undef FULL_READ_RESIDENT_DY
"""
    if scaled_gradient_banks:
        acknowledgement += "#undef FULL_READ_RESIDENT_SCALED_DY\n"
    stripped = source.rstrip()
    if not stripped.endswith("}"):
        raise RuntimeError("grouped reader no longer ends with a brace")
    return stripped[:-1] + acknowledgement + "\n}\n"


def _stream_grouped_dh_reduction(
    reader: str,
    compute: str,
    *,
    grouped_dh_arg_start: int,
) -> tuple[str, str]:
    """Stream the long-K grouped dH reduction through depth-two CBs.

    The activation island has already materialized the exact BF16(dY * scale)
    boundary.  Stream that value and B in two-tile blocks so the projection
    compute engine stays in one continuous matmul epoch.  The matmul order,
    FP32 partial spill and final BF16 pack remain unchanged.
    """

    b_arg = grouped_dh_arg_start + 1
    scaled_arg = grouped_dh_arg_start + 2
    cb_declaration = (
        "    constexpr uint32_t cb_dh_b = "
        f"get_compile_time_arg_val({b_arg});\n"
    )
    if reader.count(cb_declaration) != 1:
        raise RuntimeError("grouped dH B declaration changed")
    reader = reader.replace(
        cb_declaration,
        cb_declaration
        + "    constexpr uint32_t cb_dh_scaled = "
        + f"get_compile_time_arg_val({scaled_arg});\n",
        1,
    )
    cb_object = "    CircularBuffer dh_b(cb_dh_b);\n"
    if reader.count(cb_object) != 1:
        raise RuntimeError("grouped dH B object changed")
    reader = reader.replace(
        cb_object,
        cb_object + "    CircularBuffer dh_scaled(cb_dh_scaled);\n",
        1,
    )

    old_reader = """    if (node < m_tiles) {
        dh_dy.reserve_back(n_tiles);
        dh_b.reserve_back(n_tiles);
        for (uint32_t n = 0; n < n_tiles; ++n) {
            FULL_READ_RESIDENT_DY(
                node * n_tiles + n,
                dh_dy.get_write_ptr() + n * kGroupedTileBytes);
            noc0.async_read(
                b,
                CoreLocalMem<uint32_t>(
                    dh_b.get_write_ptr() + n * kGroupedTileBytes),
                kGroupedTileBytes,
                {.page_id = n},
                {});
        }
        noc0.async_read_barrier();
        dh_b.push_back(n_tiles);
        dh_dy.push_back(n_tiles);
    }
"""
    new_reader = """    if (node < m_tiles) {
        static_assert(n_tiles % 2 == 0, "grouped dH K must be /2");
        // Qwen has 76 blocks. Keep this as a runtime streaming loop: fully
        // unrolling repeated compute-format transitions creates enormous
        // BRISC code and pathological first-use compilation.
#pragma GCC unroll 1
        for (uint32_t block = 0; block < n_tiles / 2; ++block) {
            const uint32_t n0 = block * 2;
            dh_scaled.reserve_back(2);
            dh_b.reserve_back(2);
            for (uint32_t inner = 0; inner < 2; ++inner) {
                FULL_READ_RESIDENT_SCALED_DY(
                    node * n_tiles + n0 + inner,
                    dh_scaled.get_write_ptr() + inner * kGroupedTileBytes);
                noc0.async_read(
                    b,
                    CoreLocalMem<uint32_t>(
                        dh_b.get_write_ptr() + inner * kGroupedTileBytes),
                    kGroupedTileBytes,
                    {.page_id = n0 + inner},
                    {});
            }
            noc0.async_read_barrier();
            dh_b.push_back(2);
            dh_scaled.push_back(2);
        }
    }
"""
    if reader.count(old_reader) != 1:
        raise RuntimeError("grouped dH reader staging block changed")
    reader = reader.replace(old_reader, new_reader, 1)

    # Qwen has twelve idle nodes at the end of each 50-core member half.
    # Run the four long-K dH reductions on nodes 46..49 so no final-dX core
    # inherits their unpack/reduction history.  Coordination is deliberately
    # independent of the owners: node zero waits for four completed NoC
    # multicasts, then releases every consumer in its half.
    group_marker = "    const uint32_t group_y = member * 5;\n"
    if reader.count(group_marker) != 1:
        raise RuntimeError("grouped owner-placement marker changed")
    reader = reader.replace(
        group_marker,
        group_marker
        + "    constexpr uint32_t dh_owner_start = 46;\n"
        + "    const bool is_dh_owner = node >= dh_owner_start;\n"
        + "    const uint32_t dh_owner_index = node - dh_owner_start;\n",
        1,
    )
    owner_stream = "    if (node < m_tiles) {\n        static_assert(n_tiles % 2 == 0"
    if reader.count(owner_stream) != 1:
        raise RuntimeError("grouped streamed owner marker changed")
    reader = reader.replace(
        owner_stream,
        "    if (is_dh_owner) {\n        static_assert(n_tiles % 2 == 0",
        1,
    )
    owner_page = "                    node * n_tiles + n0 + inner,"
    if reader.count(owner_page) != 1:
        raise RuntimeError("grouped streamed owner page marker changed")
    reader = reader.replace(
        owner_page,
        "                    dh_owner_index * n_tiles + n0 + inner,",
        1,
    )

    residency_start_marker = (
        "    // Reserve the dedicated M-tile resident dH window.\n"
    )
    residency_end_marker = "    dh_resident.push_back(m_tiles);\n"
    residency_start = reader.find(residency_start_marker)
    residency_end = reader.find(residency_end_marker, residency_start)
    if residency_start < 0 or residency_end < 0:
        raise RuntimeError("resident grouped-dH protocol markers changed")
    residency_end += len(residency_end_marker)
    relocated_residency = """    // Reserve the complete local destination before any owner publishes.
    dh_resident.reserve_back(m_tiles);
    if (is_dh_owner) {
        dh_output.wait_front(1);
        const uint32_t source = dh_output.get_read_ptr();
        const uint32_t destination =
            dh_resident.get_write_ptr()
            + dh_owner_index * kGroupedTileBytes;
        for (uint32_t target = 0; target < kGroupedMemberNodes; ++target) {
            const uint32_t tx = target % kGroupedGridCols;
            const uint32_t ty = group_y + target / kGroupedGridCols;
            noc_async_write(
                source,
                get_noc_addr(
                    kGroupedPhysicalX[tx], grouped_physical_y(ty), destination),
                kGroupedTileBytes);
        }
        // The completion atomic becomes visible only after every dH payload.
        noc_async_write_barrier();
        noc_semaphore_inc(
            get_noc_addr(
                kGroupedPhysicalX[0], grouped_physical_y(group_y),
                get_semaphore(0)),
            1);
        noc_async_atomic_barrier();
        dh_output.pop_front(1);
    }

    if (node == 0) {
        volatile tt_l1_ptr uint32_t* completed =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait_min(completed, m_tiles);
        noc_semaphore_set(completed, 0);
        volatile tt_l1_ptr uint32_t* local_go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        noc_semaphore_set(local_go, 1);
        for (uint32_t target = 1; target < kGroupedMemberNodes; ++target) {
            const uint32_t tx = target % kGroupedGridCols;
            const uint32_t ty = group_y + target / kGroupedGridCols;
            noc_async_write(
                get_semaphore(1),
                get_noc_addr(
                    kGroupedPhysicalX[tx], grouped_physical_y(ty),
                    get_semaphore(1)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    volatile tt_l1_ptr uint32_t* resident_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    noc_semaphore_wait(resident_go, 1);
    noc_semaphore_set(resident_go, 0);
    dh_resident.push_back(m_tiles);
"""
    reader = (
        reader[:residency_start]
        + relocated_residency
        + reader[residency_end:]
    )

    phase_start = (
        "    // Phase 1: one complete output-feature reduction per token tile.\n"
    )
    phase_end = (
        "    // Phase 2: one complete token reduction per output-feature tile.\n"
    )
    start = compute.find(phase_start)
    end = compute.find(phase_end, start + len(phase_start))
    if start < 0 or end < 0:
        raise RuntimeError("grouped dH compute phase markers changed")
    streamed_phase = """    // Phase 1: one complete output-feature reduction per token tile.
    // Reader and compute exchange one exact pre-scaled two-tile reduction
    // block at a time; no long-K intermediate row is resident.
    for (uint32_t local_m = 0; local_m < m_tiles_per_node; ++local_m) {
        const uint32_t m_tile = node_index >= 46
            ? (node_index - 46) * m_tiles_per_node + local_m
            : total_m_tiles;
        if (m_tile >= total_m_tiles) {
            continue;
        }

        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_dh_scaled, cb_dh_b, cb_dh_partial);
        matmul_block_init(
            cb_dh_scaled,
            cb_dh_b,
            false,
            1,
            1,
            reduction_block_tiles);

#pragma GCC unroll 1
        for (uint32_t block = 0; block < num_n_blocks; ++block) {
            dh_scaled.wait_front(reduction_block_tiles);
            dh_b.wait_front(reduction_block_tiles);
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_dh_b, cb_dh_partial);
                dh_partial.wait_front(1);
                copy_block_matmul_partials(cb_dh_partial, 0, 0, 1);
                dh_partial.pop_front(1);
                reconfig_data_format_srca(cb_dh_partial, cb_dh_b);
                matmul_block_init(
                    cb_dh_scaled,
                    cb_dh_b,
                    false,
                    1,
                    1,
                    reduction_block_tiles);
            }
            for (uint32_t inner = 0; inner < reduction_block_tiles; ++inner) {
                matmul_block(
                    cb_dh_scaled,
                    cb_dh_b,
                    inner,
                    inner,
                    0,
                    false,
                    1,
                    1,
                    reduction_block_tiles);
            }

            tile_regs_commit();
            const bool last = block == num_n_blocks - 1;
            const uint32_t target_cb =
                last ? cb_dh_output : cb_dh_partial;
            if (last) {
                dh_output.reserve_back(1);
            } else {
                dh_partial.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(0, target_cb, 1);
            tile_regs_release();
            if (last) {
                dh_output.push_back(1);
            } else {
                dh_partial.push_back(1);
            }
            dh_b.pop_front(reduction_block_tiles);
            dh_scaled.pop_front(reduction_block_tiles);
        }
    }

"""
    compute = compute[:start] + streamed_phase + compute[end:]

    return reader, compute


def _grouped_reader_dram_diagnostic(source: str) -> str:
    """Compose the grouped phase while temporarily sourcing dY from DRAM."""

    source = source.replace("kGridCols", "kGroupedGridCols")
    source = source.replace("kMemberNodes", "kGroupedMemberNodes")
    source = source.replace("kPhysicalX", "kGroupedPhysicalX")
    source = source.replace("kTileBytes", "kGroupedTileBytes")
    source = re.sub(r"\bphysical_y\b", "grouped_physical_y", source)
    source = source.replace(
        "void kernel_main()", "FORCE_INLINE void full_grouped_reader_phase()", 1
    )
    marker = (
        "    const uint32_t group_y = member * 5;\n"
        "    volatile tt_l1_ptr uint32_t* go_sem ="
    )
    wait = """    // Preserve the real phase boundary so grouped semaphore
    // reuse cannot race the preceding dProduct/activation handoff.
    volatile tt_l1_ptr uint32_t* activation_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    noc_semaphore_wait(activation_ready, 1);
    noc_semaphore_inc(
        get_noc_addr(kGroupedPhysicalX[0], grouped_physical_y(0),
                     get_semaphore(12)), 1);
    noc_async_atomic_barrier();
    if (global_node == 0) {
        volatile tt_l1_ptr uint32_t* count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
        noc_semaphore_wait_min(count, 100);
        noc_semaphore_set(count, 0);
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13));
        noc_semaphore_set(go, 1);
        for (uint32_t target = 1; target < 100; ++target) {
            noc_async_write(
                get_semaphore(13),
                get_noc_addr(kGroupedPhysicalX[target % 10],
                             grouped_physical_y(target / 10),
                             get_semaphore(13)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    volatile tt_l1_ptr uint32_t* snapshot_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13));
    noc_semaphore_wait(snapshot_go, 1);
    noc_semaphore_set(snapshot_go, 0);
"""
    if source.count(marker) != 1:
        raise RuntimeError("grouped DRAM diagnostic marker changed")
    source = source.replace(
        marker,
        "    const uint32_t group_y = member * 5;\n"
        + wait
        + "    volatile tt_l1_ptr uint32_t* go_sem =",
        1,
    )
    stripped = source.rstrip()
    acknowledgement = """
    noc_async_read_barrier();
    noc_semaphore_inc(
        get_noc_addr(13, grouped_physical_y(0), get_semaphore(5)), 1);
    noc_async_atomic_barrier();
"""
    return stripped[:-1] + acknowledgement + "\n}\n"


def _handoff_only_grouped_reader(*, snapshot_barrier: bool = False) -> str:
    barrier = (
        ""
        if not snapshot_barrier
        else """
    noc_semaphore_inc(
        get_noc_addr(1, handoff_only_physical_y(0), get_semaphore(12)), 1);
    noc_async_atomic_barrier();
    const uint32_t global_node =
        get_absolute_logical_y() * 10 + get_absolute_logical_x();
    if (global_node == 0) {
        volatile tt_l1_ptr uint32_t* count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
        noc_semaphore_wait_min(count, 100);
        noc_semaphore_set(count, 0);
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13));
        noc_semaphore_set(go, 1);
        constexpr uint8_t px[10] = {1,2,3,4,5,6,7,10,11,12};
        for (uint32_t target = 1; target < 100; ++target) {
            noc_async_write(
                get_semaphore(13),
                get_noc_addr(px[target % 10],
                             handoff_only_physical_y(target / 10),
                             get_semaphore(13)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    volatile tt_l1_ptr uint32_t* snapshot_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13));
    noc_semaphore_wait(snapshot_go, 1);
    noc_semaphore_set(snapshot_go, 0);
"""
    )
    return (
        """
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
FORCE_INLINE uint8_t handoff_only_physical_y(uint32_t y) {
    return static_cast<uint8_t>(y + 2);
}
FORCE_INLINE void full_grouped_reader_phase() {
    volatile tt_l1_ptr uint32_t* ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    noc_semaphore_wait(ready, 1);
"""
        + barrier
        + """
    noc_semaphore_inc(
        get_noc_addr(13, handoff_only_physical_y(0), get_semaphore(5)), 1);
    noc_async_atomic_barrier();
}
"""
    )


def _compose_reader(down: str, grouped: str) -> str:
    down = down.replace(
        "void kernel_main()", "FORCE_INLINE void full_down_reader_phase()", 1
    )
    return (
        down
        + "\n"
        + grouped
        + "\nvoid kernel_main() {\n    full_down_reader_phase();\n    full_grouped_reader_phase();\n}\n"
    )


def _compose_compute(down: str, grouped: str, *, m_tiles: int) -> str:
    down = down.replace(
        "void kernel_main()", "FORCE_INLINE void full_down_compute_phase()", 1
    )
    release = f"""
    CircularBuffer full_down_resident_dh(get_compile_time_arg_val(16));
    full_down_resident_dh.pop_front({m_tiles});
"""
    marker = "    resident_down_tail();\n}"
    if down.count(marker) != 1:
        raise RuntimeError("down compute release marker changed")
    down = down.replace(marker, "    resident_down_tail();" + release + "}", 1)
    return (
        down
        + "\n"
        + _prefix_grouped_compute(grouped)
        + "\nvoid kernel_main() {\n    full_down_compute_phase();\n    full_grouped_compute_phase();\n}\n"
    )


def _compose_writer(
    down: str,
    grouped: str,
    *,
    m_tiles: int,
    intermediate_tiles: int,
    hidden_tiles: int,
    total_cbs: int,
    diagnostic_shared_output: bool = False,
) -> str:
    def output_accessor(
        name: str, *, tensor_index: int, common_index: int, tile_bytes: str
    ) -> str:
        return f"""    const uint32_t {name}_addr =
        get_common_arg_val<uint32_t>({common_index});
    const auto {name} = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                {tensor_index}, {total_cbs}>(),
            {common_index}>(),
        {name}_addr,
        {tile_bytes});"""

    arena_declaration = re.compile(
        r"    const uint32_t arena_addr = get_common_arg_val<uint32_t>\(0\);\n"
        r"    const auto arena = TensorAccessor\(.*?\n"
        r"        arena_addr,\n"
        r"        (?:kTileBytes|kGroupedWriterTileBytes)\);",
        re.DOTALL,
    )
    down = down.replace(
        "void kernel_main()", "FORCE_INLINE void full_down_writer_phase()", 1
    )
    down_output_accessors = [
        output_accessor(
            "shared_output",
            tensor_index=17,
            common_index=0,
            tile_bytes="kTileBytes",
        ),
        output_accessor(
            "down_a_output",
            tensor_index=22,
            common_index=5,
            tile_bytes="kTileBytes",
        ),
        output_accessor(
            "down_b_output",
            tensor_index=23,
            common_index=6,
            tile_bytes="kTileBytes",
        ),
    ]
    down_outputs = "\n".join(down_output_accessors)
    down, substitutions = arena_declaration.subn(down_outputs, down, count=1)
    if substitutions != 1:
        raise RuntimeError("down writer arena declaration changed")
    down = down.replace(
        "            arena,\n            kTileBytes,\n            {},\n"
        "            {.page_id = (2 * m_tiles + 1 + n) * arena_cols}",
        "            down_b_output,\n            kTileBytes,\n            {},\n"
        "            {.page_id = n}",
        1,
    ).replace(
        "            arena,\n            kTileBytes,\n            {},\n"
        "            {.page_id = 2 * m_tiles * arena_cols + k}",
        "            down_a_output,\n            kTileBytes,\n            {},\n"
        "            {.page_id = k}",
        1,
    )
    if "            arena,\n" in down:
        raise RuntimeError("down writer still targets the shared arena")

    grouped = grouped.replace("kGridCols", "kGroupedWriterGridCols")
    grouped = grouped.replace("kMemberNodes", "kGroupedWriterMemberNodes")
    grouped = grouped.replace("kPhysicalX", "kGroupedWriterPhysicalX")
    grouped = grouped.replace("kTileBytes", "kGroupedWriterTileBytes")
    grouped = re.sub(r"\bphysical_y\b", "grouped_writer_physical_y", grouped)
    grouped = grouped.replace(
        "void kernel_main()", "FORCE_INLINE void full_grouped_writer_phase()", 1
    )
    grouped_outputs = "\n".join(
        output_accessor(
            name,
            tensor_index=index,
            common_index=index - 17,
            tile_bytes="kGroupedWriterTileBytes",
        )
        for name, index in (
            ("input_gradient_output", 17),
            ("gate_a_output", 18),
            ("gate_b_output", 19),
            ("up_a_output", 20),
            ("up_b_output", 21),
        )
    )
    grouped, substitutions = arena_declaration.subn(grouped_outputs, grouped, count=1)
    if substitutions != 1:
        raise RuntimeError("grouped writer arena declaration changed")
    db_write = """            noc1.async_write(
                CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                arena,
                kGroupedWriterTileBytes,
                {},
                {.page_id = row * arena_cols});"""
    db_direct = """            if (member == 0) {
                noc1.async_write(
                    CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                    gate_b_output, kGroupedWriterTileBytes, {},
                    {.page_id = n_tile});
            } else {
                noc1.async_write(
                    CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                    up_b_output, kGroupedWriterTileBytes, {},
                    {.page_id = n_tile});
            }"""
    if grouped.count(db_write) != 1:
        raise RuntimeError("grouped dB arena write changed")
    grouped = grouped.replace(db_write, db_direct, 1)
    da_write = """            noc1.async_write(
                CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                arena,
                kGroupedWriterTileBytes,
                {},
                {.page_id = row * arena_cols + node});"""
    da_direct = """            if (member == 0) {
                noc1.async_write(
                    CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                    gate_a_output, kGroupedWriterTileBytes, {},
                    {.page_id = node});
            } else {
                noc1.async_write(
                    CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                    up_a_output, kGroupedWriterTileBytes, {},
                    {.page_id = node});
            }"""
    if grouped.count(da_write) != 1:
        raise RuntimeError("grouped dA arena write changed")
    grouped = grouped.replace(da_write, da_direct, 1)
    dx_write = """                    arena,
                    kGroupedWriterTileBytes,
                    {},
                    {.page_id = job}"""
    dx_direct = """                    input_gradient_output,
                    kGroupedWriterTileBytes,
                    {},
                    {.page_id = job}"""
    if grouped.count(dx_write) != 1:
        raise RuntimeError("grouped dX arena write changed")
    grouped = grouped.replace(dx_write, dx_direct, 1)
    if "                arena,\n" in grouped:
        raise RuntimeError("grouped writer still targets the shared arena")
    if diagnostic_shared_output:
        gate_da_row = m_tiles
        up_da_row = gate_da_row + 1
        down_da_row = up_da_row + 1
        gate_db_row = down_da_row + 1
        up_db_row = gate_db_row + intermediate_tiles
        down_db_row = up_db_row + intermediate_tiles
        down = down.replace(
            "down_b_output,\n            kTileBytes,\n            {},\n"
            "            {.page_id = n}",
            "shared_output,\n            kTileBytes,\n            {},\n"
            f"            {{.page_id = ({down_db_row} + n) * {intermediate_tiles}}}",
            1,
        ).replace(
            "down_a_output,\n            kTileBytes,\n            {},\n"
            "            {.page_id = k}",
            "shared_output,\n            kTileBytes,\n            {},\n"
            f"            {{.page_id = {down_da_row} * {intermediate_tiles} + k}}",
            1,
        )
        grouped = (
            grouped.replace(
                "gate_b_output, kGroupedWriterTileBytes, {},\n"
                "                    {.page_id = n_tile}",
                "input_gradient_output, kGroupedWriterTileBytes, {},\n"
                f"                    {{.page_id = ({gate_db_row} + n_tile) * {intermediate_tiles}}}",
                1,
            )
            .replace(
                "up_b_output, kGroupedWriterTileBytes, {},\n"
                "                    {.page_id = n_tile}",
                "input_gradient_output, kGroupedWriterTileBytes, {},\n"
                f"                    {{.page_id = ({up_db_row} + n_tile) * {intermediate_tiles}}}",
                1,
            )
            .replace(
                "gate_a_output, kGroupedWriterTileBytes, {},\n"
                "                    {.page_id = node}",
                "input_gradient_output, kGroupedWriterTileBytes, {},\n"
                f"                    {{.page_id = {gate_da_row} * {intermediate_tiles} + node}}",
                1,
            )
            .replace(
                "up_a_output, kGroupedWriterTileBytes, {},\n"
                "                    {.page_id = node}",
                "input_gradient_output, kGroupedWriterTileBytes, {},\n"
                f"                    {{.page_id = {up_da_row} * {intermediate_tiles} + node}}",
                1,
            )
            .replace(
                "{.page_id = job}",
                "{.page_id = (job / "
                f"{hidden_tiles}) * {intermediate_tiles} + (job % {hidden_tiles})}}",
                1,
            )
        )
    return (
        down
        + "\n"
        + grouped
        + "\nvoid kernel_main() {\n    full_down_writer_phase();\n    full_grouped_writer_phase();\n}\n"
    )


def _inject_down_h_recompute(
    reader: str,
    compute: str,
    *,
    product_cb: int,
    a_cb: int,
    partial_cb: int,
    output_cb: int,
    gate_cb: int,
    up_cb: int,
    sigmoid_cb: int,
    silu_cb: int,
    gate_saved_cb: int,
    m_tiles: int,
    k_tiles: int,
    h_tiles: int,
    da_k_tiles_per_node: int,
) -> tuple[str, str]:
    """Recompute the SwiGLU product and down rank-H entirely in L1."""

    marker = "    constexpr uint32_t cb_tail_x = " "get_compile_time_arg_val(18);\n"
    declarations = (
        f"    constexpr uint32_t cb_recompute_product = "
        f"get_compile_time_arg_val({product_cb});\n"
        f"    constexpr uint32_t cb_recompute_a = "
        f"get_compile_time_arg_val({a_cb});\n"
        f"    constexpr uint32_t cb_recompute_h = "
        f"get_compile_time_arg_val({output_cb});\n"
        f"    constexpr uint32_t cb_recompute_gate = "
        f"get_compile_time_arg_val({gate_cb});\n"
        f"    constexpr uint32_t cb_recompute_up = "
        f"get_compile_time_arg_val({up_cb});\n"
        f"    constexpr uint32_t cb_recompute_gate_saved = "
        f"get_compile_time_arg_val({gate_saved_cb});\n"
    )
    if reader.count(marker) != 1:
        raise RuntimeError("down reader CB declaration marker changed")
    reader = reader.replace(marker, marker + declarations, 1)

    marker = "    CircularBuffer tail_x(cb_tail_x);\n"
    buffers = (
        "    CircularBuffer recompute_product(cb_recompute_product);\n"
        "    CircularBuffer recompute_a(cb_recompute_a);\n"
        "    CircularBuffer recompute_h(cb_recompute_h);\n"
        "    CircularBuffer recompute_gate(cb_recompute_gate);\n"
        "    CircularBuffer recompute_up(cb_recompute_up);\n"
        "    CircularBuffer recompute_gate_saved(cb_recompute_gate_saved);\n"
    )
    if reader.count(marker) != 1:
        raise RuntimeError("down reader CB object marker changed")
    reader = reader.replace(marker, marker + buffers, 1)

    # ID 14 is private to this pre-phase. Every invocation explicitly clears
    # it in the existing reader reset loop before any H work is published.
    marker = "    // Tensor order: dY, W, B, H, A, X.\n"
    if reader.count(marker) != 1:
        raise RuntimeError("down reader tensor marker changed")

    marker = "    // Frozen-base dProduct input stream.\n"
    arena_tiles = 2 * k_tiles + 2 + h_tiles
    prephase = """    // Recreate product = BF16(BF16(gate * sigmoid(gate)) * up)
    // from the packed forward checkpoint. No product tile crosses DRAM.
    if (node < n_tiles) {
        db_hidden.reserve_back(m_tiles);
    }
    if (node < m_tiles) {
        for (uint32_t i = 0; i < k_tiles; i += 2) {
            recompute_gate.reserve_back(4);
            recompute_up.reserve_back(2);
            for (uint32_t inner = 0; inner < 2; ++inner) {
                noc0.async_read(
                    x,
                    CoreLocalMem<uint32_t>(
                        recompute_gate.get_write_ptr()
                        + (2 * inner) * kTileBytes),
                    kTileBytes,
                    {.page_id = node * @ARENA_TILES@ + i + inner}, {});
                noc0.async_read(
                    x,
                    CoreLocalMem<uint32_t>(
                        recompute_up.get_write_ptr()
                        + inner * kTileBytes),
                    kTileBytes,
                    {.page_id = node * @ARENA_TILES@ + k_tiles + i + inner}, {});
                noc0.async_read(
                    x,
                    CoreLocalMem<uint32_t>(
                        recompute_gate.get_write_ptr()
                        + (2 * inner + 1) * kTileBytes),
                    kTileBytes,
                    {.page_id = node * @ARENA_TILES@ + i + inner}, {});
            }
            noc0.async_read_barrier();
            recompute_gate.push_back(4);
            recompute_up.push_back(2);
        }
    }

    // dA owners need one complete token column of the same product. Feed
    // those tiles into CB18, now compute-produced rather than reader-produced.
    for (uint32_t local_k = 0; local_k < da_k_tiles_per_node; ++local_k) {
        const uint32_t k = node * da_k_tiles_per_node + local_k;
        if (k >= k_tiles) continue;
        recompute_gate.reserve_back(2 * m_tiles);
        recompute_up.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                x,
                CoreLocalMem<uint32_t>(
                    recompute_gate.get_write_ptr() + 2 * m * kTileBytes),
                kTileBytes,
                {.page_id = m * @ARENA_TILES@ + k}, {});
            noc0.async_read(
                x,
                CoreLocalMem<uint32_t>(
                    recompute_up.get_write_ptr() + m * kTileBytes),
                kTileBytes,
                {.page_id = m * @ARENA_TILES@ + k_tiles + k}, {});
            noc0.async_read(
                x,
                CoreLocalMem<uint32_t>(
                    recompute_gate.get_write_ptr()
                    + (2 * m + 1) * kTileBytes),
                kTileBytes,
                {.page_id = m * @ARENA_TILES@ + k}, {});
        }
        noc0.async_read_barrier();
        recompute_gate.push_back(2 * m_tiles);
        recompute_up.push_back(m_tiles);
    }

    // A is intentionally streamed last. Its CB is small, so filling it
    // before the dA product inputs would deadlock the reader against compute.
    if (node < m_tiles) {
        for (uint32_t i = 0; i < k_tiles; i += 2) {
            recompute_a.reserve_back(2);
            for (uint32_t inner = 0; inner < 2; ++inner) {
                noc0.async_read(
                    a,
                    CoreLocalMem<uint32_t>(
                        recompute_a.get_write_ptr() + inner * kTileBytes),
                    kTileBytes, {.page_id = i + inner}, {});
            }
            noc0.async_read_barrier();
            recompute_a.push_back(2);
        }
    }

    // H = product @ A.T: token-row owners reproduce the accepted two-K-tile
    // spill schedule, then multicast H into the dB consumers' L1 windows.
    if (node < m_tiles) {
        recompute_h.wait_front(1);
        for (uint32_t target = 0; target < n_tiles; ++target) {
            noc_async_write(
                recompute_h.get_read_ptr(),
                get_noc_addr(
                    kPhysicalX[target % kGridCols],
                    physical_y(target / kGridCols),
                    db_hidden.get_write_ptr() + node * kTileBytes),
                kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t target = 0; target < n_tiles; ++target) {
            noc_semaphore_inc(
                get_noc_addr(
                    kPhysicalX[target % kGridCols],
                    physical_y(target / kGridCols), get_semaphore(14)),
                1);
        }
        noc_async_atomic_barrier();
        recompute_h.pop_front(1);
    }
    if (node < n_tiles) {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        noc_semaphore_wait_min(ready, m_tiles);
        noc_semaphore_set(ready, 0);
        db_hidden.push_back(m_tiles);
    }

""".replace("@ARENA_TILES@", str(arena_tiles)).replace(
        "@DA_K_TILES_PER_NODE@", str(da_k_tiles_per_node)
    )
    if reader.count(marker) != 1:
        raise RuntimeError("down reader first stream marker changed")
    reader = reader.replace(marker, prephase + marker, 1)

    old_db = """        db_dy.reserve_back(m_tiles);
        db_hidden.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                dy,
                CoreLocalMem<uint32_t>(db_dy.get_write_ptr() + m * kTileBytes),
                kTileBytes, {.page_id = m * n_tiles + n}, {});
            noc0.async_read(
                hidden,
                CoreLocalMem<uint32_t>(db_hidden.get_write_ptr() + m * kTileBytes),
                kTileBytes, {.page_id = m}, {});
        }
        noc0.async_read_barrier();
        db_hidden.push_back(m_tiles);
        db_dy.push_back(m_tiles);"""
    new_db = """        db_dy.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                dy,
                CoreLocalMem<uint32_t>(db_dy.get_write_ptr() + m * kTileBytes),
                kTileBytes, {.page_id = m * n_tiles + n}, {});
        }
        noc0.async_read_barrier();
        db_dy.push_back(m_tiles);"""
    if reader.count(old_db) != 1:
        raise RuntimeError("down dB reader block changed")
    reader = reader.replace(old_db, new_db, 1)

    old_da = """        tail_x.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                x,
                CoreLocalMem<uint32_t>(tail_x.get_write_ptr() + m * kTileBytes),
                kTileBytes, {.page_id = m * k_tiles + k}, {});
        }
        noc0.async_read_barrier();
        tail_x.push_back(m_tiles);"""
    if reader.count(old_da) != 1:
        raise RuntimeError("down dA product reader block changed")
    reader = reader.replace(
        old_da,
        "        // Product tiles are recreated by the compute RISC in CB18.",
        1,
    )

    helper = f"""
ALWI void full_mlp_bf16_unary_init(uint32_t input_id, uint32_t output_id) {{
#ifndef ARCH_QUASAR
    state_configure<Operand::SRCA, Operand::PACK>(
        input_id, output_id, __builtin_LINE());
    UNPACK((llk_unpack_hw_configure<false>(input_id)));
    UNPACK((llk_unpack_A_init<
            BroadcastType::NONE, false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(false, false, input_id)));
    PACK((llk_pack_hw_configure<false>(output_id)));
    PACK((llk_pack_init(output_id)));
    PACK((llk_pack_dest_init<false, PackMode::Default>()));
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D, false, BroadcastType::NONE>(input_id)));
    MATH((llk_math_pack_sync_init<false>()));
    MATH((llk_math_hw_configure<false>(input_id, input_id)));
    MATH((ckernel::math::_configure_unary_preserve_zero_flag_state_()));
#endif
}}

ALWI void full_mlp_bf16_copy_init(uint32_t input_id) {{
    UNPACK((llk_unpack_A_init<
            BroadcastType::NONE, false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(false, false, input_id)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D, false, BroadcastType::NONE>(input_id)));
}}

ALWI void full_mlp_bf16_copy_tile(
    uint32_t input_id, uint32_t input_tile, uint32_t destination_tile) {{
    UNPACK((llk_unpack_A<
            BroadcastType::NONE, false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(input_id, input_tile)));
    MATH((llk_math_eltwise_unary_datacopy<
          DataCopyType::A2D, false, BroadcastType::NONE,
          UnpackToDestEn>(destination_tile, input_id)));
}}

static __attribute__((noinline, noclone)) void full_mlp_exact_sigmoid_one(
    uint32_t input_id, uint32_t output_id) {{
    CircularBuffer input(input_id);
    CircularBuffer output(output_id);
    full_mlp_bf16_unary_init(input_id, output_id);
    input.wait_front(1);
    output.reserve_back(1);
    tile_regs_acquire();
    full_mlp_bf16_copy_init(input_id);
    full_mlp_bf16_copy_tile(input_id, 0, 0);
    MATH((SFPU_UNARY_INIT_FN(sigmoid, sfpu::sigmoid_init, (false))));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, false, calculate_sigmoid,
        (false, false, 8), 0, VectorMode::RC)));
    MATH((llk_math_dest_section_done<false>()));
    tile_regs_wait();
    PACK((llk_pack_reconfig_data_format<false>(output_id)));
    PACK((llk_pack<false, false, PackMode::Default>(0, output_id, 0)));
    PACK((llk_pack_dest_section_done<false>()));
    output.push_back(1);
    input.pop_front(1);
}}

static __attribute__((noinline, noclone)) void full_mlp_exact_multiply_one(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id) {{
    CircularBuffer lhs(lhs_id);
    CircularBuffer rhs(rhs_id);
    CircularBuffer output(output_id);
    full_mlp_bf16_unary_init(lhs_id, output_id);
    MATH((SFPU_BINARY_INIT_FN(
        unused, sfpu::sfpu_binary_init,
        (false, ckernel::BinaryOp::MUL))));
    lhs.wait_front(1);
    rhs.wait_front(1);
    output.reserve_back(1);
    tile_regs_acquire();
    full_mlp_bf16_copy_init(lhs_id);
    full_mlp_bf16_copy_tile(lhs_id, 0, 0);
    full_mlp_bf16_copy_init(rhs_id);
    full_mlp_bf16_copy_tile(rhs_id, 0, 1);
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, false, calculate_sfpu_binary_mul,
        (false, ckernel::BinaryOp::MUL, 8, false),
        0, 1, 0, VectorMode::RC)));
    MATH((llk_math_dest_section_done<false>()));
    tile_regs_wait();
    PACK((llk_pack_reconfig_data_format<false>(output_id)));
    PACK((llk_pack<false, false, PackMode::Default>(0, output_id, 0)));
    PACK((llk_pack_dest_section_done<false>()));
    output.push_back(1);
}}

static __attribute__((noinline, noclone)) void full_mlp_exact_product_one(
    uint32_t gate_id, uint32_t up_id, uint32_t sigmoid_id,
    uint32_t silu_id, uint32_t output_id) {{
    CircularBuffer gate(gate_id);
    CircularBuffer up(up_id);
    CircularBuffer sigmoid(sigmoid_id);
    CircularBuffer silu(silu_id);
    full_mlp_exact_sigmoid_one(gate_id, sigmoid_id);
    full_mlp_exact_multiply_one(gate_id, sigmoid_id, silu_id);
    sigmoid.pop_front(1);
    full_mlp_exact_multiply_one(silu_id, up_id, output_id);
    silu.pop_front(1);
    gate.pop_front(1);
    up.pop_front(1);
}}

FORCE_INLINE void full_recompute_swiglu_products_phase() {{
    constexpr uint32_t cb_gate = get_compile_time_arg_val({gate_cb});
    constexpr uint32_t cb_up = get_compile_time_arg_val({up_cb});
    constexpr uint32_t cb_sigmoid = get_compile_time_arg_val({sigmoid_cb});
    constexpr uint32_t cb_silu = get_compile_time_arg_val({silu_cb});
    constexpr uint32_t cb_gate_saved = get_compile_time_arg_val({gate_saved_cb});
    constexpr uint32_t cb_h_product = get_compile_time_arg_val({product_cb});
    constexpr uint32_t cb_da_product = get_compile_time_arg_val(18);
    constexpr uint32_t m_tiles = {m_tiles};
    constexpr uint32_t k_tiles = {k_tiles};
    constexpr uint32_t da_k_tiles_per_node = {da_k_tiles_per_node};
    const uint32_t node =
        get_absolute_logical_y() * 10 + get_absolute_logical_x();
    // Prevent the SFPI/LTO pipeline from cloning the complete exact SFPU
    // sequence once per compile-time tile. These values are invariant at
    // runtime, but volatile keeps the generated program as compact loops.
    volatile uint32_t runtime_k_tiles = k_tiles;
    volatile uint32_t runtime_m_tiles = m_tiles;
    volatile uint32_t runtime_da_tiles = da_k_tiles_per_node;
    if (node < m_tiles) {{
        for (uint32_t k = 0; k < runtime_k_tiles; ++k) {{
            full_mlp_exact_product_one(
                cb_gate, cb_up, cb_sigmoid, cb_silu, cb_h_product);
        }}
    }}
    for (uint32_t local_k = 0; local_k < runtime_da_tiles; ++local_k) {{
        const uint32_t k = node * da_k_tiles_per_node + local_k;
        if (k >= k_tiles) continue;
        for (uint32_t m = 0; m < runtime_m_tiles; ++m) {{
            full_mlp_exact_product_one(
                cb_gate, cb_up, cb_sigmoid, cb_silu, cb_da_product);
        }}
    }}
    // Restore the FP32 destination mode used by all following reductions.
    enable_fp32_dest_acc();
}}

FORCE_INLINE void full_recompute_down_h_phase() {{
    constexpr uint32_t cb_product = get_compile_time_arg_val({product_cb});
    constexpr uint32_t cb_a = get_compile_time_arg_val({a_cb});
    constexpr uint32_t cb_partial = get_compile_time_arg_val({partial_cb});
    constexpr uint32_t cb_h = get_compile_time_arg_val({output_cb});
    constexpr uint32_t m_tiles = {m_tiles};
    constexpr uint32_t k_tiles = {k_tiles};
    const uint32_t node =
        get_absolute_logical_y() * 10 + get_absolute_logical_x();
    // Every dA owner (not only the token-row H owners) must consume its
    // gate/up stream and populate CB18 before the common down tail starts.
    full_recompute_swiglu_products_phase();
    if (node >= m_tiles) return;
    CircularBuffer product(cb_product);
    CircularBuffer a(cb_a);
    CircularBuffer partial(cb_partial);
    CircularBuffer h(cb_h);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_product, cb_a, cb_partial);
    matmul_block_init(cb_product, cb_a, true, 1, 1, 2);
    for (uint32_t block = 0; block < k_tiles / 2; ++block) {{
        product.wait_front(2);
        a.wait_front(2);
        tile_regs_acquire();
        if (block > 0) {{
            copy_tile_to_dst_init_short_with_dt(cb_a, cb_partial);
            partial.wait_front(1);
            copy_tile(cb_partial, 0, 0);
            partial.pop_front(1);
            reconfig_data_format_srca(cb_partial, cb_a);
        }}
        matmul_block_init(cb_product, cb_a, true, 1, 1, 2);
        // Preserve the accepted per-rank dynamic-throttle reset even for
        // the single physical rank tile.
        matmul_block_init(cb_product, cb_a, true, 1, 1, 2);
        for (uint32_t inner = 0; inner < 2; ++inner) {{
            matmul_block(
                cb_product, cb_a, inner, inner, 0,
                true, 1, 1, 2);
        }}
        tile_regs_commit();
        const bool last = block == k_tiles / 2 - 1;
        if (last) {{
            h.reserve_back(1);
        }} else {{
            partial.reserve_back(1);
        }}
        tile_regs_wait();
        const uint32_t target = last ? cb_h : cb_partial;
        PACK((pack_reconfig_data_format(target)));
        pack_tile<true>(0, target, 0);
        tile_regs_release();
        if (last) {{
            h.push_back(1);
        }} else {{
            partial.push_back(1);
        }}
        a.pop_front(2);
        product.pop_front(2);
    }}
}}
"""
    marker = "void kernel_main() {\n"
    if compute.count(marker) != 1:
        raise RuntimeError("down compute kernel marker changed")
    compute = compute.replace(marker, helper + marker, 1)
    compute = compute.replace(
        marker, marker + "    full_recompute_down_h_phase();\n", 1
    )
    return reader, compute


def _matmul_product_from_activation_l1(
    reader: str, compute: str, *, m_tiles: int, k_tiles: int
) -> tuple[str, str]:
    """Replace local FP32-island SwiGLU with a BF16-island NoC handoff."""

    reset = "for (uint32_t semaphore = 0; semaphore < 16; ++semaphore)"
    if reader.count(reset) != 1:
        raise RuntimeError("matmul semaphore reset marker changed")
    # IDs 12..15 are cross-island protocol state. Activation core zero owns
    # their reset on every matmul core and publishes the go token afterward.
    reader = reader.replace(
        reset,
        "for (uint32_t semaphore = 0; semaphore < 12; ++semaphore)",
        1,
    )

    start_marker = "    // Recreate product = BF16(BF16(gate * sigmoid(gate)) * up)\n"
    end_marker = "    // H = product @ A.T: token-row owners reproduce"
    start = reader.find(start_marker)
    end = reader.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("matmul product prephase markers changed")
    handoff = f"""    // The 10-core BF16 activation island computes every product tile
    // once and writes it into these exact H-row / dA-column L1 windows.
    volatile tt_l1_ptr uint32_t* product_reset_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
    noc_semaphore_wait(product_reset_ready, 1);
    noc_semaphore_set(product_reset_ready, 0);

    if (node < m_tiles) {{
        recompute_product.reserve_back(k_tiles);
    }}
    uint32_t valid_da_tiles = 0;
    for (uint32_t local_k = 0; local_k < da_k_tiles_per_node; ++local_k) {{
        if (node * da_k_tiles_per_node + local_k < k_tiles) {{
            ++valid_da_tiles;
        }}
    }}
    if (valid_da_tiles != 0) {{
        tail_x.reserve_back(valid_da_tiles * m_tiles);
    }}

    // All destinations must be reserved before any activation writer starts.
    noc_semaphore_inc(
        get_noc_addr(kActivationPhysicalX, physical_y(0), get_semaphore(13)),
        1);
    noc_async_atomic_barrier();
    if (node == 0) {{
        volatile tt_l1_ptr uint32_t* h_address =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        volatile tt_l1_ptr uint32_t* da_address =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
        noc_semaphore_set(h_address, recompute_product.get_write_ptr());
        noc_semaphore_set(da_address, tail_x.get_write_ptr());
        noc_async_write(
            get_semaphore(14),
            get_noc_addr(kActivationPhysicalX, physical_y(0), get_semaphore(14)),
            sizeof(uint32_t));
        noc_async_write(
            get_semaphore(15),
            get_noc_addr(kActivationPhysicalX, physical_y(0), get_semaphore(15)),
            sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_set(h_address, 0);
        noc_semaphore_set(da_address, 0);
    }}

    volatile tt_l1_ptr uint32_t* products_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
    noc_semaphore_wait(products_ready, 1);
    noc_semaphore_set(products_ready, 0);
    if (node < m_tiles) {{
        recompute_product.push_back(k_tiles);
    }}
    if (valid_da_tiles != 0) {{
        tail_x.push_back(valid_da_tiles * m_tiles);
    }}

    // Stream down A only after the product handoff so its small CB cannot
    // block the reader before activation writers observe all destinations.
    if (node < m_tiles) {{
        for (uint32_t i = 0; i < k_tiles; i += 2) {{
            recompute_a.reserve_back(2);
            for (uint32_t inner = 0; inner < 2; ++inner) {{
                noc0.async_read(
                    a,
                    CoreLocalMem<uint32_t>(
                        recompute_a.get_write_ptr() + inner * kTileBytes),
                    kTileBytes, {{.page_id = i + inner}}, {{}});
            }}
            noc0.async_read_barrier();
            recompute_a.push_back(2);
        }}
    }}

"""
    reader = reader[:start] + handoff + reader[end:]
    call = "    full_recompute_swiglu_products_phase();\n"
    if compute.count(call) != 1:
        raise RuntimeError("matmul product compute call changed")
    compute = compute.replace(call, "", 1)
    return reader, compute


def _activation_scaled_gradient_compute_source(
    source: str,
    *,
    activation_banks: int,
    scaled_arg_start: int,
    scale_bits: int,
) -> str:
    """Retain exact BF16-scaled dGate/dUp beside the raw gradients."""

    include = '#include "api/compute/eltwise_unary/eltwise_unary.h"'
    if source.count(include) != 1:
        raise RuntimeError("activation unary include marker changed")
    source = source.replace(
        include,
        include
        + '\n#include "api/compute/eltwise_unary/binop_with_scalar.h"',
        1,
    )
    namespace_end = "}  // namespace\n\nvoid kernel_main()"
    if source.count(namespace_end) != 1:
        raise RuntimeError("activation helper namespace marker changed")
    helper = """
static __attribute__((noinline, noclone)) void
full_scale_resident_gradient_one(
    uint32_t input_id,
    uint32_t input_tile,
    uint32_t output_id,
    uint32_t scale_bits
) {
    CircularBuffer input(input_id);
    CircularBuffer output(output_id);
    // Raw dGate/dUp remain retained for frozen-base dX and dB. This indexed
    // consumer observes readiness but deliberately does not acknowledge it.
    input.wait_front(input_tile + 1);
    output.reserve_back(1);
    init_sfpu(input_id, output_id);
    tile_regs_acquire();
    copy_tile_init(input_id);
    copy_tile(input_id, input_tile, 0);
    binop_with_scalar_tile_init();
    volatile int32_t scalar = static_cast<int32_t>(scale_bits);
    mul_unary_tile(0, scalar);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

"""
    source = source.replace(namespace_end, helper + namespace_end, 1)

    if activation_banks == 1:
        scaled_gate = f"get_compile_time_arg_val({scaled_arg_start})"
        scaled_up = f"get_compile_time_arg_val({scaled_arg_start + 1})"
        local_index = "job"
    elif activation_banks == 2:
        scaled_gate = (
            "job < 32 ? "
            f"get_compile_time_arg_val({scaled_arg_start}) : "
            f"get_compile_time_arg_val({scaled_arg_start + 2})"
        )
        scaled_up = (
            "job < 32 ? "
            f"get_compile_time_arg_val({scaled_arg_start + 1}) : "
            f"get_compile_time_arg_val({scaled_arg_start + 3})"
        )
        local_index = "job < 32 ? job : job - 32"
    else:  # pragma: no cover - guarded by the planner
        raise ValueError("scaled activation handoff supports one or two banks")

    marker = """        sigmoid.pop_front(1);
        dproduct.pop_front(1);"""
    if source.count(marker) != 1:
        raise RuntimeError("activation gradient completion marker changed")
    scale = f"""        {{
            DeviceZoneScopedN("swiglu_backward_scaled_gradient_handoff");
            const uint32_t scaled_gate_cb = {scaled_gate};
            const uint32_t scaled_up_cb = {scaled_up};
            const uint32_t retained_tile = {local_index};
            full_scale_resident_gradient_one(
                cb_dgate, retained_tile, scaled_gate_cb, {scale_bits}u);
            full_scale_resident_gradient_one(
                cb_dup, retained_tile, scaled_up_cb, {scale_bits}u);
        }}

"""
    return source.replace(marker, scale + marker, 1)


def _activation_scaled_gradient_writer_source(
    source: str,
    *,
    activation_banks: int,
    scaled_arg_start: int,
) -> str:
    """Publish and retain scaled-gradient bank addresses with the raw banks."""

    declarations = """    CircularBuffer dgate0(cb_dgate0);
    CircularBuffer dup0(cb_dup0);
    dgate0.wait_front(bank0_jobs);
    dup0.wait_front(bank0_jobs);
    uint32_t dgate_addresses[2] = {dgate0.get_read_ptr(), 0};
    uint32_t dup_addresses[2] = {dup0.get_read_ptr(), 0};
"""
    if source.count(declarations) != 1:
        raise RuntimeError("activation writer bank declarations changed")
    scaled_declarations = declarations + f"""    CircularBuffer scaled_dgate0(
        get_compile_time_arg_val({scaled_arg_start}));
    CircularBuffer scaled_dup0(
        get_compile_time_arg_val({scaled_arg_start + 1}));
    scaled_dgate0.wait_front(bank0_jobs);
    scaled_dup0.wait_front(bank0_jobs);
    uint32_t scaled_dgate_addresses[2] = {{scaled_dgate0.get_read_ptr(), 0}};
    uint32_t scaled_dup_addresses[2] = {{scaled_dup0.get_read_ptr(), 0}};
"""
    source = source.replace(declarations, scaled_declarations, 1)

    bank_one = """        dgate_addresses[1] = dgate1.get_read_ptr();
        dup_addresses[1] = dup1.get_read_ptr();
"""
    if source.count(bank_one) != 1:
        raise RuntimeError("activation writer second-bank marker changed")
    if activation_banks == 2:
        source = source.replace(
            bank_one,
            bank_one
            + f"""        CircularBuffer scaled_dgate1(
            get_compile_time_arg_val({scaled_arg_start + 2}));
        CircularBuffer scaled_dup1(
            get_compile_time_arg_val({scaled_arg_start + 3}));
        scaled_dgate1.wait_front(bank1_jobs);
        scaled_dup1.wait_front(bank1_jobs);
        scaled_dgate_addresses[1] = scaled_dgate1.get_read_ptr();
        scaled_dup_addresses[1] = scaled_dup1.get_read_ptr();
""",
            1,
        )

    publication = """                noc_async_write(
                    get_semaphore(address_start + 2 * bank + 1),
                    get_noc_addr(
                        kMatmulPhysicalX[target_x], physical_y(target_y),
                        get_semaphore(address_start + 2 * bank + 1)),
                    sizeof(uint32_t));
"""
    if source.count(publication) != 1:
        raise RuntimeError("activation writer publication marker changed")
    source = source.replace(
        publication,
        publication
        + """                constexpr uint32_t scaled_address_start = 7;
                volatile tt_l1_ptr uint32_t* scaled_dgate_mailbox =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(scaled_address_start + 2 * bank));
                volatile tt_l1_ptr uint32_t* scaled_dup_mailbox =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(scaled_address_start + 2 * bank + 1));
                noc_semaphore_set(
                    scaled_dgate_mailbox, scaled_dgate_addresses[bank]);
                noc_semaphore_set(
                    scaled_dup_mailbox, scaled_dup_addresses[bank]);
                noc_async_write(
                    get_semaphore(scaled_address_start + 2 * bank),
                    get_noc_addr(
                        kMatmulPhysicalX[target_x], physical_y(target_y),
                        get_semaphore(scaled_address_start + 2 * bank)),
                    sizeof(uint32_t));
                noc_async_write(
                    get_semaphore(scaled_address_start + 2 * bank + 1),
                    get_noc_addr(
                        kMatmulPhysicalX[target_x], physical_y(target_y),
                        get_semaphore(scaled_address_start + 2 * bank + 1)),
                    sizeof(uint32_t));
""",
        1,
    )

    release = """    dgate0.pop_front(bank0_jobs);
    dup0.pop_front(bank0_jobs);
"""
    if source.count(release) != 1:
        raise RuntimeError("activation writer release marker changed")
    source = source.replace(
        release,
        release
        + "    scaled_dgate0.pop_front(bank0_jobs);\n"
        + "    scaled_dup0.pop_front(bank0_jobs);\n",
        1,
    )
    second_release = """        dgate1.pop_front(bank1_jobs);
        dup1.pop_front(bank1_jobs);
"""
    if source.count(second_release) != 1:
        raise RuntimeError("activation writer second release marker changed")
    if activation_banks == 2:
        source = source.replace(
            second_release,
            second_release
            + f"""        CircularBuffer scaled_dgate1(
            get_compile_time_arg_val({scaled_arg_start + 2}));
        CircularBuffer scaled_dup1(
            get_compile_time_arg_val({scaled_arg_start + 3}));
        scaled_dgate1.pop_front(bank1_jobs);
        scaled_dup1.pop_front(bank1_jobs);
""",
            1,
        )
    else:
        # The standalone compiler eliminates these two branches after
        # resolving ``bank_count == 1``.  The cross-owner source composer must
        # map compile-time CB arguments before C++ constant folding, however,
        # so dead references to the absent second activation bank would become
        # false ABI requirements.  Remove only the two balanced constexpr
        # blocks whose condition is statically false for this generated shape.
        marker = "    if constexpr (bank_count == 2) {"
        if source.count(marker) != 2:
            raise RuntimeError("one-bank activation writer branches changed")
        for _ in range(2):
            start = source.find(marker)
            brace = source.find("{", start)
            depth = 0
            end = -1
            for index in range(brace, len(source)):
                if source[index] == "{":
                    depth += 1
                elif source[index] == "}":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            if end < 0:
                raise RuntimeError("unterminated one-bank activation branch")
            if end < len(source) and source[end] == "\n":
                end += 1
            source = source[:start] + source[end:]
    return source


def _activation_product_sources(
    reader: str,
    compute: str,
    writer: str,
    *,
    total_cbs: int,
    m_tiles: int,
    i_tiles: int,
    da_k_tiles_per_node: int,
    gate_arg: int,
    gate_saved_arg: int,
    up_arg: int,
    sigmoid_arg: int,
    silu_arg: int,
    output_arg: int,
) -> tuple[str, str, str]:
    """Prepend exact product production/distribution to the BF16 island."""

    jobs = m_tiles * i_tiles
    jobs_per_node = -(-jobs // 10)
    arena_tiles = 2 * i_tiles + 2 + (0 if m_tiles < 0 else 0)

    # Adding checkpoint tensor 5 before the existing gate/up operands shifts
    # their reader common-argument slots from 0/1 to 1/2.
    reader = (
        reader.replace(
            "get_common_arg_val<uint32_t>(0)",
            "get_common_arg_val<uint32_t>(@GATE_COMMON@)",
        )
        .replace(
            "get_common_arg_val<uint32_t>(1)",
            "get_common_arg_val<uint32_t>(2)",
        )
        .replace("@GATE_COMMON@", "1")
    )
    reader = reader.replace(
        f"6, {total_cbs}>(),\n            0>()",
        f"6, {total_cbs}>(),\n            1>()",
    ).replace(
        f"7, {total_cbs}>(),\n            1>()",
        f"7, {total_cbs}>(),\n            2>()",
    )

    reset_end = """    volatile tt_l1_ptr uint32_t* init_ack =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(kInitAckSemaphore));
"""
    if reader.count(reset_end) != 1:
        raise RuntimeError("activation reset marker changed")
    activation_reset = "for (uint32_t semaphore = 0; semaphore < 16; ++semaphore)"
    if reader.count(activation_reset) != 1:
        raise RuntimeError("activation semaphore reset loop changed")
    reader = reader.replace(
        activation_reset,
        "for (uint32_t semaphore = 0; semaphore < 12; ++semaphore)",
        1,
    )

    # The actual row stride is substituted later by the caller because it
    # also owns hidden_tiles.
    prelude = f"""    // One coordinator owns cross-island reset. Other readers never
    // clear IDs 12..15, so a published go token cannot be erased by a race.
    if (activation_node == 0) {{
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12)),
            0);
        for (uint32_t target = 0; target < kActivationNodes; ++target) {{
            for (uint32_t semaphore = 12; semaphore < 16; ++semaphore) {{
                noc_async_write(
                    get_semaphore(12),
                    get_noc_addr(13, physical_y(target),
                                 get_semaphore(semaphore)),
                    sizeof(uint32_t));
            }}
        }}
        for (uint32_t target = 0; target < kProducerNodes; ++target) {{
            for (uint32_t semaphore = 12; semaphore < 16; ++semaphore) {{
                noc_async_write(
                    get_semaphore(12),
                    get_noc_addr(
                        kProducerPhysicalX[target % kProducerCols],
                        physical_y(target / kProducerCols),
                        get_semaphore(semaphore)),
                    sizeof(uint32_t));
            }}
        }}
        noc_async_write_barrier();
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
        noc_semaphore_set(go, 1);
        for (uint32_t target = 0; target < kActivationNodes; ++target) {{
            noc_async_write(
                get_semaphore(12),
                get_noc_addr(13, physical_y(target), get_semaphore(12)),
                sizeof(uint32_t));
        }}
        for (uint32_t target = 0; target < kProducerNodes; ++target) {{
            noc_async_write(
                get_semaphore(12),
                get_noc_addr(
                    kProducerPhysicalX[target % kProducerCols],
                    physical_y(target / kProducerCols), get_semaphore(15)),
                sizeof(uint32_t));
        }}
        noc_async_write_barrier();
    }}
    volatile tt_l1_ptr uint32_t* activation_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
    noc_semaphore_wait(activation_go, 1);
    noc_semaphore_set(activation_go, 0);

    constexpr uint32_t cb_product_gate = get_compile_time_arg_val({gate_arg});
    constexpr uint32_t cb_product_gate_saved =
        get_compile_time_arg_val({gate_saved_arg});
    constexpr uint32_t cb_product_up = get_compile_time_arg_val({up_arg});
    CircularBuffer product_gate(cb_product_gate);
    CircularBuffer product_gate_saved(cb_product_gate_saved);
    CircularBuffer product_up(cb_product_up);
    const uint32_t checkpoint_addr = get_common_arg_val<uint32_t>(0);
    const auto checkpoint = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                5, {total_cbs}>(), 0>(), checkpoint_addr, kTileBytes);
    constexpr uint32_t product_jobs = {jobs};
    constexpr uint32_t product_jobs_per_node = {jobs_per_node};
    constexpr uint32_t product_i_tiles = {i_tiles};
    constexpr uint32_t product_row_tiles = @PRODUCT_ROW_TILES@;
    const uint32_t product_first = activation_node * product_jobs_per_node;
    for (uint32_t local = 0; local < product_jobs_per_node; ++local) {{
        const uint32_t job = product_first + local;
        if (job >= product_jobs) continue;
        const uint32_t m = job / product_i_tiles;
        const uint32_t k = job - m * product_i_tiles;
        product_gate.reserve_back(1);
        product_gate_saved.reserve_back(1);
        product_up.reserve_back(1);
        noc0.async_read(
            checkpoint, CoreLocalMem<uint32_t>(product_gate.get_write_ptr()),
            kTileBytes, {{.page_id = m * product_row_tiles + k}}, {{}});
        noc0.async_read(
            checkpoint,
            CoreLocalMem<uint32_t>(product_gate_saved.get_write_ptr()),
            kTileBytes, {{.page_id = m * product_row_tiles + k}}, {{}});
        noc0.async_read(
            checkpoint, CoreLocalMem<uint32_t>(product_up.get_write_ptr()),
            kTileBytes,
            {{.page_id = m * product_row_tiles + product_i_tiles + k}}, {{}});
        noc0.async_read_barrier();
        product_gate.push_back(1);
        product_gate_saved.push_back(1);
        product_up.push_back(1);
    }}

"""
    reader = reader.replace(reset_end, prelude + reset_end, 1)

    product_compute = _cpp("swiglu_exact_compute.cpp")
    # The standalone diagnostic executes all local sigmoid tiles, then all
    # gate multiplies, then all up multiplies.  That schedule requires four
    # whole-island staging windows and exceeds per-core L1 for Qwen's 61-tile
    # activation chunks.  The operations have no cross-tile dependency, so
    # execute the same three BF16 pack boundaries tile-by-tile.  Reader,
    # compute and writer can then overlap through depth-two CBs while every
    # visible arithmetic/rounding boundary remains unchanged.
    product_main = product_compute.find("void kernel_main()")
    if product_main < 0:
        raise RuntimeError("native SwiGLU product kernel main changed")
    product_compute = product_compute[:product_main] + """void kernel_main() {
    constexpr uint32_t total_jobs = @TOTAL_JOBS@;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t grid_cols = @GRID_COLS@;
    const uint32_t node =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();
    const uint32_t first_job = node * jobs_per_node;
    const uint32_t remaining =
        first_job >= total_jobs ? 0 : total_jobs - first_job;
    const uint32_t local_jobs =
        remaining < jobs_per_node ? remaining : jobs_per_node;
    for (uint32_t job = 0; job < local_jobs; ++job) {
        run_sigmoid(cb_gate, cb_sigmoid, 1);
        run_sfpu_multiply(cb_gate_saved, cb_sigmoid, cb_silu, 1);
        run_sfpu_multiply(cb_silu, cb_up, cb_output, 1);
    }
}
"""
    product_compute = (
        product_compute.replace(
            "constexpr uint32_t cb_gate = 0;",
            f"constexpr uint32_t cb_gate = get_compile_time_arg_val({gate_arg});",
        )
        .replace(
            "constexpr uint32_t cb_gate_saved = 1;",
            "constexpr uint32_t cb_gate_saved = "
            f"get_compile_time_arg_val({gate_saved_arg});",
        )
        .replace(
            "constexpr uint32_t cb_up = 2;",
            f"constexpr uint32_t cb_up = get_compile_time_arg_val({up_arg});",
        )
        .replace(
            "constexpr uint32_t cb_sigmoid = 3;",
            f"constexpr uint32_t cb_sigmoid = get_compile_time_arg_val({sigmoid_arg});",
        )
        .replace(
            "constexpr uint32_t cb_silu = 4;",
            f"constexpr uint32_t cb_silu = get_compile_time_arg_val({silu_arg});",
        )
        .replace(
            "constexpr uint32_t cb_output = 5;",
            f"constexpr uint32_t cb_output = get_compile_time_arg_val({output_arg});",
        )
    )
    for key, value in {
        "@TOTAL_JOBS@": str(jobs),
        "@JOBS_PER_NODE@": str(jobs_per_node),
        "@GRID_COLS@": "1",
        # The full training graph's accepted down-gradient edge is the
        # explicitly decomposed BF16 schedule: sigmoid -> gate multiply ->
        # up multiply.  Builtin SiLU can produce the same final model output
        # after the following long-K down projection while still changing
        # dA/dB by one BF16 quantum, so preserve all three pack boundaries.
        "@STAGE_ID@": "3",
        "@STARTS_FP32@": "false",
    }.items():
        product_compute = product_compute.replace(key, value)
    for cb_name in (
        "cb_gate",
        "cb_gate_saved",
        "cb_up",
        "cb_sigmoid",
        "cb_silu",
        "cb_output",
    ):
        product_compute = re.sub(
            rf"\b{cb_name}\b",
            f"full_product_{cb_name}",
            product_compute,
        )
    product_node = """    const uint32_t node =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();"""
    if product_compute.count(product_node) != 1:
        raise RuntimeError("activation product node mapping changed")
    product_compute = product_compute.replace(
        product_node,
        "    const uint32_t node = get_absolute_logical_y();",
        1,
    )
    product_compute = product_compute.replace(
        "void kernel_main()",
        "FORCE_INLINE void full_activation_product_compute_phase()",
        1,
    )
    compute = compute.replace(
        "void kernel_main()",
        "FORCE_INLINE void full_activation_backward_compute_phase()",
        1,
    )
    compute = product_compute + "\n" + compute + """
void kernel_main() {
    full_activation_product_compute_phase();
    full_activation_backward_compute_phase();
}
"""

    writer = writer.replace(
        "void kernel_main()",
        "FORCE_INLINE void full_activation_backward_writer_phase()",
        1,
    )
    product_writer = f"""
FORCE_INLINE void full_activation_product_writer_phase() {{
    constexpr uint32_t cb_product = get_compile_time_arg_val({output_arg});
    constexpr uint32_t product_jobs = {jobs};
    constexpr uint32_t jobs_per_node = {jobs_per_node};
    constexpr uint32_t i_tiles = {i_tiles};
    constexpr uint32_t m_tiles = {m_tiles};
    constexpr uint32_t da_tiles = {da_k_tiles_per_node};
    constexpr uint32_t kTileBytes = 2048;
    constexpr uint8_t px[10] = {{1,2,3,4,5,6,7,10,11,12}};
    const uint32_t node = get_absolute_logical_y();
    CircularBuffer product(cb_product);

    volatile tt_l1_ptr uint32_t* destinations =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13));
    if (node == 0) {{
        noc_semaphore_wait_min(destinations, 100);
        noc_semaphore_set(destinations, 0);
        volatile tt_l1_ptr uint32_t* h_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        volatile tt_l1_ptr uint32_t* da_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
        noc_semaphore_wait_min(h_addr, 1);
        noc_semaphore_wait_min(da_addr, 1);
        for (uint32_t target = 1; target < 10; ++target) {{
            noc_async_write(
                get_semaphore(14),
                get_noc_addr(13, physical_y(target), get_semaphore(14)),
                sizeof(uint32_t));
            noc_async_write(
                get_semaphore(15),
                get_noc_addr(13, physical_y(target), get_semaphore(15)),
                sizeof(uint32_t));
        }}
        noc_async_write_barrier();
    }}
    volatile tt_l1_ptr uint32_t* h_addr_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
    volatile tt_l1_ptr uint32_t* da_addr_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
    noc_semaphore_wait_min(h_addr_flag, 1);
    noc_semaphore_wait_min(da_addr_flag, 1);
    const uint32_t h_base = *h_addr_flag;
    const uint32_t da_base = *da_addr_flag;
    const uint32_t first = node * jobs_per_node;
    for (uint32_t local = 0; local < jobs_per_node; ++local) {{
        const uint32_t job = first + local;
        if (job >= product_jobs) continue;
        const uint32_t m = job / i_tiles;
        const uint32_t k = job - m * i_tiles;
        product.wait_front(1);
        const uint32_t source = product.get_read_ptr();
        noc_async_write(
            source,
            get_noc_addr(px[m % 10], physical_y(m / 10),
                         h_base + k * kTileBytes),
            kTileBytes);
        const uint32_t owner = k / da_tiles;
        const uint32_t slot = (k % da_tiles) * m_tiles + m;
        noc_async_write(
            source,
            get_noc_addr(px[owner % 10], physical_y(owner / 10),
                         da_base + slot * kTileBytes),
            kTileBytes);
        noc_async_write_barrier();
        product.pop_front(1);
    }}
    noc_semaphore_inc(
        get_noc_addr(13, physical_y(0), get_semaphore(13)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {{
        noc_semaphore_wait_min(destinations, 10);
        noc_semaphore_set(destinations, 0);
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        noc_semaphore_set(ready, 1);
        for (uint32_t target = 0; target < 100; ++target) {{
            noc_async_write(
                get_semaphore(14),
                get_noc_addr(px[target % 10], physical_y(target / 10),
                             get_semaphore(14)),
                sizeof(uint32_t));
        }}
        noc_async_write_barrier();
    }}
}}
"""
    product_writer = product_writer.replace("physical_y(", "full_product_physical_y(")
    product_writer = """
#include <cstdint>
#include "api/compile_time_args.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
FORCE_INLINE uint8_t full_product_physical_y(uint32_t y) {
    return static_cast<uint8_t>(y + 2);
}
""" + product_writer
    writer = product_writer + "\n" + writer + """
void kernel_main() {
    full_activation_product_writer_phase();
    full_activation_backward_writer_phase();
}
"""
    return reader, compute, writer


@lru_cache(maxsize=16)
def build_tt_lang_full_mlp_backward(
    *,
    down_scale: float,
    gate_up_scale: float,
    tokens: int,
    hidden_features: int,
    intermediate_features: int,
    physical_rank: int,
    grouped_dy_from_activation_l1: bool = True,
    grouped_execution_stage: int = 1,
    diagnostic_shared_output: bool = False,
    stream_grouped_dh: bool = False,
    artifact_program_sequence: bool = False,
) -> Callable:
    if grouped_execution_stage not in (0, 1, 2):
        raise ValueError("grouped execution stage must be 0, 1, or 2")
    if stream_grouped_dh and not grouped_dy_from_activation_l1:
        raise ValueError("streamed grouped dH requires the resident activation-L1 path")
    error = full_mlp_backward_support_error(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        physical_rank=physical_rank,
    )
    if error is not None:
        raise ValueError(error)
    plan = plan_full_mlp_backward(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        physical_rank=physical_rank,
    )
    m_tiles = tokens // TILE_SIZE
    h_tiles = hidden_features // TILE_SIZE
    i_tiles = intermediate_features // TILE_SIZE
    down_jobs = plan.down_jobs_per_core
    down_total_jobs = m_tiles * i_tiles
    down_da_per_core = -(-i_tiles // _MATMUL_NODES)
    down_n_per_core = -(-h_tiles // _MATMUL_NODES)
    group_jobs = m_tiles * h_tiles
    group_jobs_per_core = -(-group_jobs // _MEMBER_NODES)
    group_n_per_core = -(-i_tiles // _MEMBER_NODES)
    activation_banks = plan.activation_banks_per_output
    activation_extra_cbs = 2 * (activation_banks - 1)
    grouped_transpose_arg = _DOWN_CBS + _ACTIVATION_CBS + activation_extra_cbs
    grouped_peer_arg = grouped_transpose_arg + 1
    grouped_dh_arg_start = grouped_peer_arg + 1
    # Down-H recomputation completes before grouped gate/up dH starts. Reuse
    # the latter's large K-row CBs instead of reserving a second copy of the
    # intermediate dimension. This is required for Qwen's 152 K tiles and is
    # safe because the grouped reader is device-barriered on activation-ready.
    recompute_product_arg = grouped_dh_arg_start
    recompute_a_arg = grouped_dh_arg_start + 1
    recompute_partial_arg = grouped_dh_arg_start + 3
    recompute_h_arg = grouped_dh_arg_start + 4
    # These are referenced only by the now-dead local-product helper source;
    # map them to valid reused arguments until that diagnostic code is fully
    # removed, without allocating any additional hardware CBs.
    recompute_gate_arg = grouped_dh_arg_start
    recompute_up_arg = grouped_dh_arg_start + 1
    recompute_sigmoid_arg = grouped_dh_arg_start + 2
    recompute_silu_arg = grouped_dh_arg_start + 3
    recompute_gate_saved_arg = grouped_dh_arg_start + 4
    activation_product_gate_arg = grouped_dh_arg_start + 5
    activation_product_gate_saved_arg = activation_product_gate_arg + 1
    activation_product_up_arg = activation_product_gate_saved_arg + 1
    activation_product_sigmoid_arg = activation_product_up_arg + 1
    activation_product_silu_arg = activation_product_sigmoid_arg + 1
    activation_product_output_arg = activation_product_silu_arg + 1
    activation_scaled_arg_start = activation_product_output_arg + 1
    activation_scaled_count = 2 * activation_banks if stream_grouped_dh else 0
    # In streamed mode grouped tail-A phase-aliases the now-empty down tail-A
    # FIFO.  It has identical reader->compute ownership and is exercised on
    # the complete grid, unlike the dH/B FIFOs touched only by four owner
    # cores.  It also stays below Blackhole's 32 compute data-format slots.
    grouped_tail_a_arg = 17
    total_cbs = activation_scaled_arg_start + activation_scaled_count

    down_scale_bf16, down_scale_bits = _scale_bits(down_scale)
    group_scale_bf16, group_scale_bits = _scale_bits(gate_up_scale)

    mailbox = 2 * m_tiles + 1
    ready = mailbox + 1
    init_ack = mailbox + 2
    num_semaphores = max(init_ack + 1, 16)
    if num_semaphores > 16:
        raise ValueError("full MLP schedule exceeds Blackhole semaphore IDs")

    down_replacements = {
        "@M_TILES@": str(m_tiles),
        "@K_TILES@": str(i_tiles),
        "@N_TILES@": str(h_tiles),
        "@BASE_REDUCTION_TILES@": str(plan.down_base_dx_phase),
        "@JOBS_PER_NODE@": str(down_jobs),
        "@DA_K_TILES_PER_NODE@": str(down_da_per_core),
        "@N_TILES_PER_NODE@": str(down_n_per_core),
        "@TOTAL_JOBS@": str(down_total_jobs),
        "@REDUCTION_TILES_PER_PHASE@": str(plan.down_base_dx_phase),
        "@GRID_COLS@": "10",
        "@NUM_N_BLOCKS@": str(h_tiles // 2),
        "@M_TILES_PER_NODE@": "1",
        "@TOTAL_M_TILES@": str(m_tiles),
        "@TOTAL_N_TILES@": str(h_tiles),
        "@REDUCTION_TILES@": str(m_tiles),
        "@SCALE_BITS@": str(down_scale_bits),
        "@MAILBOX_SEMAPHORE@": str(mailbox),
        "@READY_SEMAPHORE@": str(ready),
        "@INIT_ACK_SEMAPHORE@": str(init_ack),
        "@NUM_SEMAPHORES@": str(num_semaphores),
        "@TOTAL_CBS@": str(total_cbs),
        "@ACTIVATION_JOBS@": str(plan.activation_jobs_per_core),
        "@ACTIVATION_BANKS@": str(activation_banks),
        "@OUTPUT_ADDRESS_SEMAPHORE_START@": "0",
        "@OUTPUT_READY_SEMAPHORE@": "4",
        "@CONSUMER_DONE_SEMAPHORE@": "5",
        "@OUTPUT_RELEASE_SEMAPHORE@": "6",
    }
    group_replacements = {
        "@M_TILES@": str(m_tiles),
        "@K_TILES@": str(h_tiles),
        "@N_TILES@": str(i_tiles),
        "@BASE_REDUCTION_TILES@": str(plan.gate_up_base_dx_phase),
        "@JOBS_PER_NODE@": str(group_jobs_per_core),
        "@TOTAL_JOBS@": str(group_jobs),
        "@REDUCTION_TILES_PER_PHASE@": str(plan.gate_up_base_dx_phase),
        "@GRID_COLS@": "10",
        "@NUM_N_BLOCKS@": str(i_tiles // 2),
        "@M_TILES_PER_NODE@": "1",
        "@TOTAL_M_TILES@": str(m_tiles),
        "@N_TILES_PER_NODE@": str(group_n_per_core),
        "@TOTAL_N_TILES@": str(i_tiles),
        "@REDUCTION_TILES@": str(m_tiles),
        "@SCALE_BITS@": str(group_scale_bits),
        "@MEMBER_NODES@": str(_MEMBER_NODES),
        "@DX_JOBS_PER_NODE@": str(group_jobs_per_core),
        "@TOTAL_DX_JOBS@": str(group_jobs),
        "@K_TILES_PER_NODE@": "1",
        "@TOTAL_K_TILES@": str(h_tiles),
        "@RANK_TILES@": "1",
    }

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("full MLP backward requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _program(
        down_dy,
        down_weight,
        down_b,
        down_hidden,
        down_a,
        product,
        gate,
        up,
        gate_weight,
        gate_b,
        gate_hidden,
        gate_a,
        up_weight,
        up_b,
        up_hidden,
        up_a,
        inputs,
        input_gradient_output,
        gate_a_gradient_output,
        gate_b_gradient_output,
        up_a_gradient_output,
        up_b_gradient_output,
        down_a_gradient_output,
        down_b_gradient_output,
    ):
        b0 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b1 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b2 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b3 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b4 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b5 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b6 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b7 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b8 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b9 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b10 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b11 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b12 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b13 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b14 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b15 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b16 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b17 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b18 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b19 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b20 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b21 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b22 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b23 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )
        b24 = ttl.make_dataflow_buffer_like(
            input_gradient_output, shape=(1, 1), block_count=2
        )

        @ttl.datamovement()
        def read():
            with b0.reserve() as block:
                tx = ttl.copy(down_dy[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(down_weight[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(down_b[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(down_hidden[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(down_a[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(product[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(gate[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(up[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(gate_weight[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(gate_b[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(gate_hidden[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(gate_a[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(up_weight[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(up_b[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(up_hidden[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(up_a[0:1, 0:1], block)
                tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(inputs[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            with b0.wait() as value:
                with b24.reserve() as output:
                    output.store(value)
            node_col, node_row = ttl.node(dims=2)
            if node_col + node_row < 0:
                with b1.wait() as _value:
                    pass
                with b2.wait() as _value:
                    pass
                with b3.wait() as _value:
                    pass
                with b4.wait() as _value:
                    pass
                with b5.wait() as _value:
                    pass
                with b6.wait() as _value:
                    pass
                with b7.wait() as _value:
                    pass
                with b8.wait() as _value:
                    pass
                with b9.wait() as _value:
                    pass
                with b10.wait() as _value:
                    pass
                with b11.wait() as _value:
                    pass
                with b12.wait() as _value:
                    pass
                with b13.wait() as _value:
                    pass
                with b14.wait() as _value:
                    pass
                with b15.wait() as _value:
                    pass
                with b16.wait() as _value:
                    pass
                with b17.wait() as _value:
                    pass
                with b18.wait() as _value:
                    pass
                with b19.wait() as _value:
                    pass
                with b20.wait() as _value:
                    pass
                with b21.wait() as _value:
                    pass
                with b22.wait() as _value:
                    pass
                with b23.wait() as _value:
                    pass

        @ttl.datamovement()
        def write():
            with b24.wait() as value:
                tx = ttl.copy(value, input_gradient_output[0:1, 0:1])
                tx.wait()

    variant = hashlib.sha256(
        (
            f"{down_scale_bf16:.17g}:{group_scale_bf16:.17g}:"
            f"{tokens}:{hidden_features}:{intermediate_features}:"
            f"{physical_rank}:l1dy={grouped_dy_from_activation_l1}:"
            f"group-stage={grouped_execution_stage}:"
            f"shared-output={diagnostic_shared_output}:"
            f"stream-grouped-dh={stream_grouped_dh}:"
            "full-mlp-resident-backward-v31-streamed-reconfig"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.full_mlp_backward.{variant}",
        arg_roles=("in",) * 17 + ("out",) * 7,
        version_tag=f"full-mlp-backward-{variant}",
    )
    def operation(
        down_dy,
        down_weight,
        down_b,
        down_hidden,
        down_a,
        product,
        gate,
        up,
        gate_weight,
        gate_b,
        gate_hidden,
        gate_a,
        up_weight,
        up_b,
        up_hidden,
        up_a,
        inputs,
        input_gradient_output,
        gate_a_gradient_output,
        gate_b_gradient_output,
        up_a_gradient_output,
        up_b_gradient_output,
        down_a_gradient_output,
        down_b_gradient_output,
    ):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(
            down_dy,
            down_weight,
            down_b,
            down_hidden,
            down_a,
            product,
            gate,
            up,
            gate_weight,
            gate_b,
            gate_hidden,
            gate_a,
            up_weight,
            up_b,
            up_hidden,
            up_a,
            inputs,
            input_gradient_output,
            gate_a_gradient_output,
            gate_b_gradient_output,
            up_a_gradient_output,
            up_b_gradient_output,
            down_a_gradient_output,
            down_b_gradient_output,
        )
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != _DOWN_CBS:
                    raise RuntimeError(
                        f"full MLP probe expected 25 CBs, got {len(compiled.cb_configs)}"
                    )
                template = compiled.cb_configs[24]
                for activation_id in range(_ACTIVATION_CBS):
                    config = copy(template)
                    config.shape = (1, 1)
                    config.block_count = 2
                    config._cb_index = activation_id
                    compiled.cb_configs.append(config)
                first_bank = min(32, plan.activation_jobs_per_core)
                compiled.cb_configs[35].block_count = first_bank
                compiled.cb_configs[36].block_count = first_bank
                for bank in range(1, activation_banks):
                    for local_id in (13, 14):
                        config = copy(template)
                        config.shape = (1, 1)
                        config.block_count = max(
                            2, plan.activation_jobs_per_core - 32 * bank
                        )
                        config._cb_index = local_id
                        compiled.cb_configs.append(config)
                grouped_transpose = copy(template)
                grouped_transpose.shape = (1, m_tiles)
                grouped_transpose.block_count = 1
                grouped_transpose._cb_index = 25
                compiled.cb_configs.append(grouped_transpose)
                grouped_peer = copy(template)
                grouped_peer.shape = (1, 1)
                grouped_peer.block_count = max(2, group_jobs_per_core)
                grouped_peer._cb_index = 26
                compiled.cb_configs.append(grouped_peer)
                grouped_dh_configs = (
                    ((1, i_tiles), 1, 27, False),
                    ((i_tiles, 1), 1, 28, False),
                    ((1, i_tiles), 1, 29, False),
                    ((1, 1), 2, 30, True),
                    ((1, 1), max(2, group_jobs_per_core), 31, False),
                )
                for shape, block_count, local_id, fp32 in grouped_dh_configs:
                    config = copy(template)
                    config.shape = shape
                    config.block_count = block_count
                    config._cb_index = local_id
                    if fp32:
                        config.tensor = _FP32_DTYPE_ONLY_TENSOR
                    compiled.cb_configs.append(config)
                for local_id in range(15, 21):
                    config = copy(template)
                    config.shape = (1, 1)
                    config.block_count = 2
                    config._cb_index = local_id
                    compiled.cb_configs.append(config)
                if stream_grouped_dh:
                    bank_jobs = (
                        min(32, plan.activation_jobs_per_core),
                        max(0, plan.activation_jobs_per_core - 32),
                    )
                    for bank in range(activation_banks):
                        for output in range(2):
                            config = copy(template)
                            config.shape = (1, 1)
                            config.block_count = max(2, bank_jobs[bank])
                            config._cb_index = 21 + 2 * bank + output
                            compiled.cb_configs.append(config)
                # Reused CBs must cover the larger of the down and grouped
                # phases. Shapes encode capacity without violating the
                # hardware's 32-block-count field.
                shapes = {
                    0: (
                        (1, max(plan.down_base_dx_phase, plan.gate_up_base_dx_phase)),
                        2,
                    ),
                    1: (
                        (max(plan.down_base_dx_phase, plan.gate_up_base_dx_phase), 1),
                        2,
                    ),
                    3: ((1, 1), max(2, down_jobs, group_jobs_per_core)),
                    # Grouped dH owns CB27..31, so these remain down-only.
                    4: ((1, h_tiles), 1),
                    5: ((h_tiles, 1), 1),
                    6: ((1, h_tiles), 1),
                    8: ((1, 1), max(2, down_jobs, group_jobs_per_core)),
                    9: ((m_tiles, 1), 1),
                    10: ((m_tiles, 1), 1),
                    # Grouped phase-one uses these as M-tile dB staging
                    # windows before the tail reuses them for dX sums.
                    11: ((1, max(m_tiles, group_jobs_per_core)), 1),
                    12: ((1, max(m_tiles, group_jobs_per_core)), 1),
                    15: ((1, 1), max(2, down_n_per_core, group_n_per_core)),
                    16: ((1, m_tiles), 1),
                    17: ((1, m_tiles), max(1, down_jobs)),
                    18: ((m_tiles, 1), max(1, down_da_per_core)),
                    19: ((1, m_tiles), max(1, down_da_per_core)),
                    21: ((1, 1), 1),
                    22: ((1, 1), max(2, down_da_per_core)),
                    23: ((1, 1), max(2, down_jobs)),
                    24: ((1, 1), max(2, down_jobs)),
                }
                for index, (shape, block_count) in shapes.items():
                    compiled.cb_configs[index].shape = shape
                    compiled.cb_configs[index].block_count = block_count
                for index in (2, 7, 13, 20):
                    compiled.cb_configs[index].tensor = _FP32_DTYPE_ONLY_TENSOR
                if (
                    stream_grouped_dh
                    and grouped_execution_stage == 1
                    and grouped_dy_from_activation_l1
                ):
                    # CB27 remains the complete down-product row used by the
                    # preceding phase. Grouped B/scaled-dY stream through
                    # CB28/29 at the exact two-tile reduction granularity.
                    #
                    # The later ``tail_a`` stream has a dedicated CB.  Do not
                    # make CB28 depth three: its repeated reserve(2) geometry
                    # would cross the physical ring boundary on every second
                    # block and corrupt the adjacent L1 allocation.
                    grouped_b_depth, scaled_dy_depth = (
                        _streamed_grouped_dh_cb_depths(
                            group_jobs_per_core=group_jobs_per_core
                        )
                    )
                    compiled.cb_configs[43].shape = (1, 1)
                    compiled.cb_configs[43].block_count = grouped_b_depth
                    compiled.cb_configs[44].shape = (1, 1)
                    compiled.cb_configs[44].block_count = scaled_dy_depth

                down_reader = _down_reader_source(
                    down_replacements,
                    num_semaphores=num_semaphores,
                    init_ack=init_ack,
                    total_cbs=total_cbs,
                )
                down_compute = _down_compute_source(down_replacements)
                down_reader, down_compute = _inject_down_h_recompute(
                    down_reader,
                    down_compute,
                    product_cb=recompute_product_arg,
                    a_cb=recompute_a_arg,
                    partial_cb=recompute_partial_arg,
                    output_cb=recompute_h_arg,
                    gate_cb=recompute_gate_arg,
                    up_cb=recompute_up_arg,
                    sigmoid_cb=recompute_sigmoid_arg,
                    silu_cb=recompute_silu_arg,
                    gate_saved_cb=recompute_gate_saved_arg,
                    m_tiles=m_tiles,
                    k_tiles=i_tiles,
                    h_tiles=h_tiles,
                    da_k_tiles_per_node=down_da_per_core,
                )
                down_reader, down_compute = _matmul_product_from_activation_l1(
                    down_reader,
                    down_compute,
                    m_tiles=m_tiles,
                    k_tiles=i_tiles,
                )
                down_compute = down_compute.replace(
                    '#include "api/compute/eltwise_binary.h"',
                    '#include "api/compute/compute_kernel_api.h"\n'
                    '#include "api/compute/eltwise_binary.h"\n'
                    '#include "api/compute/eltwise_binary_sfpu.h"',
                    1,
                )
                down_writer = _replace_all(
                    _cpp("resident_down_activation_matmul_writer.cpp"),
                    down_replacements,
                )
                group_compute, group_reader, group_writer = _grouped_sources(
                    replacements=group_replacements, total_cbs=total_cbs
                )
                # CB17 and CB18 change producer RISC between the down and
                # grouped schedules. Give their grouped incarnations unique
                # local IDs so producer ownership remains stable.
                for name, source in (
                    ("compute", group_compute),
                    ("reader", group_reader),
                    ("writer", group_writer),
                ):
                    source = source.replace(
                        "get_compile_time_arg_val(17)",
                        f"get_compile_time_arg_val({grouped_transpose_arg})",
                    ).replace(
                        "get_compile_time_arg_val(18)",
                        f"get_compile_time_arg_val({grouped_peer_arg})",
                    )
                    for old_id in range(4, 9):
                        source = source.replace(
                            f"get_compile_time_arg_val({old_id})",
                            "get_compile_time_arg_val("
                            f"{grouped_dh_arg_start + old_id - 4})",
                        )
                    if name == "compute":
                        group_compute = source
                    elif name == "reader":
                        group_reader = source
                    else:
                        group_writer = source

                if stream_grouped_dh:
                    # Logical grouped arg 5 has now been shifted to the
                    # depth-two grouped-B FIFO.  Tail-A must instead alias
                    # logical arg 17 of the complete MLP program: the empty,
                    # reader-produced down tail-A FIFO.  Remap both
                    # producer and consumer after the generic grouped shift.
                    shifted_tail_arg = grouped_dh_arg_start + 1
                    tail_a_declaration = (
                        "constexpr uint32_t cb_tail_a = "
                        f"get_compile_time_arg_val({shifted_tail_arg});"
                    )
                    tail_a_compute_declaration = (
                        "constexpr uint32_t cb_a = "
                        f"get_compile_time_arg_val({shifted_tail_arg});"
                    )
                    if group_reader.count(tail_a_declaration) != 1:
                        raise RuntimeError("grouped tail-A declaration changed")
                    if group_compute.count(tail_a_compute_declaration) != 1:
                        raise RuntimeError(
                            "grouped tail-A compute declaration changed"
                        )
                    group_reader = group_reader.replace(
                        tail_a_declaration,
                        "constexpr uint32_t cb_tail_a = "
                        f"get_compile_time_arg_val({grouped_tail_a_arg});",
                        1,
                    )
                    group_compute = group_compute.replace(
                        tail_a_compute_declaration,
                        "constexpr uint32_t cb_a = "
                        f"get_compile_time_arg_val({grouped_tail_a_arg});",
                        1,
                    )

                # Map the grouped operation's logical tensor slots onto the
                # complete MLP operation. All inputs are reader common args,
                # so common positions equal operation tensor indices.
                tensor_map = {
                    0: 6,
                    1: 8,
                    2: 9,
                    3: 10,
                    4: 11,
                    5: 7,
                    6: 12,
                    7: 13,
                    8: 14,
                    9: 15,
                    10: 16,
                }
                group_reader = re.sub(
                    r"get_common_arg_val<uint32_t>\((\d+)\)",
                    lambda match: (
                        "get_common_arg_val<uint32_t>("
                        f"{tensor_map[int(match.group(1))]})"
                    ),
                    group_reader,
                )
                # Tensor slot 10 is now the complete grouped-forward arena,
                # not a PJRT slice. Resolve the saved-X tile offset in the
                # native reader so no activation view crosses the ABI.
                old_x_page = "{.page_id = m * k_tiles + node}"
                grouped_arena_tiles = 2 * i_tiles + 2 + h_tiles
                checkpoint_tile_offset = 2 * i_tiles + 2
                if group_reader.count(old_x_page) != 1:
                    raise RuntimeError("grouped X page-id marker changed")
                group_reader = group_reader.replace(
                    old_x_page,
                    "{.page_id = m * "
                    f"{grouped_arena_tiles} + "
                    f"{checkpoint_tile_offset} + node}}",
                    1,
                )
                old_hidden_page = "{.page_id = m}"
                if group_reader.count(old_hidden_page) != 1:
                    raise RuntimeError("grouped saved-H page-id marker changed")
                group_reader = group_reader.replace(
                    old_hidden_page,
                    "{.page_id = m * "
                    f"{grouped_arena_tiles} + "
                    f"{2 * i_tiles} + member}}",
                    1,
                )
                group_reader = re.sub(
                    r"get_tensor_accessor_args_cta_offset<(\d+),\s*\d+>\(\),\s*(\d+)",
                    lambda match: (
                        "get_tensor_accessor_args_cta_offset<"
                        f"{tensor_map[int(match.group(1))]}, {total_cbs}>(), "
                        f"{tensor_map[int(match.group(2))]}"
                    ),
                    group_reader,
                )
                if grouped_execution_stage in (0, 2):
                    group_reader = _handoff_only_grouped_reader(
                        snapshot_barrier=grouped_execution_stage == 2
                    )
                    group_compute = "void kernel_main() {}\n"
                    group_writer = "void kernel_main() {}\n"
                elif grouped_dy_from_activation_l1:
                    group_reader = _grouped_reader_from_activation_l1(
                        group_reader,
                        down_jobs_per_node=down_jobs,
                        down_total_jobs=down_total_jobs,
                        activation_banks=activation_banks,
                        scaled_gradient_banks=stream_grouped_dh,
                    )
                    if stream_grouped_dh:
                        group_reader, group_compute = _stream_grouped_dh_reduction(
                            group_reader,
                            group_compute,
                            grouped_dh_arg_start=grouped_dh_arg_start,
                        )
                else:
                    group_reader = _grouped_reader_dram_diagnostic(group_reader)

                matmul_reader = _compose_reader(down_reader, group_reader)
                matmul_compute = _compose_compute(
                    down_compute, group_compute, m_tiles=m_tiles
                )
                matmul_writer = _compose_writer(
                    down_writer,
                    group_writer,
                    m_tiles=m_tiles,
                    intermediate_tiles=i_tiles,
                    hidden_tiles=h_tiles,
                    total_cbs=total_cbs,
                    diagnostic_shared_output=diagnostic_shared_output,
                )
                activation_compute = _banked_activation_compute_source(
                    _activation_compute_source(
                        down_replacements,
                        producer_jobs=down_jobs,
                        total_jobs=down_total_jobs,
                        diagnostic_stage=6,
                    ),
                    activation_banks=activation_banks,
                )
                activation_writer = _replace_all(
                    _cpp("resident_down_activation_l1_writer.cpp"),
                    down_replacements,
                )
                if stream_grouped_dh:
                    activation_compute = _activation_scaled_gradient_compute_source(
                        activation_compute,
                        activation_banks=activation_banks,
                        scaled_arg_start=activation_scaled_arg_start,
                        scale_bits=group_scale_bits,
                    )
                    activation_writer = _activation_scaled_gradient_writer_source(
                        activation_writer,
                        activation_banks=activation_banks,
                        scaled_arg_start=activation_scaled_arg_start,
                    )
                activation_reader = _replace_all(
                    _cpp("resident_down_activation_reader.cpp"),
                    down_replacements,
                )
                # Gate and up are two regions in the same grouped-forward
                # arena. Keep the complete arena as the TT-Metal argument and
                # resolve both row strides/column offsets here.
                activation_page = "{.page_id = job}"
                if activation_reader.count(activation_page) != 2:
                    raise RuntimeError("activation gate/up page-id markers changed")
                gate_page = (
                    "{.page_id = (job / "
                    f"{i_tiles}) * {grouped_arena_tiles} + "
                    f"(job % {i_tiles})}}"
                )
                up_page = (
                    "{.page_id = (job / "
                    f"{i_tiles}) * {grouped_arena_tiles} + "
                    f"{i_tiles} + (job % {i_tiles})}}"
                )
                activation_reader = activation_reader.replace(
                    activation_page, gate_page, 1
                ).replace(activation_page, up_page, 1)
                activation_reader, activation_compute, activation_writer = (
                    _activation_product_sources(
                        activation_reader,
                        activation_compute,
                        activation_writer,
                        total_cbs=total_cbs,
                        m_tiles=m_tiles,
                        i_tiles=i_tiles,
                        da_k_tiles_per_node=down_da_per_core,
                        gate_arg=activation_product_gate_arg,
                        gate_saved_arg=activation_product_gate_saved_arg,
                        up_arg=activation_product_up_arg,
                        sigmoid_arg=activation_product_sigmoid_arg,
                        silu_arg=activation_product_silu_arg,
                        output_arg=activation_product_output_arg,
                    )
                )
                activation_reader = activation_reader.replace(
                    "@PRODUCT_ROW_TILES@",
                    str(2 * i_tiles + 2 + h_tiles),
                )

                generated = Path(compiled.kernel_paths[0][0]).parent
                source_specs = (
                    ("full_mlp_matmul_reader.cpp", "noc", matmul_reader),
                    ("full_mlp_matmul_compute.cpp", "compute", matmul_compute),
                    ("full_mlp_matmul_writer.cpp", "noc", matmul_writer),
                    ("full_mlp_activation_reader.cpp", "noc", activation_reader),
                    ("full_mlp_activation_compute.cpp", "compute", activation_compute),
                    ("full_mlp_activation_writer.cpp", "noc", activation_writer),
                )
                paths = []
                for filename, kind, source in source_specs:
                    path = generated / filename
                    path.write_text(source, encoding="utf-8")
                    dump_root = os.environ.get("TT_FINETUNE_SOURCE_DUMP_DIR")
                    if dump_root:
                        dump = Path(dump_root) / (
                            f"full_mlp_{tokens}x{hidden_features}x"
                            f"{intermediate_features}"
                        )
                        dump.mkdir(parents=True, exist_ok=True)
                        (dump / filename).write_text(source, encoding="utf-8")
                    paths.append((str(path), kind))

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
                    ttnn.ReaderConfigDescriptor(),
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,
                        fp32_dest_acc_en=False,
                    ),
                    ttnn.WriterConfigDescriptor(),
                ]
                compiled.kernel_tensor_indices = [
                    list(range(17)),
                    [],
                    list(range(17, 24)),
                    [5, 6, 7],
                    [],
                    [],
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
                compiled.cb_core_ranges = (
                    [_MATMUL_RANGE] * _DOWN_CBS
                    + [_ACTIVATION_RANGE] * (_ACTIVATION_CBS + activation_extra_cbs)
                    + [_MATMUL_RANGE] * 2
                    + [_MATMUL_RANGE] * 5
                    + [_ACTIVATION_RANGE] * 6
                    + [_ACTIVATION_RANGE] * activation_scaled_count
                )
                dump_root = os.environ.get("TT_FINETUNE_SOURCE_DUMP_DIR")
                if dump_root:
                    dump = Path(dump_root) / (
                        f"full_mlp_{tokens}x{hidden_features}x"
                        f"{intermediate_features}"
                    )
                    metadata = {
                        "cb_configs": [
                            {key: str(value) for key, value in vars(config).items()}
                            for config in compiled.cb_configs
                        ],
                        "cb_core_ranges": compiled.cb_core_ranges,
                        "kernel_tensor_indices": (compiled.kernel_tensor_indices),
                        "kernel_core_ranges": compiled.kernel_core_ranges,
                    }
                    (dump / "metadata.json").write_text(
                        json.dumps(metadata, indent=2) + "\n",
                        encoding="utf-8",
                    )
                compiled.semaphore_configs = [
                    {
                        "id": semaphore,
                        "core_type": "worker",
                        "core_range": _ALL_WORKERS_RANGE,
                        "initial_value": 0,
                    }
                    for semaphore in range(num_semaphores)
                ]
                compiled.num_pipe_sync_semaphores = num_semaphores
                compiled.kernel_arg_specs = [[] for _ in range(6)]
                if artifact_program_sequence:
                    compiled.wrap_as_program_sequence = True
            break
        return result

    operation.output_shape = full_mlp_backward_output_shape(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
    )
    operation.experimental_backend = "full_mlp_resident_v1"
    operation.intermediate_pjrt_outputs = 0
    operation.grouped_dy_from_activation_l1 = bool(grouped_dy_from_activation_l1)
    operation.grouped_execution_stage = grouped_execution_stage
    operation.stream_grouped_dh = bool(stream_grouped_dh)
    operation.artifact_program_sequence = bool(artifact_program_sequence)
    return operation
