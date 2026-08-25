"""One-call resident output-projection plus attention-core backward.

This is the first true full-attention fusion boundary. A 50-core projection
island computes exact frozen+LoRA output-projection backward and sends dO
directly over NoC into a disjoint 60-core attention island. The accepted
resident attention core then emits dQ/dK/dV. No dO tensor crosses PJRT/DRAM.
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

from .attention_block_planner import plan_attention_block
from .full_attention_handoff import plan_output_attention_l1_handoff
from .full_attention_projection_handoff import (
    plan_attention_projection_l1_handoff,
)
from .native_frozen_lora_autograd import exact_base_dx_k_phase
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID
from .tt_lang_frozen_linear_lora_backward import (
    _shift_compile_time_cb_indices,
)
from .tt_lang_lora import TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR


ttl = None
tt_lang_operation = None

_PROJECTION_RANGE = {"start": [0, 0], "end": [4, 9]}
_ATTENTION_RANGE = {"start": [5, 0], "end": [10, 9]}
_ALL_WORKERS_RANGE = {"start": [0, 0], "end": [10, 9]}
_PROJECTION_CBS = 25
_ATTENTION_CBS = 38
_TOTAL_CBS = _PROJECTION_CBS + _ATTENTION_CBS
_FULL_QKV_TOTAL_CBS = _TOTAL_CBS + 32


def _attention_cb_blocks(
    *, seq_tiles: int, dim_tiles: int, vector_tiles: int, group: int,
) -> tuple[int, ...]:
    """Return resident attention CB capacities in local-ID order.

    ``cb_sum_dk_b`` (local ID 26) is deliberately a complete dK/dV window.
    The original writer consumed each reduced tile immediately and therefore
    needed only one slot.  Inverse RoPE consumes dK only after the complete
    GQA reduction, so a one-tile CB blocks its producer before RoPE can start
    and deadlocks the program.  Keep this policy named and unit-tested.
    """

    blocks = (
        seq_tiles, seq_tiles, dim_tiles, vector_tiles,
        vector_tiles, vector_tiles, vector_tiles, 1,
        seq_tiles, dim_tiles, vector_tiles, seq_tiles,
        seq_tiles, 1, seq_tiles, seq_tiles, seq_tiles,
        dim_tiles, seq_tiles, seq_tiles, seq_tiles,
        vector_tiles, dim_tiles, (group - 1) * dim_tiles,
        (group - 1) * dim_tiles, 2 * dim_tiles * group,
        2 * dim_tiles, 1, dim_tiles, dim_tiles, dim_tiles, seq_tiles,
        dim_tiles, dim_tiles, 2, 2, dim_tiles, dim_tiles,
    )
    if len(blocks) != _ATTENTION_CBS:
        raise RuntimeError(
            f"attention CB contract has {len(blocks)} entries, "
            f"expected {_ATTENTION_CBS}"
        )
    return blocks


def _projection_cb_shapes(
    *, m_tiles: int, n_tiles: int, base_phase: int,
    jobs_per_node: int, da_k_tiles_per_node: int,
    n_tiles_per_node: int, rank_tiles: int = 1,
) -> dict[int, tuple[tuple[int, int], int]]:
    """Exact CB windows required by the compact projection program."""

    return {
        0: ((1, base_phase), 2),
        1: ((base_phase, 1), 2),
        2: ((1, 1), 2),
        3: ((1, 1), max(2, jobs_per_node)),
        4: ((1, n_tiles), 1),
        5: ((n_tiles, rank_tiles), 1),
        6: ((1, n_tiles), 1),
        7: ((1, rank_tiles), 1),
        8: ((1, rank_tiles), max(2, jobs_per_node)),
        9: ((m_tiles, 1), 1),
        10: ((m_tiles, rank_tiles), 1),
        # dB scale/transpose operate on the complete token reduction
        # window. Describing either CB as a scalar deadlocks at
        # reserve_back(m_tiles), before dB consumes its first input tile.
        11: ((m_tiles, rank_tiles), 1),
        12: ((rank_tiles, m_tiles), 1),
        13: ((1, rank_tiles), 1),
        14: ((1, rank_tiles), 1),
        15: ((1, rank_tiles), max(2, n_tiles_per_node)),
        16: ((rank_tiles, m_tiles), 1),
        17: ((1, 1), max(2, jobs_per_node)),
        18: ((m_tiles, 1), max(2, da_k_tiles_per_node)),
        19: ((1, m_tiles), max(2, da_k_tiles_per_node)),
        20: ((1, 1), 1),
        21: ((1, 1), 1),
        22: ((1, 1), max(2, da_k_tiles_per_node)),
        23: ((1, 1), max(2, jobs_per_node)),
        24: ((1, 1), max(2, jobs_per_node)),
    }


def _full_serial_qkv_cb_capacities(
    *, m_tiles: int, hidden_tiles: int, kv_tiles: int,
    jobs_per_node: int,
) -> dict[int, int]:
    """Physical FIFO history for V -> K -> Q on one projection island.

    The isolated QKV owner runs K/V on separate core groups, so its shared-CB
    policy accounts for one KV publication followed by Q.  The whole
    attention owner intentionally serializes V, K and Q on the same 50 cores;
    multi-tile reserves cannot straddle the end of a CB allocation, therefore
    the additional V history must be represented explicitly.
    """

    return {
        4: 2 * kv_tiles + hidden_tiles,
        5: 2 * (kv_tiles + jobs_per_node) + hidden_tiles + jobs_per_node,
        6: 2 * (kv_tiles + jobs_per_node) + hidden_tiles + jobs_per_node,
        8: 3 + jobs_per_node,
        9: 4 * m_tiles,
        10: 3 * m_tiles,
        11: 3 * m_tiles + jobs_per_node,
    }


def _complete_projection_jobs_per_node(
    *, total_jobs: int, projection_workers: int,
) -> int:
    """Choose a full-job worker quotient for the resident serial owner.

    A CB is local to a Tensix core.  Letting the last active core publish a
    partial FIFO epoch (one tile in a three-tile window for Qwen) leaves its
    terminal pack/write pointer on a different epoch from every peer.  The
    full owner has no reason to create that exceptional lifecycle: choose the
    first jobs-per-core value whose active workers divide the work exactly.
    Idle projection cores still participate in the device barriers.
    """

    jobs_per_node = -(-total_jobs // projection_workers)
    while total_jobs % jobs_per_node:
        jobs_per_node += 1
    return jobs_per_node


def output_attention_projection_arena_shape(
    *, tokens: int, hidden_features: int
) -> tuple[int, int]:
    # Compatibility layout of resident_down_backward. dO occupies the first
    # rows logically but is deliberately never written by the fused writer.
    return tokens + TILE_SIZE + hidden_features, hidden_features


def output_attention_arena_shape(
    *, tokens: int, hidden_features: int, query_heads: int,
    kv_heads: int, head_dim: int,
) -> tuple[int, int]:
    """Single-output arena required by the current TT PJRT custom-call ABI."""

    projection_rows, _ = output_attention_projection_arena_shape(
        tokens=tokens, hidden_features=hidden_features
    )
    core_rows = (query_heads + 2 * kv_heads) * tokens
    return projection_rows + core_rows, hidden_features


def full_attention_backward_arena_shape(
    *, tokens: int, hidden_features: int, kv_features: int,
) -> tuple[int, int]:
    """Final-only arena for the complete resident attention backward owner.

    The layout deliberately contains no dQ/dK/dV regions.  Those tensors are
    an internal L1/NoC edge between attention and the Q/K/V projection
    phases.  Rows are, in order::

        dHidden, o_dA, o_dB, q_dA, k_dA, v_dA, q_dB, k_dB, v_dB

    Every logical-rank adapter gradient uses a physical rank-32 tile row.
    """

    return (
        tokens
        + 4 * TILE_SIZE
        + 2 * hidden_features
        + 2 * kv_features,
        hidden_features,
    )


def output_attention_support_error(
    *, sequence_length: int, hidden_features: int, query_heads: int,
    kv_heads: int, head_dim: int, physical_rank: int,
) -> str | None:
    try:
        plan_attention_block(
            batch_size=1,
            sequence_length=sequence_length,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        plan_output_attention_l1_handoff(
            sequence_length=sequence_length,
            hidden_features=hidden_features,
            query_heads=query_heads,
            head_dim=head_dim,
        )
    except ValueError as error:
        return str(error)
    if sequence_length != 128:
        return "first output+attention owner requires sequence length 128"
    if physical_rank != TILE_SIZE:
        return "output+attention owner requires physical rank 32"
    if query_heads // kv_heads % 2 != 1:
        return "output+attention owner requires an odd GQA group"
    phase = exact_base_dx_k_phase(hidden_features, hidden_features)
    if phase is None:
        return (
            "no exact output-projection dX schedule for "
            f"{hidden_features}->{hidden_features}"
        )
    return None


def full_attention_projection_support_error(
    *, sequence_length: int, hidden_features: int, query_heads: int,
    kv_heads: int, head_dim: int, physical_rank: int,
) -> str | None:
    """Return why the resident attention->QKV projection owner cannot run."""

    error = output_attention_support_error(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
    )
    if error is not None:
        return error
    try:
        plan_attention_projection_l1_handoff(
            sequence_length=sequence_length,
            hidden_features=hidden_features,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
    except ValueError as exc:
        return str(exc)
    kv_features = kv_heads * head_dim
    if exact_base_dx_k_phase(hidden_features, kv_features) is None:
        return (
            "no exact K/V projection dX schedule for "
            f"{hidden_features}->{kv_features}"
        )
    return None


def _replace_all(source: str, replacements: dict[str, str]) -> str:
    for marker, value in replacements.items():
        source = source.replace(marker, value)
    unresolved = re.findall(r"@[A-Z0-9_]+@", source)
    if unresolved:
        raise RuntimeError(f"unexpanded fused-attention markers: {unresolved}")
    return source


def _append_before_kernel_end(source: str, body: str) -> str:
    """Append a phased kernel body before the outer ``kernel_main`` brace."""

    source, separator, tail = source.rpartition("\n}")
    if not separator or tail.strip():
        raise RuntimeError("TT-Metal kernel outer brace changed")
    return source + "\n" + body.rstrip() + "\n}"


def _projection_handoff_reader_tail() -> str:
    return r"""
    // Phase 3 diagnostic: pull the post-RoPE dQ/dK and reduced dV tiles
    // directly from their physical attention-core owners.  CB0 is empty
    // after the output-projection compute phase and is reused as a one-block
    // stream; no tensor or PJRT result exists at this boundary.
    constexpr uint32_t qkv_cb = get_compile_time_arg_val(0);
    constexpr uint32_t qkv_seq_tiles = @SEQ_TILES@;
    constexpr uint32_t qkv_hidden_tiles = @K_TILES@;
    constexpr uint32_t qkv_dim_tiles = @DIM_TILES@;
    constexpr uint32_t qkv_query_heads = @QUERY_HEADS@;
    constexpr uint32_t qkv_kv_heads = @KV_HEADS@;
    constexpr uint32_t qkv_group = qkv_query_heads / qkv_kv_heads;
    constexpr uint32_t qkv_tiles = qkv_seq_tiles * qkv_hidden_tiles;
    constexpr uint32_t one_kv_tiles =
        qkv_seq_tiles * qkv_kv_heads * qkv_dim_tiles;
    constexpr uint32_t total_handoff_tiles = qkv_tiles + 2 * one_kv_tiles;
    constexpr uint32_t active_attention_nodes =
        qkv_query_heads * qkv_seq_tiles;
    constexpr uint8_t all_physical_x[11] = {
        1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
    volatile tt_l1_ptr uint32_t* attention_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* handoff_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    if (node == 0) {
        noc_semaphore_wait_min(attention_ready, active_attention_nodes);
        noc_semaphore_set(attention_ready, 0);
        for (uint32_t target = 1; target < kWorkerNodes; ++target) {
            for (uint32_t word = 11; word <= 13; ++word) {
                noc_async_write(
                    get_semaphore(word),
                    get_noc_addr(
                        kPhysicalX[target % kGridCols],
                        physical_y(target / kGridCols),
                        get_semaphore(word)),
                    sizeof(uint32_t));
            }
        }
        noc_async_write_barrier();
        noc_semaphore_set(handoff_go, 1);
        for (uint32_t target = 1; target < kWorkerNodes; ++target) {
            noc_async_write(
                get_semaphore(4),
                get_noc_addr(
                    kPhysicalX[target % kGridCols],
                    physical_y(target / kGridCols), get_semaphore(4)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(handoff_go, 1);
    noc_semaphore_set(handoff_go, 0);
    const uint32_t q_address =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint32_t k_address =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
    const uint32_t v_address =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
    CircularBuffer qkv_stream(qkv_cb);
    for (uint32_t linear = node; linear < total_handoff_tiles;
         linear += kWorkerNodes) {
        uint32_t local = linear;
        uint32_t source_address = q_address;
        uint32_t source_node = 0;
        uint32_t source_tile = 0;
        if (local < qkv_tiles) {
            const uint32_t token = local / qkv_hidden_tiles;
            const uint32_t output = local - token * qkv_hidden_tiles;
            const uint32_t head = output / qkv_dim_tiles;
            source_tile = output - head * qkv_dim_tiles;
            source_node = head * qkv_seq_tiles + token;
        } else {
            local -= qkv_tiles;
            const bool is_v = local >= one_kv_tiles;
            if (is_v) local -= one_kv_tiles;
            const uint32_t kv_width = qkv_kv_heads * qkv_dim_tiles;
            const uint32_t token = local / kv_width;
            const uint32_t output = local - token * kv_width;
            const uint32_t kv_head = output / qkv_dim_tiles;
            source_tile = output - kv_head * qkv_dim_tiles;
            source_node = kv_head * qkv_group * qkv_seq_tiles + token;
            source_address = is_v ? v_address : k_address;
        }
        const uint32_t sx = 5 + source_node % 6;
        const uint32_t sy = source_node / 6;
        qkv_stream.reserve_back(1);
        noc_async_read(
            get_noc_addr(
                all_physical_x[sx], physical_y(sy),
                source_address + source_tile * kTileBytes),
            qkv_stream.get_write_ptr(), kTileBytes);
        noc_async_read_barrier();
        qkv_stream.push_back(1);
    }
"""


def _projection_handoff_writer_tail() -> str:
    return r"""
    constexpr uint32_t qkv_cb = get_compile_time_arg_val(0);
    constexpr uint32_t qkv_seq_tiles = @SEQ_TILES@;
    constexpr uint32_t qkv_hidden_tiles = @K_TILES@;
    constexpr uint32_t qkv_dim_tiles = @DIM_TILES@;
    constexpr uint32_t qkv_query_heads = @QUERY_HEADS@;
    constexpr uint32_t qkv_kv_heads = @KV_HEADS@;
    constexpr uint32_t qkv_width = qkv_kv_heads * qkv_dim_tiles;
    constexpr uint32_t qkv_tiles = qkv_seq_tiles * qkv_hidden_tiles;
    constexpr uint32_t one_kv_tiles = qkv_seq_tiles * qkv_width;
    constexpr uint32_t total_handoff_tiles = qkv_tiles + 2 * one_kv_tiles;
    constexpr uint32_t core_output_row = @CORE_OUTPUT_ROW_OFFSET@;
    CircularBuffer qkv_stream(qkv_cb);
    for (uint32_t linear = node; linear < total_handoff_tiles;
         linear += kProjectionWorkers) {
        uint32_t local = linear;
        uint32_t row_offset = 0;
        uint32_t width = qkv_hidden_tiles;
        if (local >= qkv_tiles) {
            local -= qkv_tiles;
            row_offset = qkv_seq_tiles;
            width = qkv_width;
            if (local >= one_kv_tiles) {
                local -= one_kv_tiles;
                row_offset += qkv_seq_tiles;
            }
        }
        const uint32_t token = local / width;
        const uint32_t column = local - token * width;
        qkv_stream.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(qkv_stream.get_read_ptr()),
            arena, kTileBytes, {},
            {.page_id = (core_output_row + row_offset + token) *
                        arena_cols + column});
        noc1.async_write_barrier();
        qkv_stream.pop_front(1);
    }
    // Every projection writer participates, including workers with no tile,
    // so attention node zero has an exact phase-completion count.
    noc_semaphore_inc(
        attention_core_l1(0, get_semaphore(10)), 1);
    noc_async_atomic_barrier();
"""


def _attention_writer_to_projection_handoff(source: str) -> str:
    """Keep final attention gradients resident for projection-side pulls."""

    q_start_marker = '''    {
        DeviceZoneScopedN("attention_final_dq_write");'''
    q_end_marker = "    if constexpr (diagnostic_prefix)"
    if source.count(q_start_marker) != 1 or source.count(q_end_marker) != 1:
        raise RuntimeError("resident q handoff writer marker changed")
    q_start = source.index(q_start_marker)
    q_end = source.index(q_end_marker, q_start)
    source = (
        source[:q_start]
        + "    // rope_q remains at the CB front for the projection island.\n"
        + source[q_end:]
    )

    peer_return = '''        dk.pop_front(dim_tiles);
        dv.pop_front(dim_tiles);
        return;
'''
    peer_wait = '''        dk.pop_front(dim_tiles);
        dv.pop_front(dim_tiles);
        // Projection readers publish this only after every output-projection
        // writer has stopped touching the shared semaphore map.
        volatile tt_l1_ptr uint32_t* qkv_publish_go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(14));
        noc_semaphore_wait(qkv_publish_go, 1);
        noc_semaphore_set(qkv_publish_go, 0);
        // Publish only after this worker has finished all GQA peer writes.
        noc_semaphore_inc(
            get_noc_addr(1, physical_y(0), get_semaphore(0)), 1);
        noc_async_atomic_barrier();
        volatile tt_l1_ptr uint32_t* release =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(11));
        noc_semaphore_wait(release, 1);
        noc_semaphore_set(release, 0);
        rope_q.pop_front(dim_tiles);
        return;
'''
    if source.count(peer_return) != 1:
        raise RuntimeError("resident peer handoff marker changed")
    source = source.replace(peer_return, peer_wait, 1)

    resident_body = r'''        // Do not publish Q/K/V addresses into the
        // projection-local 0..4 map until resident-down has retired there.
        volatile tt_l1_ptr uint32_t* qkv_publish_go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(14));
        noc_semaphore_wait(qkv_publish_go, 1);
        noc_semaphore_set(qkv_publish_go, 0);
        // Semaphore 10 is the projection-writer completion counter on
        // attention node zero.  It may contain state from an unrelated
        // program (or an earlier invocation), so establish a clean epoch
        // before this node publishes the last attention-ready credit.  The
        // projection readers cannot start until every attention worker,
        // including node zero, has published that credit; consequently no
        // projection writer can race this reset.
        if (node == 0) {
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(10)),
                0);
        }
        WAYPOINT("AH0");
        rope_k.wait_front(dim_tiles);
        // Skip the unrotated dK window.  The reduced dV window remains at
        // the front beside post-RoPE K until the projection readers finish.
        gqa_reduced.pop_front(dim_tiles);
        gqa_reduced.wait_front(dim_tiles);
        WAYPOINT("AH1");
        if (node == 0) {
            volatile tt_l1_ptr uint32_t* address_word =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(15));
            const uint32_t addresses[3] = {
                rope_q.get_read_ptr(), rope_k.get_read_ptr(),
                gqa_reduced.get_read_ptr()};
            DPRINT_DATA1(
                "attention resident addresses q={} k={} v={}\n",
                addresses[0], addresses[1], addresses[2]);
            for (uint32_t index = 0; index < 3; ++index) {
                noc_semaphore_set(address_word, addresses[index]);
                noc_async_write(
                    get_semaphore(15),
                    get_noc_addr(
                        1, physical_y(0), get_semaphore(11 + index)),
                    sizeof(uint32_t));
                noc_async_write_barrier();
            }
        }
        noc_semaphore_inc(
            get_noc_addr(1, physical_y(0), get_semaphore(0)), 1);
        noc_async_atomic_barrier();
        WAYPOINT("AH2");
        volatile tt_l1_ptr uint32_t* release =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(11));
        if (node == 0) {
            WAYPOINT("AH3");
            volatile tt_l1_ptr uint32_t* projection_done =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(10));
            noc_semaphore_wait_min(projection_done, 50);
            noc_semaphore_set(projection_done, 0);
            noc_semaphore_set(release, 1);
            for (uint32_t target = 1; target < jobs; ++target) {
                noc_async_write(
                    get_semaphore(11),
                    core_l1(target, get_semaphore(11)), sizeof(uint32_t));
            }
            noc_async_write_barrier();
        }
        noc_semaphore_wait(release, 1);
        noc_semaphore_set(release, 0);
        rope_q.pop_front(dim_tiles);
        rope_k.pop_front(dim_tiles);
        gqa_reduced.pop_front(dim_tiles);
        WAYPOINT("AW4");
        return;
'''
    resident_body = resident_body.replace(r'\"', '"')
    bf16_start_marker = "        rope_k.wait_front(dim_tiles);"
    bf16_end_marker = '''        WAYPOINT("AW4");
        return;
'''
    if source.count(bf16_start_marker) != 1:
        raise RuntimeError("resident owner handoff marker changed")
    bf16_start = source.index(bf16_start_marker)
    bf16_end_start = source.index(bf16_end_marker, bf16_start)
    bf16_end = bf16_end_start + len(bf16_end_marker)
    return source[:bf16_start] + resident_body + source[bf16_end:]


def _projection_sources(
    *, replacements: dict[str, str], output_tensor_index: int,
    diagnostic_write_dout: bool, resident_qkv_handoff: bool = False,
    core_output_row_offset: int = 0,
) -> tuple[str, str, str]:
    cpp = Path(__file__).with_name("cpp")
    lora = _shift_compile_time_cb_indices(
        (cpp / "lora_backward_arena_exact_compute.cpp").read_text(), 4
    ).replace(
        "void kernel_main()", "FORCE_INLINE void lora_backward_phase()", 1
    )
    lora = lora.replace(
        "    // Phase 1: one complete output-feature reduction per token tile.",
        "    WAYPOINT(\"PDH\");\n"
        "    // Phase 1: one complete output-feature reduction per token tile.",
    ).replace(
        "    // Phase 2: one complete token reduction per output-feature tile.",
        "    WAYPOINT(\"PDB\");\n"
        "    // Phase 2: one complete token reduction per output-feature tile.",
    )
    base = (cpp / "frozen_linear_backward_base_phase.cpp").read_text().replace(
        "void kernel_main()", "FORCE_INLINE void resident_down_phase_one()", 1
    )
    helper = (
        cpp / "grouped2_frozen_lora_backward_tail_exact_compute.cpp"
    ).read_text().split("\nvoid kernel_main()", 1)[0]
    helper = helper.replace(
        "            dh.wait_front(reduction_block_tiles);\n",
        "            // Complete dH window is already resident.\n",
    ).replace(
        "                    cb_transposed_x,\n"
        "                    cb_dh,\n"
        "                    inner,\n"
        "                    inner,\n",
        "                    cb_transposed_x,\n"
        "                    cb_dh,\n"
        "                    inner,\n"
        "                    block * reduction_block_tiles + inner,\n",
    ).replace(
        "            dh.pop_front(reduction_block_tiles);\n",
        "            // Keep dH resident for LoRA dO.\n",
    )
    compute = (
        lora
        + "\n"
        + base
        + "\n"
        + helper
        + "\n"
        + (cpp / "resident_down_backward_tail.cpp").read_text()
    )
    compute = compute.replace(
        '#include "tools/profiler/kernel_profiler.hpp"',
        '#include "tools/profiler/kernel_profiler.hpp"\n'
        '#include "api/debug/waypoint.h"',
        1,
    ).replace(
        "        resident_down_phase_one();",
        "        WAYPOINT(\"PB0\");\n"
        "        resident_down_phase_one();\n"
        "        WAYPOINT(\"PB1\");",
        1,
    ).replace(
        "    resident_down_tail();",
        "    WAYPOINT(\"PT0\");\n"
        "    resident_down_tail();\n"
        "    WAYPOINT(\"PT1\");",
        1,
    )
    # The accepted resident-down sources normally cover the full 11x10
    # worker rectangle. This fused owner dedicates only the left 5x10 island
    # to the output projection, so every compute phase must use the same
    # compact node numbering as its reader/writer.
    compute = compute.replace(
        "constexpr uint32_t grid_cols = 11;",
        "constexpr uint32_t grid_cols = 5;",
    )
    reader = (cpp / "resident_down_backward_reader.cpp").read_text()
    reader = reader.replace(
        "constexpr uint32_t kGridCols = 11;\n"
        "constexpr uint32_t kWorkerNodes = 110;",
        "constexpr uint32_t kGridCols = 5;\n"
        "constexpr uint32_t kWorkerNodes = 50;",
    ).replace(
        "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
        "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};",
        "constexpr uint8_t kPhysicalX[kGridCols] = {1, 2, 3, 4, 5};",
    )
    reader = re.sub(
        r"get_tensor_accessor_args_cta_offset<(\d+), 25>\(\),\s*(\d+)",
        lambda match: (
            "get_tensor_accessor_args_cta_offset<"
            f"{match.group(1)}, {_TOTAL_CBS}>(), {match.group(2)}"
        ),
        reader,
    )
    writer = (cpp / "full_attention_output_handoff_writer.cpp").read_text()
    writer_replacements = dict(replacements)
    writer_replacements.update(
        {
            "@ATTENTION_DOUT_ROW_CB_ARG@": str(_PROJECTION_CBS + 2),
            "@ATTENTION_DOUT_FULL_CB_ARG@": str(_PROJECTION_CBS + 3),
            "@OUTPUT_TENSOR_INDEX@": str(output_tensor_index),
            "@TOTAL_CBS@": str(_TOTAL_CBS),
            "@DIAGNOSTIC_WRITE_DOUT@": (
                "1" if diagnostic_write_dout else "0"
            ),
            "@CORE_OUTPUT_ROW_OFFSET@": str(core_output_row_offset),
        }
    )
    if resident_qkv_handoff:
        reader = _append_before_kernel_end(
            reader, _projection_handoff_reader_tail()
        )
        writer = _append_before_kernel_end(
            writer, _projection_handoff_writer_tail()
        )
    return (
        _replace_all(reader, replacements),
        _replace_all(compute, replacements),
        _replace_all(writer, writer_replacements),
    )


def _attention_sources(
    *, replacements: dict[str, str], output_tensor_index: int,
    output_row_offset: int, output_arena_col_tiles: int,
    inverse_rope: bool, resident_qkv_handoff: bool = False,
) -> tuple[str, str, str]:
    cpp = Path(__file__).with_name("cpp")
    reader = (cpp / "attention_resident_backward_reader.cpp").read_text()
    compute = (cpp / "attention_resident_backward_compute.cpp").read_text()
    writer = (cpp / "attention_resident_backward_writer.cpp").read_text()

    if inverse_rope:
        reader = reader.replace(
            "    constexpr uint32_t cb_gqa_scaler = get_compile_time_arg_val(27);",
            "    constexpr uint32_t cb_gqa_scaler = get_compile_time_arg_val(27);\n"
            "    constexpr uint32_t cb_rope_cos = get_compile_time_arg_val(32);\n"
            "    constexpr uint32_t cb_rope_sin = get_compile_time_arg_val(33);",
            1,
        )
        accessor_marker = "    p_row.reserve_back(seq_tiles);\n"
        rope_inputs = """    const uint32_t cos_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t sin_addr = get_common_arg_val<uint32_t>(5);
    const auto cos_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<10, @TOTAL_CBS@>(), 4>(),
        cos_addr, kTileBytes);
    const auto sin_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<11, @TOTAL_CBS@>(), 5>(),
        sin_addr, kTileBytes);
    CircularBuffer rope_cos(cb_rope_cos), rope_sin(cb_rope_sin);
    rope_cos.reserve_back(dim_tiles);
    rope_sin.reserve_back(dim_tiles);
    for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
        const uint32_t page = row * dim_tiles + tile;
        noc0.async_read(
            cos_tensor,
            CoreLocalMem<uint32_t>(rope_cos.get_write_ptr() + tile * kTileBytes),
            kTileBytes, {.page_id = page}, {});
        noc0.async_read(
            sin_tensor,
            CoreLocalMem<uint32_t>(rope_sin.get_write_ptr() + tile * kTileBytes),
            kTileBytes, {.page_id = page}, {});
    }
    noc0.async_read_barrier();
    rope_cos.push_back(dim_tiles);
    rope_sin.push_back(dim_tiles);

"""
        if reader.count(accessor_marker) != 1:
            raise RuntimeError("attention input marker changed")
        reader = reader.replace(accessor_marker, rope_inputs + accessor_marker, 1)

        compute = compute.replace(
            "    constexpr uint32_t cb_p_bf16 = get_compile_time_arg_val(31);",
            "    constexpr uint32_t cb_p_bf16 = get_compile_time_arg_val(31);\n"
            "    constexpr uint32_t cb_rope_cos = get_compile_time_arg_val(32);\n"
            "    constexpr uint32_t cb_rope_sin = get_compile_time_arg_val(33);\n"
            "    constexpr uint32_t cb_rope_cos_product = get_compile_time_arg_val(34);\n"
            "    constexpr uint32_t cb_rope_sin_product = get_compile_time_arg_val(35);\n"
            "    constexpr uint32_t cb_rope_q = get_compile_time_arg_val(36);\n"
            "    constexpr uint32_t cb_rope_k = get_compile_time_arg_val(37);",
            1,
        )
        rope_compute_marker = "    WAYPOINT(\"AGD\");\n\n"
        rope_compute = """    WAYPOINT("AGD");

    // Inverse RoPE remains inside the attention island. Match the framework's
    // two separately packed BF16 products before the final add/sub boundary.
    CircularBuffer rope_cos(cb_rope_cos), rope_sin(cb_rope_sin);
    CircularBuffer dq_gradient(cb_dq), rope_q(cb_rope_q);
    compute_kernel_hw_startup(cb_dq, cb_rope_cos, cb_rope_cos_product);
    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
        const uint32_t peer = dim ^ 1u;
        sfpu_binary_tile_to_cb<0, true>(
            cb_dq, cb_rope_cos, cb_rope_cos_product, dim, dim);
        sfpu_binary_tile_to_cb<0, true>(
            cb_dq, cb_rope_sin, cb_rope_sin_product, peer, peer);
        if (dim == 0)
            fpu_add_tile_to_cb(
                cb_rope_cos_product, cb_rope_sin_product, cb_rope_q, 0, 0);
        else
            fpu_sub_tile_to_cb(
                cb_rope_cos_product, cb_rope_sin_product, cb_rope_q, 0);
        CircularBuffer cos_product(cb_rope_cos_product);
        CircularBuffer sin_product(cb_rope_sin_product);
        cos_product.pop_front(1);
        sin_product.pop_front(1);
    }
    dq_gradient.pop_front(dim_tiles);

    if (member == 0) {
        CircularBuffer reduced(cb_sum_dk_b), rope_k(cb_rope_k);
        compute_kernel_hw_startup(
            cb_sum_dk_b, cb_rope_cos, cb_rope_cos_product);
        for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
            const uint32_t peer = dim ^ 1u;
            sfpu_binary_tile_to_cb<0, true>(
                cb_sum_dk_b, cb_rope_cos, cb_rope_cos_product, dim, dim);
            sfpu_binary_tile_to_cb<0, true>(
                cb_sum_dk_b, cb_rope_sin, cb_rope_sin_product, peer, peer);
            if (dim == 0)
                fpu_add_tile_to_cb(
                    cb_rope_cos_product, cb_rope_sin_product, cb_rope_k, 0, 0);
            else
                fpu_sub_tile_to_cb(
                    cb_rope_cos_product, cb_rope_sin_product, cb_rope_k, 0);
            CircularBuffer cos_product(cb_rope_cos_product);
            CircularBuffer sin_product(cb_rope_sin_product);
            cos_product.pop_front(1);
            sin_product.pop_front(1);
        }
    }
    rope_cos.pop_front(dim_tiles);
    rope_sin.pop_front(dim_tiles);

"""
        if compute.count(rope_compute_marker) != 1:
            raise RuntimeError("attention GQA completion marker changed")
        compute = compute.replace(rope_compute_marker, rope_compute, 1)

        writer = writer.replace(
            "    constexpr uint32_t cb_p_bf16 = get_compile_time_arg_val(31);",
            "    constexpr uint32_t cb_p_bf16 = get_compile_time_arg_val(31);\n"
            "    constexpr uint32_t cb_rope_q = get_compile_time_arg_val(36);\n"
            "    constexpr uint32_t cb_rope_k = get_compile_time_arg_val(37);",
            1,
        ) if "cb_p_bf16" in writer else writer.replace(
            "    constexpr uint32_t cb_final_dv = get_compile_time_arg_val(30);",
            "    constexpr uint32_t cb_final_dv = get_compile_time_arg_val(30);\n"
            "    constexpr uint32_t cb_rope_q = get_compile_time_arg_val(36);\n"
            "    constexpr uint32_t cb_rope_k = get_compile_time_arg_val(37);",
            1,
        )
        writer = writer.replace(
            "    else dq.wait_front(dim_tiles);",
            "    else rope_q.wait_front(dim_tiles);",
            1,
        ).replace(
            "                    ? ds_handoff.get_read_ptr() : dq.get_read_ptr());",
            "                    ? ds_handoff.get_read_ptr() : rope_q.get_read_ptr());",
            1,
        ).replace(
            "        dq.pop_front(dim_tiles);",
            "        rope_q.pop_front(dim_tiles);",
            1,
        )
        writer = writer.replace(
            "    CircularBuffer dq(cb_dq), dv(cb_dv), dk(cb_dk);",
            "    CircularBuffer dq(cb_dq), dv(cb_dv), dk(cb_dk);\n"
            "    CircularBuffer rope_q(cb_rope_q), rope_k(cb_rope_k);",
            1,
        )
        old_final = """        for (uint32_t kind = 0; kind < 2; ++kind) {
            const uint32_t output_base = kind == 0 ? dk_base : dv_base;
            for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
                WAYPOINT("AW3");
                gqa_reduced.wait_front(1);
                const uint32_t page =
                    kv_head * vector_tiles + row * dim_tiles + tile;
                noc1.async_write(
                    CoreLocalMem<uint32_t>(gqa_reduced.get_read_ptr()),
                    output, kTileBytes, {},
                    {.page_id = output_base + page});
                noc1.async_write_barrier();
                gqa_reduced.pop_front(1);
            }
        }
"""
        new_final = """        rope_k.wait_front(dim_tiles);
        for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
            WAYPOINT("AW3");
            const uint32_t page =
                kv_head * vector_tiles + row * dim_tiles + tile;
            noc1.async_write(
                CoreLocalMem<uint32_t>(rope_k.get_read_ptr() + tile * kTileBytes),
                output, kTileBytes, {}, {.page_id = dk_base + page});
        }
        noc1.async_write_barrier();
        rope_k.pop_front(dim_tiles);
        // Skip the unrotated dK tiles retained at the front; only dV remains
        // a final core output without an inverse-RoPE phase.
        gqa_reduced.pop_front(dim_tiles);
        gqa_reduced.wait_front(dim_tiles);
        for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
            const uint32_t page =
                kv_head * vector_tiles + row * dim_tiles + tile;
            noc1.async_write(
                CoreLocalMem<uint32_t>(gqa_reduced.get_read_ptr() + tile * kTileBytes),
                output, kTileBytes, {}, {.page_id = dv_base + page});
        }
        noc1.async_write_barrier();
        gqa_reduced.pop_front(dim_tiles);
"""
        if writer.count(old_final) != 1:
            raise RuntimeError("attention BF16 final writer block changed")
        writer = writer.replace(old_final, new_final, 1)

        if resident_qkv_handoff:
            writer = _attention_writer_to_projection_handoff(writer)

    old_grid = (
        "constexpr uint32_t kGridCols = 11;"
    )
    old_x = (
        "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
        "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};"
    )
    for name, source in (("reader", reader), ("writer", writer)):
        source = source.replace(old_grid, "constexpr uint32_t kGridCols = 6;")
        source = source.replace(
            old_x,
            "constexpr uint8_t kPhysicalX[kGridCols] = "
            "{6, 7, 10, 11, 12, 13};",
        )
        if name == "reader":
            reader = source
        else:
            writer = source

    node_old = (
        "get_absolute_logical_y() * kGridCols + get_absolute_logical_x()"
    )
    node_new = (
        "get_absolute_logical_y() * kGridCols + "
        "(get_absolute_logical_x() - 5)"
    )
    reader = reader.replace(node_old, node_new)
    compute = compute.replace(node_old, node_new).replace(
        "get_absolute_logical_y() * 11 + get_absolute_logical_x()",
        "get_absolute_logical_y() * 6 + "
        "(get_absolute_logical_x() - 5)",
    )
    writer = writer.replace(node_old, node_new)

    # Remove the external dO tensor from the attention reader ABI. P/Q/K/V
    # become its four common arguments at positions 0..3.
    reader = reader.replace(
        "    const uint32_t dout_addr = get_common_arg_val<uint32_t>(1);\n",
        "",
    ).replace(
        "    const uint32_t q_addr = get_common_arg_val<uint32_t>(2);",
        "    const uint32_t q_addr = get_common_arg_val<uint32_t>(1);",
    ).replace(
        "    const uint32_t k_addr = get_common_arg_val<uint32_t>(3);",
        "    const uint32_t k_addr = get_common_arg_val<uint32_t>(2);",
    ).replace(
        "    const uint32_t v_addr = get_common_arg_val<uint32_t>(4);",
        "    const uint32_t v_addr = get_common_arg_val<uint32_t>(3);",
    )
    reader = re.sub(
        r"    const auto dout_tensor = TensorAccessor\(.*?dout_addr, kTileBytes\);\n",
        "",
        reader,
        flags=re.DOTALL,
    )
    tensor_map = {0: (6, 0), 2: (7, 1), 3: (8, 2), 4: (9, 3)}
    reader = re.sub(
        r"get_tensor_accessor_args_cta_offset<(\d+), 32>\(\),\s*(\d+)",
        lambda match: (
            "get_tensor_accessor_args_cta_offset<"
            f"{tensor_map[int(match.group(1))][0]}, {_TOTAL_CBS}>(), "
            f"{tensor_map[int(match.group(1))][1]}"
        ),
        reader,
    )

    old_dout = """    dout_row.reserve_back(dim_tiles);
    dout_full.reserve_back(vector_tiles);
    q_full.reserve_back(vector_tiles);
    for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
        noc0.async_read(
            dout_tensor,
            CoreLocalMem<uint32_t>(dout_full.get_write_ptr() + tile * kTileBytes),
            kTileBytes, {.page_id = head * vector_tiles + tile}, {});
        noc0.async_read(
            q_tensor,
            CoreLocalMem<uint32_t>(q_full.get_write_ptr() + tile * kTileBytes),
            kTileBytes, {.page_id = head * vector_tiles + tile}, {});
    }
    for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
        noc0.async_read(
            dout_tensor,
            CoreLocalMem<uint32_t>(dout_row.get_write_ptr() + tile * kTileBytes),
            kTileBytes,
            {.page_id = head * vector_tiles + row * dim_tiles + tile}, {});
    }
"""
    new_dout = """    dout_row.reserve_back(dim_tiles);
    dout_full.reserve_back(vector_tiles);
    WAYPOINT("OR0");
    // Publish reservation only after clearing the credit semaphore. Projection
    // node zero waits for every active consumer before releasing producers.
    volatile tt_l1_ptr uint32_t* dout_credit =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
    noc_semaphore_set(dout_credit, 0);

    // Establish a clean epoch for projection-node-zero's consumer-ready
    // counter before publishing any credits.  A program launch does not
    // guarantee that semaphore 13 is zero: unrelated programs can leave a
    // value at the same L1 address.  If the projection writer observes that
    // stale value, it can send dO before a slow attention reader clears its
    // local semaphore 12, and that clear permanently erases early credits.
    //
    // Semaphore 11 is otherwise unused on the attention island until the
    // final Q/K/V lifetime release, and semaphore 10 is unused until the
    // final projection-completion count.  Reuse them here for a two-stage
    // reader barrier, then leave both at zero for their later owners.
    constexpr uint32_t active_attention_readers = query_heads * seq_tiles;
    volatile tt_l1_ptr uint32_t* ready_epoch_count =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
    volatile tt_l1_ptr uint32_t* ready_epoch_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(11));
    if (node == 0) {
        noc_semaphore_set(ready_epoch_count, 0);
        noc_semaphore_set(ready_epoch_go, 1);
        for (uint32_t target = 1;
             target < active_attention_readers; ++target) {
            noc_async_write(
                get_semaphore(11), core_l1(target, get_semaphore(11)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(ready_epoch_go, 1);
    noc_semaphore_set(ready_epoch_go, 0);
    noc_semaphore_inc(core_l1(0, get_semaphore(10)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        noc_semaphore_wait_min(
            ready_epoch_count, active_attention_readers);
        noc_semaphore_set(ready_epoch_count, 0);
        // Reset the remote counter only after every attention reader has
        // cleared its own dO credit word.
        noc_async_write(
            get_semaphore(10),
            get_noc_addr(1, physical_y(0), get_semaphore(13)),
            sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_set(ready_epoch_go, 1);
        for (uint32_t target = 1;
             target < active_attention_readers; ++target) {
            noc_async_write(
                get_semaphore(11), core_l1(target, get_semaphore(11)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(ready_epoch_go, 1);
    noc_semaphore_set(ready_epoch_go, 0);
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* published_address =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
        noc_semaphore_set(published_address, dout_row.get_write_ptr());
        noc_async_write(
            get_semaphore(15),
            get_noc_addr(1, physical_y(0), get_semaphore(11)),
            sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_set(published_address, dout_full.get_write_ptr());
        noc_async_write(
            get_semaphore(15),
            get_noc_addr(1, physical_y(0), get_semaphore(12)),
            sizeof(uint32_t));
        noc_async_write_barrier();
    }
    noc_semaphore_inc(get_noc_addr(1, physical_y(0), get_semaphore(13)), 1);
    noc_async_atomic_barrier();
    WAYPOINT("OR1");
    q_full.reserve_back(vector_tiles);
    for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
        noc0.async_read(
            q_tensor,
            CoreLocalMem<uint32_t>(q_full.get_write_ptr() + tile * kTileBytes),
            kTileBytes, {.page_id = head * vector_tiles + tile}, {});
    }
"""
    if reader.count(old_dout) != 1:
        raise RuntimeError("attention dO reader block changed")
    reader = reader.replace(old_dout, new_dout)
    barrier_marker = (
        "    noc0.async_read_barrier();\n"
        "    p_row.push_back(seq_tiles);"
    )
    if reader.count(barrier_marker) != 1:
        raise RuntimeError("attention input barrier count changed")
    reader = reader.replace(
        barrier_marker,
        "    noc0.async_read_barrier();\n"
        "    WAYPOINT(\"OR2\");\n"
        "    noc_semaphore_wait_min(dout_credit, vector_tiles);\n"
        "    noc_semaphore_set(dout_credit, 0);\n"
        "    p_row.push_back(seq_tiles);",
        1,
    )
    reader = reader.replace(
        "for (uint32_t semaphore = 0; semaphore < 12; ++semaphore)",
        "for (uint32_t semaphore = 0; semaphore < 13; ++semaphore)",
    )

    # Each attention kernel receives compile-time CB arguments 25..62. The
    # corresponding configs retain local IDs 0..37 (with the dO/GQA swap).
    def shift_cb_args(source: str) -> str:
        return re.sub(
            r"get_compile_time_arg_val\((\d+)\)",
            lambda match: (
                "get_compile_time_arg_val("
                f"{_PROJECTION_CBS + int(match.group(1))})"
            ),
            source,
        )

    reader = shift_cb_args(reader)
    compute = shift_cb_args(compute)
    writer = shift_cb_args(writer)
    writer = writer.replace(
        "get_tensor_accessor_args_cta_offset<5, 32>(), 0>()",
        "get_tensor_accessor_args_cta_offset<"
        f"{output_tensor_index}, {_TOTAL_CBS}>(), 0>()",
    )
    page_helper_marker = (
        "    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;\n"
    )
    page_helper = page_helper_marker + (
        f"    constexpr uint32_t output_row_offset = {output_row_offset};\n"
        f"    constexpr uint32_t output_arena_col_tiles = "
        f"{output_arena_col_tiles};\n"
        "    auto output_page = [](uint32_t packed_page) {\n"
        "        return output_row_offset * output_arena_col_tiles +\n"
        "               (packed_page / dim_tiles) * output_arena_col_tiles +\n"
        "               packed_page % dim_tiles;\n"
        "    };\n"
    )
    if writer.count(page_helper_marker) != 1:
        raise RuntimeError("attention writer vector-tile marker changed")
    writer = writer.replace(page_helper_marker, page_helper, 1)
    writer = writer.replace(
        ".page_id = node * dim_tiles + tile",
        ".page_id = output_page(node * dim_tiles + tile)",
    ).replace(
        ".page_id = output_base + page",
        ".page_id = output_page(output_base + page)",
    ).replace(
        ".page_id = dk_base + page",
        ".page_id = output_page(dk_base + page)",
    ).replace(
        ".page_id = dv_base + page",
        ".page_id = output_page(dv_base + page)",
    ).replace(
        ".page_id = dk_base + kv_page",
        ".page_id = output_page(dk_base + kv_page)",
    ).replace(
        ".page_id = dv_base + kv_page",
        ".page_id = output_page(dv_base + kv_page)",
    )
    source_replacements = {
        "@TOTAL_CBS@": str(_TOTAL_CBS),
        **replacements,
    }
    return tuple(
        _replace_all(source, source_replacements)
        for source in (reader, compute, writer)
    )


def _full_attention_qkv_sources(
    *,
    projection_replacements: dict[str, str],
    attention_replacements: dict[str, str],
    sequence_tiles: int,
    hidden_tiles: int,
    kv_tiles: int,
    dimension_tiles: int,
    query_heads: int,
    kv_heads: int,
    scale_bits: int,
    base_phase: int,
    output_tensor_index: int,
) -> tuple[str, str, str, str, str, str]:
    """Compose one output-attention-QKV program without a DRAM boundary."""

    # Imported lazily because the isolated QKV builder reuses this module's
    # projection CB policy.
    from .tt_lang_qkv_resident_backward import (
        _compose_qkv_sources,
        _alias_equivalent_function,
        _merge_phased_kernel_source,
        _noinline_functions,
        _share_function_definitions,
        _share_projection_phase_one_by_width,
        _remap_qkv_reader_abi,
        _remap_qkv_writer_abi,
        _resident_attention_dy_reader_source,
        _resident_attention_qkv_writer_source,
    )

    output_row_offset = sequence_tiles + 1 + hidden_tiles
    projection = _projection_sources(
        replacements=projection_replacements,
        output_tensor_index=output_tensor_index,
        diagnostic_write_dout=False,
        resident_qkv_handoff=False,
        core_output_row_offset=output_row_offset,
    )
    attention = _attention_sources(
        replacements=attention_replacements,
        output_tensor_index=output_tensor_index,
        output_row_offset=output_row_offset,
        output_arena_col_tiles=hidden_tiles,
        inverse_rope=True,
        resident_qkv_handoff=True,
    )
    qkv_reader, qkv_compute, qkv_writer = _compose_qkv_sources(
        m_tiles=sequence_tiles,
        hidden_tiles=hidden_tiles,
        kv_tiles=kv_tiles,
        scale_bits=scale_bits,
        base_phase=base_phase,
        arena_index=16,
        phase_count=3,
    )
    # The standalone QKV builder uses ceil(total/50).  The full owner may
    # deliberately select a complete per-core FIFO epoch (for Qwen, four
    # jobs on 28 active cores instead of 3/3/.../1 on 38 cores).  Apply that
    # schedule consistently to all three RISCs before sharing equivalent
    # projection functions.
    standalone_jobs = -(-(sequence_tiles * hidden_tiles) // 50)
    owner_jobs = int(projection_replacements["@JOBS_PER_NODE@"]) 
    if owner_jobs != standalone_jobs:
        old_jobs = f"constexpr uint32_t jobs_per_node = {standalone_jobs};"
        new_jobs = f"constexpr uint32_t jobs_per_node = {owner_jobs};"
        qkv_reader = qkv_reader.replace(old_jobs, new_jobs)
        qkv_compute = qkv_compute.replace(old_jobs, new_jobs)
        qkv_writer = qkv_writer.replace(old_jobs, new_jobs)
    # The standalone serial program reuses CB8 as both dH handoff and exact
    # accumulation scratch because a program boundary gives it a narrow,
    # predictable epoch.  Inside the full owner the reader and compute RISCs
    # overlap across phases, so K can reserve scratch while CB8 still carries
    # the dH tile being multicast.  CB26 is a dedicated projection-local
    # scratch window in this owner; moving both K and Q there changes no
    # arithmetic or BF16 boundary.
    qkv_scratch = (
        "constexpr uint32_t cb_accum_scratch = get_compile_time_arg_val(8);"
    )
    if qkv_compute.count(qkv_scratch) != 2:
        raise RuntimeError("serial QKV exact-add scratch contract changed")
    qkv_compute = qkv_compute.replace(
        qkv_scratch,
        "constexpr uint32_t cb_accum_scratch = 26;",
        1,
    ).replace(
        qkv_scratch,
        "constexpr uint32_t cb_accum_scratch = 31;",
        1,
    )
    qkv_compute_fence = (
        "CircularBuffer phase_done(get_compile_time_arg_val(7));"
    )
    qkv_writer_fence = (
        "CircularBuffer phase_done_cb(get_compile_time_arg_val(7));"
    )
    if qkv_compute.count(qkv_compute_fence) != 2:
        raise RuntimeError("serial QKV compute-fence contract changed")
    if qkv_writer.count(qkv_writer_fence) != 3:
        raise RuntimeError("serial QKV writer-fence contract changed")
    q_writer_start = qkv_writer.index("FORCE_INLINE void q_writer_phase() {")
    q_writer_end = qkv_writer.index("void kernel_main()", q_writer_start)
    q_writer_body = qkv_writer[q_writer_start:q_writer_end]
    remapped_q_writer_body = q_writer_body.replace(
        "constexpr uint32_t cb_db = get_compile_time_arg_val(15);",
        "constexpr uint32_t cb_db = 34;",
        1,
    ).replace(
        "constexpr uint32_t cb_da = get_compile_time_arg_val(22);",
        "constexpr uint32_t cb_da = 35;",
        1,
    ).replace(
        "constexpr uint32_t cb_final_dx = get_compile_time_arg_val(24);",
        "constexpr uint32_t cb_final_dx = 56;",
        1,
    )
    if remapped_q_writer_body == q_writer_body:
        raise RuntimeError("Q writer output-FIFO remap did not change")
    qkv_writer = (
        qkv_writer[:q_writer_start] + remapped_q_writer_body
        + qkv_writer[q_writer_end:]
    )
    compute_publish = (
        "phase_done.reserve_back(1);\n    phase_done.push_back(1);"
    )
    if qkv_compute.count(compute_publish) != 2:
        raise RuntimeError("serial QKV compute publication changed")
    def raw_phase_publish(physical_cb: int, indent: str = "") -> str:
        return (
            f"{indent}CircularBuffer phase_done({physical_cb});\n"
            f'{indent}DPRINT_PACK("phase raw enter cb={{}}\\n", '
            f"{physical_cb});\n"
            f"{indent}PACK((*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
            f"{indent}        phase_done.get_write_ptr() <<\n"
            f"{indent}        LOG_BASE_2_OF_L1_ALIGNMENT) = 1));\n"
            f'{indent}DPRINT_PACK("phase raw write cb={{}} addr={{}} val={{}}\\n",\n'
            f"{indent}            {physical_cb},\n"
            f"{indent}            phase_done.get_write_ptr() <<\n"
            f"{indent}                LOG_BASE_2_OF_L1_ALIGNMENT,\n"
            f"{indent}            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
            f"{indent}                phase_done.get_write_ptr() <<\n"
            f"{indent}                LOG_BASE_2_OF_L1_ALIGNMENT));"
        )

    fence_block = qkv_compute_fence + "\n    " + compute_publish
    kv_publication = (
        "if (accumulate) {\n"
        + raw_phase_publish(28, "        ")
        + "\n    } else {\n"
        + raw_phase_publish(25, "        ")
        + "\n    }"
    )
    q_publication = raw_phase_publish(29)
    if qkv_compute.count(fence_block) != 2:
        raise RuntimeError("serial QKV complete fence blocks changed")
    qkv_compute = qkv_compute.replace(
        fence_block, kv_publication, 1
    ).replace(
        fence_block, q_publication, 1
    )
    writer_consume = (
        "phase_done_cb.wait_front(1);\n"
        "    phase_done_cb.pop_front(1);"
    )
    if qkv_writer.count(writer_consume) != 3:
        raise RuntimeError("serial QKV writer consumption changed")
    # Writer source order is V, K, Q. Each polls and clears one independent L1
    # word; no CB pointers or producer/consumer counters are advanced.
    for physical_cb in (25, 28, 29):
        qkv_writer = qkv_writer.replace(
            qkv_writer_fence,
            f"CircularBuffer phase_done_cb({physical_cb});",
            1,
        ).replace(
            writer_consume,
            "volatile tt_l1_ptr uint32_t* phase_done_word =\n"
            "        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
            "            phase_done_cb.get_read_ptr());\n"
            "    DPRINT_DATA1(\"phase raw wait addr={} val={}\\n\",\n"
            "                 phase_done_cb.get_read_ptr(),\n"
            "                 *phase_done_word);\n"
            "    while (*phase_done_word != 1) {}\n"
            "    *phase_done_word = 0;",
            1,
        )
    if qkv_writer_fence in qkv_writer or writer_consume in qkv_writer:
        raise RuntimeError("serial QKV writer fence replacement incomplete")
    shared_compute_helpers = (
        "add_tiles_bf16", "commit_bf16", "pack_reconfig_bf16",
        "pack_bf16", "release_bf16", "run_lora_dx",
        "run_bf16_add", "run_da",
    )
    qkv_compute = _share_function_definitions(
        projection[1],
        qkv_compute,
        shared_compute_helpers,
    )
    # Q has the same output width and exact schedule as the output projection.
    # Reuse those bodies instead of storing a second 14-KiB source copy in the
    # same TRISC program. The diagnostic waypoints are removed before the
    # equivalence check; they do not participate in arithmetic or CB state.
    projection = (
        projection[0],
        re.sub(r'\s*WAYPOINT\("P(?:DH|DB)"\);', "", projection[1]),
        projection[2],
    )
    for owner_name, consumer_name in (
        ("lora_backward_phase", "q_lora_backward_phase"),
        ("resident_down_phase_one", "q_resident_down_phase_one"),
        ("run_resident_down_lora_dx", "q_run_resident_down_lora_dx"),
    ):
        qkv_compute = _alias_equivalent_function(
            projection[1],
            qkv_compute,
            owner_name=owner_name,
            consumer_name=consumer_name,
        )
    # The standalone sources deliberately force-inline every phase. In the
    # four-projection owner that clones the same exact schedule at each call
    # site and overflows Blackhole's kernel-config text budget (98,592 bytes
    # versus 70,656). Preserve the arithmetic and CB schedule but share the
    # high-level phase bodies as ordinary device functions.
    projection = (
        projection[0],
        _noinline_functions(
            projection[1],
            (
                "lora_backward_phase",
                "resident_down_phase_one",
                "run_lora_dx",
                "run_bf16_add",
                "run_da",
                "run_resident_down_lora_dx",
                "resident_down_tail",
            ),
            optimize="Os",
        ),
        projection[2],
    )
    projection_compute_release = f"""
    // Output dH has reached both terminal consumers (dA and LoRA dO). The
    // split path reclaimed CB16 at program teardown; the one-call owner must
    // end that lifetime explicitly before V reuses the same local buffer.
    CircularBuffer output_resident_dh(get_compile_time_arg_val(16));
    output_resident_dh.pop_front({sequence_tiles});
"""
    projection = (
        projection[0],
        _append_before_kernel_end(
            projection[1], projection_compute_release
        ),
        projection[2],
    )
    qkv_compute = _noinline_functions(
        qkv_compute,
        (
            "kv_lora_backward_phase",
            "kv_resident_down_phase_one",
            "kv_run_resident_down_lora_dx",
            "kv_resident_down_tail",
            "q_resident_down_tail",
        ),
        optimize="Os",
    )
    qkv_compute = _share_projection_phase_one_by_width(
        _merge_phased_kernel_source(projection[1], qkv_compute),
        hidden_tiles=hidden_tiles,
        kv_tiles=kv_tiles,
        workers=50,
    )
    # CB4/CB9 carry the scaled-gradient inputs for LoRA dH and dB. In this
    # owner Q is their fourth producer/consumer epoch (O -> V -> K -> Q).
    # Although the allocation has enough physical history, silicon parity
    # showed that the fourth epoch consumed an old window without hanging.
    # Give Q clean input FIFOs while retaining the exact qualified arithmetic
    # and every other buffer. This adds only N+M BF16 tiles to the live plan.
    lora_start = qkv_compute.index(
        '__attribute__((optimize("Os"), noinline)) void '
        "lora_backward_phase("
    )
    lora_brace = qkv_compute.index("{", lora_start)
    lora_end = -1
    depth = 0
    for position in range(lora_brace, len(qkv_compute)):
        depth += (qkv_compute[position] == "{") - (
            qkv_compute[position] == "}"
        )
        if depth == 0:
            lora_end = position + 1
            break
    if lora_end < 0:
        raise RuntimeError("unterminated shared LoRA phase")
    lora_body = qkv_compute[lora_start:lora_end]
    q_lora_body = lora_body.replace(
        "lora_backward_phase(", "q_lora_backward_phase(", 1
    ).replace(
        "constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(4);",
        "constexpr uint32_t cb_dh_dy = 32;",
        1,
    ).replace(
        "constexpr uint32_t cb_db_dy = get_compile_time_arg_val(9);",
        "constexpr uint32_t cb_db_dy = 33;",
        1,
    ).replace(
        "constexpr uint32_t cb_db_output = get_compile_time_arg_val(15);",
        "constexpr uint32_t cb_db_output = 34;",
        1,
    ).replace(
        "constexpr uint32_t cb_dh_b = get_compile_time_arg_val(5);",
        "constexpr uint32_t cb_dh_b = 36;",
        1,
    ).replace(
        "constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(10);",
        "constexpr uint32_t cb_db_hidden = 37;",
        1,
    ).replace(
        "constexpr uint32_t cb_dh_scaled = get_compile_time_arg_val(6);",
        "constexpr uint32_t cb_dh_scaled = 38;",
        1,
    ).replace(
        "constexpr uint32_t cb_dh_partial = get_compile_time_arg_val(7);",
        "constexpr uint32_t cb_dh_partial = 39;",
        1,
    ).replace(
        "constexpr uint32_t cb_dh_output = get_compile_time_arg_val(8);",
        "constexpr uint32_t cb_dh_output = 40;",
        1,
    ).replace(
        "constexpr uint32_t cb_db_scaled = get_compile_time_arg_val(11);",
        "constexpr uint32_t cb_db_scaled = 41;",
        1,
    ).replace(
        "constexpr uint32_t cb_db_transposed_hidden =\n"
        "        get_compile_time_arg_val(12);",
        "constexpr uint32_t cb_db_transposed_hidden = 42;",
        1,
    ).replace(
        "constexpr uint32_t cb_db_partial = get_compile_time_arg_val(13);",
        "constexpr uint32_t cb_db_partial = 43;",
        1,
    ).replace(
        "constexpr uint32_t cb_db_pretranspose =\n"
        "        get_compile_time_arg_val(14);",
        "constexpr uint32_t cb_db_pretranspose = 44;",
        1,
    )
    if q_lora_body == lora_body:
        raise RuntimeError("Q LoRA input-FIFO clone did not change")
    qkv_compute = (
        qkv_compute[:lora_end] + "\n\n" + q_lora_body
        + qkv_compute[lora_end:]
    )
    q_tail_start = qkv_compute.index("void q_resident_down_tail() {")
    q_tail_brace = qkv_compute.index("{", q_tail_start)
    q_tail_end = -1
    depth = 0
    for position in range(q_tail_brace, len(qkv_compute)):
        depth += (qkv_compute[position] == "{") - (
            qkv_compute[position] == "}"
        )
        if depth == 0:
            q_tail_end = position + 1
            break
    if q_tail_end < 0:
        raise RuntimeError("unterminated Q projection tail")
    q_tail_body = qkv_compute[q_tail_start:q_tail_end]
    remapped_q_tail_body = q_tail_body.replace(
        "constexpr uint32_t cb_base = get_compile_time_arg_val(3);",
        "constexpr uint32_t cb_base = 47;",
        1,
    ).replace(
        "constexpr uint32_t cb_dh = get_compile_time_arg_val(16);",
        "constexpr uint32_t cb_dh = 49;",
        1,
    ).replace(
        "constexpr uint32_t cb_a = get_compile_time_arg_val(17);",
        "constexpr uint32_t cb_a = 50;",
        1,
    ).replace(
        "constexpr uint32_t cb_x = get_compile_time_arg_val(18);",
        "constexpr uint32_t cb_x = 51;",
        1,
    ).replace(
        "constexpr uint32_t cb_transposed_x = get_compile_time_arg_val(19);",
        "constexpr uint32_t cb_transposed_x = 52;",
        1,
    ).replace(
        "constexpr uint32_t cb_da_partial = get_compile_time_arg_val(20);",
        "constexpr uint32_t cb_da_partial = 53;",
        1,
    ).replace(
        "constexpr uint32_t cb_da_pretranspose = get_compile_time_arg_val(21);",
        "constexpr uint32_t cb_da_pretranspose = 54;",
        1,
    ).replace(
        "constexpr uint32_t cb_da = get_compile_time_arg_val(22);",
        "constexpr uint32_t cb_da = 35;",
        1,
    ).replace(
        "constexpr uint32_t cb_lora_dx = get_compile_time_arg_val(23);",
        "constexpr uint32_t cb_lora_dx = 55;",
        1,
    ).replace(
        "constexpr uint32_t cb_final_dx = get_compile_time_arg_val(24);",
        "constexpr uint32_t cb_final_dx = 56;\n"
        "    constexpr uint32_t cb_k_accum_dx = 30;",
        1,
    )
    if remapped_q_tail_body == q_tail_body:
        raise RuntimeError("Q dA output-FIFO remap did not change")
    qkv_compute = (
        qkv_compute[:q_tail_start] + remapped_q_tail_body
        + qkv_compute[q_tail_end:]
    )
    # O, V and K complete balanced CB2 producer/consumer cycles. Q is the
    # fourth independent epoch and must not inherit CB2's pack/unpack counter
    # history. Clone only Q's small frozen-base phase and bind its partial FIFO
    # statically to projection-local CB27. Low-level Tensix reconfiguration
    # keeps a compile-time CB ID and every reduction/pack site stays unchanged.
    owner_start = qkv_compute.index(
        '__attribute__((optimize("Os"), noinline)) void '
        "resident_down_phase_one("
    )
    owner_brace = qkv_compute.index("{", owner_start)
    owner_end = -1
    depth = 0
    for position in range(owner_brace, len(qkv_compute)):
        depth += (qkv_compute[position] == "{") - (
            qkv_compute[position] == "}"
        )
        if depth == 0:
            owner_end = position + 1
            break
    if owner_end < 0:
        raise RuntimeError("unterminated shared frozen-base phase")
    owner_body = qkv_compute[owner_start:owner_end]
    q_owner_body = owner_body.replace(
        "resident_down_phase_one(", "q_resident_down_phase_one(", 1
    ).replace(
        "constexpr uint32_t cb_dy = get_compile_time_arg_val(0);",
        "constexpr uint32_t cb_dy = 45;",
        1,
    ).replace(
        "constexpr uint32_t cb_weight = get_compile_time_arg_val(1);",
        "constexpr uint32_t cb_weight = 46;",
        1,
    ).replace(
        "constexpr uint32_t cb_partial = get_compile_time_arg_val(2);",
        "constexpr uint32_t cb_partial = 48;",
        1,
    ).replace(
        "constexpr uint32_t cb_result = get_compile_time_arg_val(3);",
        "constexpr uint32_t cb_result = 47;",
        1,
    )
    q_owner_body = q_owner_body.replace(
        "{\n    constexpr uint32_t cb_dy",
        '{\n    DPRINT_UNPACK("Q base start\\n");\n'
        "    constexpr uint32_t cb_dy",
        1,
    ).replace(
        "    matmul_block_init(\n"
        "        cb_dy,\n"
        "        cb_weight,\n"
        "        false,\n"
        "        1,\n"
        "        1,\n"
        "        reduction_tiles_per_phase);",
        "    matmul_block_init(\n"
        "        cb_dy,\n"
        "        cb_weight,\n"
        "        false,\n"
        "        1,\n"
        "        1,\n"
        "        reduction_tiles_per_phase);\n"
        '    DPRINT_UNPACK("Q base startup cb={}\\n", cb_partial);',
        1,
    ).replace(
        "        for (uint32_t block = 0; block < reduction_blocks; ++block) {\n"
        "            dy.wait_front(reduction_tiles_per_phase);\n"
        "            weight.wait_front(reduction_tiles_per_phase);",
        "        for (uint32_t block = 0; block < reduction_blocks; ++block) {\n"
        '            DPRINT_UNPACK("Q base pre j={} b={} cb={}\\n", '\
        "local_job, block, cb_partial);\n"
        "            dy.wait_front(reduction_tiles_per_phase);\n"
        "            weight.wait_front(reduction_tiles_per_phase);\n"
        '            DPRINT_UNPACK("Q base inputs j={} b={}\\n", '\
        "local_job, block);",
        1,
    ).replace(
        "            tile_regs_acquire();\n"
        "            if (block > 0) {",
        '            DPRINT_UNPACK("Q base acquire pre j={} b={}\\n", '\
        "local_job, block);\n"
        "            tile_regs_acquire();\n"
        '            DPRINT_UNPACK("Q base acquire done j={} b={}\\n", '\
        "local_job, block);\n"
        "            if (block > 0) {",
        1,
    ).replace(
        "                copy_tile_to_dst_init_short_with_dt(\n"
        "                    cb_weight, cb_partial);\n"
        "                partial.wait_front(1);",
        "                copy_tile_to_dst_init_short_with_dt(\n"
        "                    cb_weight, cb_partial);\n"
        '                DPRINT_UNPACK("Q base partial wait j={} b={}\\n", '\
        "local_job, block);\n"
        "                partial.wait_front(1);\n"
            '                DPRINT_UNPACK("Q base partial got j={} b={}\\n", '\
            "local_job, block);",
            1,
        )
    q_owner_body = q_owner_body.replace(
        "    lora_backward_phase(projection_n_tiles, projection_workers);",
        "    q_lora_backward_phase(projection_n_tiles, projection_workers);",
        1,
    )
    if q_owner_body == owner_body:
        raise RuntimeError("Q frozen-base clone did not change")
    qkv_compute = (
        qkv_compute[:owner_end] + "\n\n" + q_owner_body
        + qkv_compute[owner_end:]
    )
    hidden_base_call = f"resident_down_phase_one({hidden_tiles}, 50);"
    if qkv_compute.count(hidden_base_call) != 2:
        raise RuntimeError("output/Q shared frozen-base calls changed")
    # Source order is Q phase followed by the outer output-projection owner.
    qkv_compute = qkv_compute.replace(
        hidden_base_call,
        f"q_resident_down_phase_one({hidden_tiles}, 50);",
        1,
    )
    # Keep the exact V -> K -> Q BF16 addition order, but never turn the same
    # CB directly from an unpack input into a pack output. V leaves dX in
    # CB24, K consumes it and publishes to CB30, then Q consumes CB30 and
    # publishes the final accumulated dHidden back to the now-empty CB24.
    final_dx_decl = (
        "    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(24);"
    )
    # Q already received its terminal-only final CB above. V and K retain the
    # shared declaration and receive the K accumulator identifier here.
    final_dx_decl_count = qkv_compute.count(final_dx_decl)
    if final_dx_decl_count != 2:
        raise RuntimeError(
            "V/K/Q final dX declarations changed: "
            f"expected 2 shared declarations, found {final_dx_decl_count}"
        )
    qkv_compute = qkv_compute.replace(
        final_dx_decl,
        final_dx_decl + "\n    constexpr uint32_t cb_k_accum_dx = 30;",
    )
    prior_add = """        run_bf16_add(
            cb_final_dx, cb_lora_dx, cb_accum_scratch,
            jobs_per_node, total_jobs, node);"""
    final_add = """        run_bf16_add(
            cb_accum_scratch, cb_base, cb_final_dx,
            jobs_per_node, total_jobs, node);"""
    if qkv_compute.count(prior_add) != 2 or qkv_compute.count(final_add) != 2:
        raise RuntimeError("K/Q accumulated dX calls changed")
    # First occurrence is K; second occurrence is Q.
    second_prior = qkv_compute.find(prior_add, qkv_compute.find(prior_add) + 1)
    q_prior_add = prior_add.replace("cb_final_dx,", "cb_k_accum_dx,", 1)
    qkv_compute = (
        qkv_compute[:second_prior] + q_prior_add
        + qkv_compute[second_prior + len(prior_add):]
    )
    first_final = qkv_compute.find(final_add)
    k_final_add = final_add.replace(
        "cb_accum_scratch, cb_base, cb_final_dx,",
        "cb_accum_scratch, cb_base, cb_k_accum_dx,",
        1,
    )
    qkv_compute = (
        qkv_compute[:first_final] + k_final_add
        + qkv_compute[first_final + len(final_add):]
    )
    lora_dh_marker = (
        "    // Phase 1: one complete output-feature reduction per token tile."
    )
    lora_db_marker = (
        "    // Phase 2: one complete token reduction per output-feature tile."
    )
    if qkv_compute.count(lora_dh_marker) != 2 or qkv_compute.count(
        lora_db_marker
    ) != 2:
        raise RuntimeError("shared/Q LoRA phase markers changed")
    qkv_compute = qkv_compute.replace(
        lora_dh_marker,
        '    DPRINT_UNPACK("LoRA dH start n={}\\n", projection_n_tiles);\n'
        + lora_dh_marker,
        1,
    ).replace(
        lora_db_marker,
        '    DPRINT_UNPACK("LoRA dH done, dB start n={}\\n", projection_n_tiles);\n'
        + lora_db_marker,
        1,
    )
    base_start = qkv_compute.index(
        '__attribute__((optimize("Os"), noinline)) void '
        "resident_down_phase_one("
    )
    base_brace = qkv_compute.index("{", base_start)
    depth = 0
    base_end = -1
    for position in range(base_brace, len(qkv_compute)):
        depth += (qkv_compute[position] == "{") - (
            qkv_compute[position] == "}"
        )
        if depth == 0:
            base_end = position + 1
            break
    if base_end < 0:
        raise RuntimeError("unterminated shared frozen-base phase")
    base_body = qkv_compute[base_start:base_end]
    base_body = base_body.replace(
        "{\n    constexpr uint32_t cb_dy",
        '{\n    DPRINT_UNPACK("base dX start n={}\\n", projection_n_tiles);\n'
        "    constexpr uint32_t cb_dy",
        1,
    ).replace(
        "    lora_backward_phase(projection_n_tiles, projection_workers);",
        '    DPRINT_UNPACK("base dX done n={}\\n", projection_n_tiles);\n'
        "    lora_backward_phase(projection_n_tiles, projection_workers);",
        1,
    )
    base_wait_marker = """        for (uint32_t block = 0; block < reduction_blocks; ++block) {
            dy.wait_front(reduction_tiles_per_phase);
            weight.wait_front(reduction_tiles_per_phase);"""
    base_wait_trace = """        for (uint32_t block = 0; block < reduction_blocks; ++block) {
            DPRINT_UNPACK("base pre n={} j={} b={}\\n",
                          projection_n_tiles, local_job, block);
            dy.wait_front(reduction_tiles_per_phase);
            weight.wait_front(reduction_tiles_per_phase);
            DPRINT_UNPACK("base inputs n={} j={} b={}\\n",
                          projection_n_tiles, local_job, block);"""
    if base_body.count(base_wait_marker) != 1:
        raise RuntimeError("shared frozen-base wait loop changed")
    base_body = base_body.replace(base_wait_marker, base_wait_trace, 1)
    acquire_marker = """            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_weight, cb_partial);
                partial.wait_front(1);
                copy_block_matmul_partials(cb_partial, 0, 0, 1);
                partial.pop_front(1);"""
    acquire_trace = """            DPRINT_UNPACK("base acquire pre n={} j={} b={}\\n",
                          projection_n_tiles, local_job, block);
            tile_regs_acquire();
            DPRINT_UNPACK("base acquire done n={} j={} b={}\\n",
                          projection_n_tiles, local_job, block);
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_weight, cb_partial);
                DPRINT_UNPACK("base partial configured n={} j={} b={}\\n",
                              projection_n_tiles, local_job, block);
                DPRINT_UNPACK("base partial wait n={} j={} b={}\\n",
                              projection_n_tiles, local_job, block);
                partial.wait_front(1);
                DPRINT_UNPACK("base partial got n={} j={} b={}\\n",
                              projection_n_tiles, local_job, block);
                copy_block_matmul_partials(cb_partial, 0, 0, 1);
                partial.pop_front(1);
                DPRINT_UNPACK("base partial popped n={} j={} b={}\\n",
                              projection_n_tiles, local_job, block);"""
    if base_body.count(acquire_marker) != 1:
        raise RuntimeError("shared frozen-base acquire/partial block changed")
    base_body = base_body.replace(acquire_marker, acquire_trace, 1)
    base_pop_marker = """            weight.pop_front(reduction_tiles_per_phase);
            dy.pop_front(reduction_tiles_per_phase);"""
    base_pop_trace = base_pop_marker + """
            DPRINT_UNPACK("base post n={} j={} b={}\\n",
                          projection_n_tiles, local_job, block);"""
    if base_body.count(base_pop_marker) != 1:
        raise RuntimeError("shared frozen-base pop loop changed")
    base_body = base_body.replace(base_pop_marker, base_pop_trace, 1)
    reserve_marker = """            if (last) {
                result.reserve_back(1);
            } else {
                partial.reserve_back(1);
            }
            tile_regs_wait();"""
    reserve_trace = """            DPRINT_UNPACK("base reserve pre n={} j={} b={}\\n",
                          projection_n_tiles, local_job, block);
            if (last) {
                result.reserve_back(1);
            } else {
                partial.reserve_back(1);
            }
            DPRINT_UNPACK("base reserve done n={} j={} b={}\\n",
                          projection_n_tiles, local_job, block);
            tile_regs_wait();"""
    if base_body.count(reserve_marker) != 1:
        raise RuntimeError("shared frozen-base reserve branch changed")
    base_body = base_body.replace(reserve_marker, reserve_trace, 1)
    qkv_compute = qkv_compute[:base_start] + base_body + qkv_compute[base_end:]
    qkv_compute = '#include "api/debug/dprint.h"\n' + qkv_compute
    first_exact_add = """        run_bf16_add(
            cb_final_dx, cb_lora_dx, cb_accum_scratch,
            jobs_per_node, total_jobs, node);"""
    second_exact_add = """        run_bf16_add(
            cb_accum_scratch, cb_base, cb_final_dx,
            jobs_per_node, total_jobs, node);"""
    if qkv_compute.count(first_exact_add) != 1 or qkv_compute.count(
        second_exact_add
    ) != 1:
        raise RuntimeError("K/Q exact accumulation path changed")
    qkv_compute = qkv_compute.replace(
        first_exact_add,
        '        DPRINT_PACK("exact add one enter\\n");\n'
        + first_exact_add
        + '\n        DPRINT_PACK("exact add one done\\n");',
    ).replace(
        second_exact_add,
        '        DPRINT_PACK("exact add two enter\\n");\n'
        + second_exact_add
        + '\n        DPRINT_PACK("exact add two done\\n");',
    )
    kv_compute_start = qkv_compute.index(
        "static __attribute__((noinline, noclone)) void "
        "kv_compute_phase(bool accumulate) {"
    )
    kv_compute_brace = qkv_compute.index("{", kv_compute_start)
    depth = 0
    kv_compute_end = -1
    for position in range(kv_compute_brace, len(qkv_compute)):
        depth += (qkv_compute[position] == "{") - (
            qkv_compute[position] == "}"
        )
        if depth == 0:
            kv_compute_end = position + 1
            break
    if kv_compute_end < 0:
        raise RuntimeError("unterminated KV compute function")
    kv_compute_body = qkv_compute[kv_compute_start:kv_compute_end]
    kv_compute_body = kv_compute_body.replace(
        "{\n    {\n        DeviceZoneScopedN(\"resident_down_phase_one_compute\");",
        "{\n"
        '    DPRINT_UNPACK("KV compute enter accumulate={}\\n", accumulate);\n'
        "    {\n        DeviceZoneScopedN(\"resident_down_phase_one_compute\");",
        1,
    ).replace(
        "    kv_resident_down_tail(accumulate);",
        '    DPRINT_UNPACK("KV compute phase one done accumulate={}\\n", accumulate);\n'
        '    DPRINT_PACK("KV pack tail enter accumulate={}\\n", accumulate);\n'
        "    kv_resident_down_tail(accumulate);\n"
        '    DPRINT_PACK("KV pack tail done accumulate={}\\n", accumulate);\n'
        '    DPRINT_UNPACK("KV compute tail done accumulate={}\\n", accumulate);',
        1,
    ).replace(
        "    resident_dh.pop_front(",
        '    DPRINT_UNPACK("KV compute releasing resident dH\\n");\n'
        "    resident_dh.pop_front(",
        1,
    ).replace(
        "    phase_done.push_back(1);",
        "    phase_done.push_back(1);\n"
        '    DPRINT_UNPACK("KV compute phase fence published accumulate={}\\n", accumulate);',
        1,
    )
    qkv_compute = (
        qkv_compute[:kv_compute_start] + kv_compute_body
        + qkv_compute[kv_compute_end:]
    )
    q_compute_start = qkv_compute.index(
        "static __attribute__((noinline, noclone)) void q_compute_phase() {"
    )
    q_compute_brace = qkv_compute.index("{", q_compute_start)
    depth = 0
    q_compute_end = -1
    for position in range(q_compute_brace, len(qkv_compute)):
        depth += (qkv_compute[position] == "{") - (
            qkv_compute[position] == "}"
        )
        if depth == 0:
            q_compute_end = position + 1
            break
    if q_compute_end < 0:
        raise RuntimeError("unterminated Q compute function")
    q_compute_body = qkv_compute[q_compute_start:q_compute_end]
    q_compute_body = q_compute_body.replace(
        "{\n    {\n        DeviceZoneScopedN(\"resident_down_phase_one_compute\");",
        "{\n"
        '    DPRINT_UNPACK("Q compute enter\\n");\n'
        "    {\n        DeviceZoneScopedN(\"resident_down_phase_one_compute\");",
        1,
    ).replace(
        "    q_resident_down_tail();",
        '    DPRINT_UNPACK("Q compute phase one done\\n");\n'
        "    q_resident_down_tail();\n"
        '    DPRINT_UNPACK("Q compute tail done\\n");',
        1,
    ).replace(
        "    phase_done.push_back(1);",
        "    phase_done.push_back(1);\n"
        '    DPRINT_UNPACK("Q compute phase fence published\\n");',
        1,
    ).replace(
        "CircularBuffer resident_dh(get_compile_time_arg_val(16));",
        "CircularBuffer resident_dh(49);",
        1,
    )
    qkv_compute = (
        qkv_compute[:q_compute_start] + q_compute_body
        + qkv_compute[q_compute_end:]
    )
    # The projection reader finishes loading before its compute/writer RISCs
    # have completed phase zero.  Without an explicit cross-RISC fence it can
    # enter QKV and observe transient values in the projection's semaphore
    # map as attention-ready credits.  Semaphore 15 is unused on the left
    # island; each output writer releases only its co-located QKV reader after
    # every dO payload and credit has been sent.
    projection_writer_start = """void kernel_main() {
    // Establish the output-writer ring mailbox before any writer can reach
    // the final handoff.  Every writer executes this as its first action;
    // the long output phase follows, so the token cannot race initialization.
    volatile tt_l1_ptr uint32_t* output_phase_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
    noc_semaphore_set(output_phase_done, 0);
"""
    projection_writer_source = projection[2]
    if projection_writer_source.count("void kernel_main() {\n") != 1:
        raise RuntimeError("output writer main changed before ring init")
    projection_writer_source = projection_writer_source.replace(
        "void kernel_main() {\n", projection_writer_start, 1
    )
    projection_writer_release = """
    if (node == 0) {
        noc_semaphore_inc(
            projection_core_l1(1, get_semaphore(15)), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait(output_phase_done, 1);
        noc_semaphore_set(output_phase_done, 2);
        for (uint32_t target = 1; target < kProjectionWorkers; ++target) {
            noc_async_write(
                get_semaphore(15),
                projection_core_l1(target, get_semaphore(15)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    } else {
        noc_semaphore_wait(output_phase_done, 1);
        const uint32_t next =
            node + 1 == kProjectionWorkers ? 0 : node + 1;
        noc_semaphore_inc(
            projection_core_l1(next, get_semaphore(15)), 1);
        noc_async_atomic_barrier();
    }
    noc_semaphore_wait(output_phase_done, 2);
"""
    projection = (
        projection[0],
        projection[1],
        _append_before_kernel_end(
            projection_writer_source, projection_writer_release
        ),
    )
    qkv_reader = _resident_attention_dy_reader_source(
        qkv_reader,
        sequence_tiles=sequence_tiles,
        hidden_tiles=hidden_tiles,
        dimension_tiles=dimension_tiles,
        query_heads=query_heads,
        kv_heads=kv_heads,
    )
    # The standalone V reader clears its local resident-down semaphore map at
    # function entry.  That is safe enough behind a program launch boundary,
    # but not after the output-projection phase: the 50 reader RISCs arrive
    # with measurable skew, so a late local clear can erase a ready flag that
    # an early dH owner has already published.  The fused owner installs a
    # complete island-wide reset/barrier below; remove the now-racy duplicate.
    v_local_reset = """    // Semaphore addresses may contain state from an unrelated prior program.
    // Reset every flag on every invocation before any producer can publish.
    for (uint32_t semaphore = 0; semaphore < 2 * m_tiles + 1; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }

"""
    if qkv_reader.count(v_local_reset) != 1:
        raise RuntimeError("fused V reader local reset marker changed")
    qkv_reader = qkv_reader.replace(v_local_reset, "", 1)
    v_phase_local_reset = """    // These two IDs are outside resident-down's
    // 0..8 protocol. Clear stale state once, before the first phase.
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9)), 0);
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10)), 0);
"""
    if qkv_reader.count(v_phase_local_reset) != 1:
        raise RuntimeError("fused V writer-barrier reset marker changed")
    qkv_reader = qkv_reader.replace(v_phase_local_reset, "", 1)
    qkv_reader_main = "void kernel_main() {\n"
    qkv_reader_gate = f"""void kernel_main() {{
    volatile tt_l1_ptr uint32_t* output_phase_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
    noc_semaphore_wait_min(output_phase_done, 2);

    // The output writers have already completed a 50-core producer-side ring
    // and left semaphore 15 at state two on every projection core.  Node zero
    // can therefore publish the next-phase gate without polling remote state
    // or racing a writer that still owns semaphore 14.
    constexpr uint8_t entry_physical_x[11] = {{
        1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13}};
    constexpr uint32_t entry_projection_workers = 50;
    constexpr uint32_t entry_attention_workers = {
        query_heads * sequence_tiles
    };
    const uint32_t entry_node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();
    constexpr uint32_t entry_start_semaphore = 14;
    volatile tt_l1_ptr uint32_t* entry_start =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(entry_start_semaphore));
    if (entry_node == 0) {{
        noc_semaphore_set(entry_start, 1);
        for (uint32_t target = 1; target < entry_projection_workers; ++target) {{
            noc_async_write(
                get_semaphore(entry_start_semaphore),
                get_noc_addr(
                    entry_physical_x[target % 5], 2 + target / 5,
                    get_semaphore(entry_start_semaphore)),
                sizeof(uint32_t));
        }}
        // Attention semaphore 14 belonged to the completed dO producer-go
        // protocol and is now the Q/K/V publication gate.
        for (uint32_t target = 0; target < entry_attention_workers; ++target) {{
            const uint32_t logical_x = 5 + target % 6;
            noc_async_write(
                get_semaphore(entry_start_semaphore),
                get_noc_addr(
                    entry_physical_x[logical_x], 2 + target / 6,
                    get_semaphore(14)),
                sizeof(uint32_t));
        }}
        noc_async_write_barrier();
    }}
    noc_semaphore_wait(entry_start, 1);
    noc_semaphore_set(entry_start, 0);
"""
    if qkv_reader.count(qkv_reader_main) != 1:
        raise RuntimeError("QKV reader main changed before output-phase gate")
    qkv_reader = qkv_reader.replace(qkv_reader_main, qkv_reader_gate, 1)

    # A program boundary used to provide a clean semaphore epoch.  The full
    # owner needs that guarantee explicitly for every V/K/Q phase.  Keep the
    # long-lived attention addresses in 11..13 and reset resident-down's
    # complete short-lived 0..10 protocol behind a two-stage 50-core barrier.
    phase_init = r'''
static __attribute__((noinline)) void qkv_projection_phase_init() {
    constexpr uint8_t phase_physical_x[5] = {1, 2, 3, 4, 5};
    constexpr uint32_t phase_workers = 50;
    constexpr uint32_t phase_ready_semaphore = 14;
    const uint32_t phase_node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();
    for (uint32_t semaphore = 0; semaphore < 11; ++semaphore) {
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore)), 0);
    }
    volatile tt_l1_ptr uint32_t* phase_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(phase_ready_semaphore));
    // Semaphore 14 was the output-to-QKV gate (state one).  Every reader
    // cleared its local word before entering here.  A one-producer ring now
    // proves that all 50 reader RISCs finished clearing the short-lived
    // protocol before any writer may consume the V phase (state two).
    if (phase_node == 0) {
        noc_semaphore_inc(
            get_noc_addr(
                phase_physical_x[1], 2,
                get_semaphore(phase_ready_semaphore)), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait_min(phase_ready, 1);
        noc_semaphore_set(phase_ready, 2);
        for (uint32_t target = 1; target < phase_workers; ++target) {
            noc_async_write(
                get_semaphore(phase_ready_semaphore),
                get_noc_addr(
                    phase_physical_x[target % 5], 2 + target / 5,
                    get_semaphore(phase_ready_semaphore)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    } else {
        noc_semaphore_wait_min(phase_ready, 1);
        const uint32_t next =
            phase_node + 1 == phase_workers ? 0 : phase_node + 1;
        noc_semaphore_inc(
            get_noc_addr(
                phase_physical_x[next % 5], 2 + next / 5,
                get_semaphore(phase_ready_semaphore)), 1);
        noc_async_atomic_barrier();
    }
    noc_semaphore_wait_min(phase_ready, 2);
}
'''
    for index, prefix in enumerate(("v", "k", "q")):
        marker = f"FORCE_INLINE void {prefix}_reader_phase() {{"
        if qkv_reader.count(marker) != 1:
            raise RuntimeError(f"QKV {prefix.upper()} reader marker changed")
        # Only V crosses from a different resident owner.  V and K writers
        # already clear 0..8 and release the next reader through their
        # compute->writer fence plus 50-core sem9/10 barrier.  Repeating a
        # reader-only barrier before K or Q adds a second epoch protocol that
        # can race the writer-owned release and strand the Q reader.
        replacement = marker
        if index == 0:
            replacement = (
                phase_init + "\n" + marker
                + "\n    qkv_projection_phase_init();"
            )
        qkv_reader = qkv_reader.replace(marker, replacement, 1)
    # Match the Q compute clone above: its two dY consumers start at fresh
    # physical FIFO epochs instead of the fourth shared CB4/CB9 epoch.
    q_reader_start = qkv_reader.index("FORCE_INLINE void q_reader_phase() {")
    q_reader_end = qkv_reader.index("void kernel_main()", q_reader_start)
    q_reader_body = qkv_reader[q_reader_start:q_reader_end]
    remapped_q_reader_body = q_reader_body.replace(
        "constexpr uint32_t cb_base_dy = get_compile_time_arg_val(0);",
        "constexpr uint32_t cb_base_dy = 45;",
        1,
    ).replace(
        "constexpr uint32_t cb_base_weight = get_compile_time_arg_val(1);",
        "constexpr uint32_t cb_base_weight = 46;",
        1,
    ).replace(
        "constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(4);",
        "constexpr uint32_t cb_dh_dy = 32;",
        1,
    ).replace(
        "constexpr uint32_t cb_db_dy = get_compile_time_arg_val(9);",
        "constexpr uint32_t cb_db_dy = 33;",
        1,
    ).replace(
        "constexpr uint32_t cb_dh_b = get_compile_time_arg_val(5);",
        "constexpr uint32_t cb_dh_b = 36;",
        1,
    ).replace(
        "constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(10);",
        "constexpr uint32_t cb_db_hidden = 37;",
        1,
    ).replace(
        "constexpr uint32_t cb_dh_output = get_compile_time_arg_val(8);",
        "constexpr uint32_t cb_dh_output = 40;",
        1,
    ).replace(
        "constexpr uint32_t cb_dh_resident = get_compile_time_arg_val(16);",
        "constexpr uint32_t cb_dh_resident = 49;",
        1,
    ).replace(
        "constexpr uint32_t cb_tail_a = get_compile_time_arg_val(17);",
        "constexpr uint32_t cb_tail_a = 50;",
        1,
    ).replace(
        "constexpr uint32_t cb_tail_x = get_compile_time_arg_val(18);",
        "constexpr uint32_t cb_tail_x = 51;",
        1,
    )
    q_node_marker = (
        "    const uint32_t node = logical_y * q_kGridCols + logical_x;\n"
    )
    q_address_trace = q_node_marker + r'''    if (node == 0) {
        DPRINT_DATA0(
            "Q resident addresses q={} k={} v={}\n",
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(11)),
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(12)),
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(13)));
    }
'''
    if remapped_q_reader_body.count(q_node_marker) != 1:
        raise RuntimeError("Q reader node marker changed")
    remapped_q_reader_body = remapped_q_reader_body.replace(
        q_node_marker, q_address_trace, 1
    )
    q_first_read_marker = """            noc0.async_read_barrier();
            base_weight.push_back(base_reduction_tiles);
            base_dy.push_back(base_reduction_tiles);"""
    q_first_read_trace = """            noc0.async_read_barrier();
            if (node == 0 && local == 0 && block == 0) {
                const volatile tt_l1_ptr uint32_t* q_words =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        base_dy.get_write_ptr());
                DPRINT_DATA0(
                    "Q first dY words {} {} {} {}\\n",
                    q_words[0], q_words[1], q_words[2], q_words[3]);
            }
            base_weight.push_back(base_reduction_tiles);
            base_dy.push_back(base_reduction_tiles);"""
    if remapped_q_reader_body.count(q_first_read_marker) != 1:
        raise RuntimeError("Q reader first dY publication marker changed")
    remapped_q_reader_body = remapped_q_reader_body.replace(
        q_first_read_marker, q_first_read_trace, 1
    )
    if remapped_q_reader_body == q_reader_body:
        raise RuntimeError("Q reader input-FIFO remap did not change")
    qkv_reader = (
        qkv_reader[:q_reader_start] + remapped_q_reader_body
        + qkv_reader[q_reader_end:]
    )
    # Full-call reader order is output inputs 0..5, followed by the thirteen
    # surviving QKV operands at global custom-call indices 12..24.
    surviving = (0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15)
    tensor_map = {
        original: (6 + position, 12 + position)
        for position, original in enumerate(surviving)
    }
    qkv_reader = _remap_qkv_reader_abi(
        qkv_reader,
        tensor_map=tensor_map,
        total_cbs=_FULL_QKV_TOTAL_CBS,
    )
    qkv_writer = _resident_attention_qkv_writer_source(
        qkv_writer,
        gradient_row_offset=1 + hidden_tiles,
    )
    qkv_writer = _remap_qkv_writer_abi(
        qkv_writer,
        output_tensor_index=output_tensor_index,
        total_cbs=_FULL_QKV_TOTAL_CBS,
    )

    # The reusable serial projection barrier originally released only the
    # reader RISC.  A non-root writer could therefore enter the next phase,
    # increment sem9 for that next epoch, and have the old phase's root erase
    # the credit when it reset sem9.  This is observable on shapes where idle
    # K workers retire much earlier than active workers.  Use sem14 as an
    # independent writer-go word; sem10 remains exclusively reader-go.
    def append_writer_epoch(source: str, function_name: str) -> str:
        start = source.index(f"FORCE_INLINE void {function_name}() {{")
        brace = source.index("{", start)
        depth = 0
        end = -1
        for position in range(brace, len(source)):
            depth += (source[position] == "{") - (source[position] == "}")
            if depth == 0:
                end = position
                break
        if end < 0:
            raise RuntimeError(f"unterminated {function_name}")
        epoch = """
    volatile tt_l1_ptr uint32_t* writer_phase_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
    if (node == 0) {
        noc_semaphore_set(writer_phase_go, 1);
        for (uint32_t target = 1; target < 50; ++target) {
            noc_async_write(
                get_semaphore(14),
                get_noc_addr(
                    1 + target % 5, 2 + target / 5,
                    get_semaphore(14)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(writer_phase_go, 1);
    noc_semaphore_set(writer_phase_go, 0);
"""
        return source[:end] + epoch + source[end:]

    v_flag_start = qkv_writer.index("FORCE_INLINE void v_writer_phase() {")
    v_flag_brace = qkv_writer.index("{", v_flag_start)
    depth = 0
    v_flag_end = -1
    for position in range(v_flag_brace, len(qkv_writer)):
        depth += (qkv_writer[position] == "{") - (
            qkv_writer[position] == "}"
        )
        if depth == 0:
            v_flag_end = position + 1
            break
    if v_flag_end < 0:
        raise RuntimeError("unterminated V writer flag function")
    v_flag_body = qkv_writer[v_flag_start:v_flag_end]
    v_protocol_start_marker = (
        "    constexpr uint32_t phase_count_semaphore = 9;"
    )
    if v_flag_body.count(v_protocol_start_marker) != 1:
        raise RuntimeError("V writer count protocol changed")
    v_protocol_start = v_flag_body.index(v_protocol_start_marker)
    v_protocol_end = v_flag_body.rfind("}")
    v_flag_protocol = r'''    constexpr uint32_t done_semaphore = 15;
    constexpr uint32_t reader_go_semaphore = 10;
    volatile tt_l1_ptr uint32_t* local_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(done_semaphore));
    // Reader-side phase initialization publishes semaphore 14 state two only
    // after every reader has reset 0..10.  Semaphore 15 remains at output
    // completion state two, so V advances it monotonically to token/release
    // states three/four without a cross-RISC clear race.
    volatile tt_l1_ptr uint32_t* readers_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
    noc_semaphore_wait_min(readers_ready, 2);
    if (node == 0) {
        DPRINT_DATA1("V ring root sending token\n");
        noc_semaphore_inc(
            get_noc_addr(2, 2, get_semaphore(done_semaphore)), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait_min(local_done, 3);
        DPRINT_DATA1("V writer completion token returned\n");
    } else {
        DPRINT_DATA1("V ring waiting node={}\n", node);
        noc_semaphore_wait_min(local_done, 3);
        DPRINT_DATA1("V ring received node={}\n", node);
        const uint32_t next = node + 1 == 50 ? 0 : node + 1;
        noc_semaphore_inc(
            get_noc_addr(
                1 + next % 5, 2 + next / 5,
                get_semaphore(done_semaphore)),
            1);
        noc_async_atomic_barrier();
        DPRINT_DATA1("V ring forwarded node={}\n", node);
    }
    if (node == 0) {
        noc_semaphore_set(local_done, 4);
        for (uint32_t target = 1; target < 50; ++target) {
            noc_async_write(
                get_semaphore(done_semaphore),
                get_noc_addr(
                    1 + target % 5, 2 + target / 5,
                    get_semaphore(done_semaphore)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait_min(local_done, 4);
    // Every writer has now retired.  Release the co-located reader locally;
    // this avoids a root-to-49 fan-out whose remote stores were observed to
    // disappear under full-program traffic even after the NoC barrier.
    volatile tt_l1_ptr uint32_t* reader_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(reader_go_semaphore));
    noc_semaphore_set(reader_go, 1);
'''
    v_flag_body = (
        v_flag_body[:v_protocol_start] + v_flag_protocol
        + v_flag_body[v_protocol_end:]
    )
    qkv_writer = (
        qkv_writer[:v_flag_start] + v_flag_body
        + qkv_writer[v_flag_end:]
    )

    # K uses per-core completion flags instead of a many-to-one atomic count.
    # Silicon traces proved all 50 atomic increments retired at their senders
    # while node zero still observed fewer than 50.  Semaphore 15 is dead
    # after the entry rendezvous and is not used by the K reader, so each K
    # writer can own its local word: 0=running, 1=done, 2=released.  Node zero
    # polls each flag and publishes both reader-go and writer-release only
    # after all 50 flags are visibly one.
    k_count_start = qkv_writer.index("FORCE_INLINE void k_writer_phase() {")
    k_count_brace = qkv_writer.index("{", k_count_start)
    depth = 0
    k_count_end = -1
    for position in range(k_count_brace, len(qkv_writer)):
        depth += (qkv_writer[position] == "{") - (
            qkv_writer[position] == "}"
        )
        if depth == 0:
            k_count_end = position + 1
            break
    if k_count_end < 0:
        raise RuntimeError("unterminated K writer count function")
    k_count_body = qkv_writer[k_count_start:k_count_end]
    k_protocol_start_marker = (
        "    constexpr uint32_t phase_count_semaphore = 9;"
    )
    k_protocol_end_marker = (
        "    noc_semaphore_set(writer_phase_go, 0);\n"
    )
    if k_count_body.count(k_protocol_start_marker) != 1:
        raise RuntimeError("K writer count protocol changed")
    if k_count_body.count(k_protocol_end_marker) != 0:
        raise RuntimeError("K writer unexpectedly retained writer-go epoch")
    k_protocol_start = k_count_body.index(k_protocol_start_marker)
    # The original count barrier ends immediately before the function brace.
    # K has no appended writer epoch; replace the remainder of the function
    # body while retaining its final brace.
    k_protocol_end = k_count_body.rfind("}")
    k_flag_protocol = r'''    constexpr uint32_t done_semaphore = 15;
    constexpr uint32_t reader_go_semaphore = 10;
    volatile tt_l1_ptr uint32_t* local_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(done_semaphore));
    // V left semaphore 15 at release state four.  K advances the same epoch
    // monotonically to token/release states five/six.
    if (node == 0) {
        DPRINT_DATA1("K ring root sending token\n");
        noc_semaphore_inc(
            get_noc_addr(2, 2, get_semaphore(done_semaphore)), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait_min(local_done, 5);
        DPRINT_DATA1("K writer completion token returned\n");
    } else {
        DPRINT_DATA1("K ring waiting node={}\n", node);
        noc_semaphore_wait_min(local_done, 5);
        DPRINT_DATA1("K ring received node={}\n", node);
        const uint32_t next = node + 1 == 50 ? 0 : node + 1;
        noc_semaphore_inc(
            get_noc_addr(
                1 + next % 5, 2 + next / 5,
                get_semaphore(done_semaphore)),
            1);
        noc_async_atomic_barrier();
        DPRINT_DATA1("K ring forwarded node={}\n", node);
    }
    if (node == 0) {
        noc_semaphore_set(local_done, 6);
        for (uint32_t target = 1; target < 50; ++target) {
            noc_async_write(
                get_semaphore(done_semaphore),
                get_noc_addr(
                    1 + target % 5, 2 + target / 5,
                    get_semaphore(done_semaphore)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait_min(local_done, 6);
    volatile tt_l1_ptr uint32_t* reader_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(reader_go_semaphore));
    noc_semaphore_set(reader_go, 1);
    noc_semaphore_set(local_done, 0);
'''
    k_count_body = (
        k_count_body[:k_protocol_start] + k_flag_protocol
        + k_count_body[k_protocol_end:]
    )
    qkv_writer = (
        qkv_writer[:k_count_start] + k_count_body
        + qkv_writer[k_count_end:]
    )

    # These compile to no-ops unless TT_METAL_DPRINT_CORES is enabled.  They
    # make a silicon deadlock attributable to a concrete V/K/Q boundary
    # without changing the numerical schedule or exposing an intermediate.
    qkv_reader = '#include "api/debug/dprint.h"\n' + qkv_reader
    qkv_writer = '#include "api/debug/dprint.h"\n' + qkv_writer

    v_reader_start = qkv_reader.index("FORCE_INLINE void v_reader_phase() {")
    v_reader_brace = qkv_reader.index("{", v_reader_start)
    depth = 0
    v_reader_end = -1
    for position in range(v_reader_brace, len(qkv_reader)):
        depth += (qkv_reader[position] == "{") - (qkv_reader[position] == "}")
        if depth == 0:
            v_reader_end = position + 1
            break
    if v_reader_end < 0:
        raise RuntimeError("unterminated V reader function")
    v_reader_body = qkv_reader[v_reader_start:v_reader_end]
    v_reader_body = v_reader_body.replace(
        "        dh_output.wait_front(1);",
        '        DPRINT_DATA0("V owner waiting dH output node={}\\n", node);\n'
        "        dh_output.wait_front(1);\n"
        '        DPRINT_DATA0("V owner received dH output node={}\\n", node);',
        1,
    ).replace(
        "    if (node == 0) {\n        DeviceZoneScopedN(\"resident_down_dh_barrier\");",
        "    if (node == 0) {\n"
        '        DPRINT_DATA0("V root entering dH owner barrier\\n");\n'
        "        DeviceZoneScopedN(\"resident_down_dh_barrier\");",
        1,
    ).replace(
        "        for (uint32_t owner = 1; owner < m_tiles; ++owner) {",
        '        DPRINT_DATA0("V root passed dH owner barrier\\n");\n'
        "        for (uint32_t owner = 1; owner < m_tiles; ++owner) {",
        1,
    ).replace(
        "    dh_resident.push_back(m_tiles);",
        "    dh_resident.push_back(m_tiles);\n"
        '    DPRINT_DATA0("V reader completed dH multicast node={}\\n", node);',
        1,
    )
    qkv_reader = (
        qkv_reader[:v_reader_start] + v_reader_body
        + qkv_reader[v_reader_end:]
    )

    k_reader_start = qkv_reader.index("FORCE_INLINE void k_reader_phase() {")
    k_reader_brace = qkv_reader.index("{", k_reader_start)
    depth = 0
    k_reader_end = -1
    for position in range(k_reader_brace, len(qkv_reader)):
        depth += (qkv_reader[position] == "{") - (qkv_reader[position] == "}")
        if depth == 0:
            k_reader_end = position + 1
            break
    if k_reader_end < 0:
        raise RuntimeError("unterminated K reader function")
    k_reader_body = qkv_reader[k_reader_start:k_reader_end]
    k_reader_body = k_reader_body.replace(
        "        dh_output.wait_front(1);",
        '        DPRINT_DATA0("K owner waiting dH output\\n");\n'
        "        dh_output.wait_front(1);\n"
        '        DPRINT_DATA0("K owner received dH output\\n");',
        1,
    ).replace(
        "    if (node == 0) {\n        DeviceZoneScopedN(\"resident_down_dh_barrier\");",
        "    if (node == 0) {\n"
        '        DPRINT_DATA0("K root entering dH owner barrier\\n");\n'
        "        DeviceZoneScopedN(\"resident_down_dh_barrier\");",
        1,
    ).replace(
        "        for (uint32_t owner = 1; owner < m_tiles; ++owner) {",
        '        DPRINT_DATA0("K root passed dH owner barrier\\n");\n'
        "        for (uint32_t owner = 1; owner < m_tiles; ++owner) {",
        1,
    ).replace(
        "    dh_resident.push_back(m_tiles);",
        "    dh_resident.push_back(m_tiles);\n"
        '    DPRINT_DATA0("K reader completed dH multicast\\n");',
        1,
    )
    qkv_reader = (
        qkv_reader[:k_reader_start] + k_reader_body
        + qkv_reader[k_reader_end:]
    )

    q_reader_start = qkv_reader.index("FORCE_INLINE void q_reader_phase() {")
    q_reader_brace = qkv_reader.index("{", q_reader_start)
    depth = 0
    q_reader_end = -1
    for position in range(q_reader_brace, len(qkv_reader)):
        depth += (qkv_reader[position] == "{") - (qkv_reader[position] == "}")
        if depth == 0:
            q_reader_end = position + 1
            break
    if q_reader_end < 0:
        raise RuntimeError("unterminated Q reader function")
    q_reader_body = qkv_reader[q_reader_start:q_reader_end]
    q_reader_body = q_reader_body.replace(
        "    // Frozen-base dProduct input stream.",
        '    DPRINT_DATA0("Q reader base stream start node={}\\n", node);\n'
        "    // Frozen-base dProduct input stream.",
        1,
    ).replace(
        "    // Exact dH owner stream.",
        '    DPRINT_DATA0("Q reader base stream done node={}\\n", node);\n'
        "    // Exact dH owner stream.",
        1,
    ).replace(
        "    // Exact dB streams.",
        '    DPRINT_DATA0("Q reader dH inputs done node={}\\n", node);\n'
        "    // Exact dB streams.",
        1,
    ).replace(
        "    // Wait for every dH producer, then multicast its tile to the same fixed",
        '    DPRINT_DATA0("Q reader dB inputs done node={}\\n", node);\n'
        "    // Wait for every dH producer, then multicast its tile to the same fixed",
        1,
    ).replace(
        "        dh_output.wait_front(1);",
        '        DPRINT_DATA0("Q owner waiting dH output node={}\\n", node);\n'
        "        dh_output.wait_front(1);\n"
        '        DPRINT_DATA0("Q owner received dH output node={}\\n", node);',
        1,
    ).replace(
        "    dh_resident.push_back(m_tiles);",
        "    dh_resident.push_back(m_tiles);\n"
        '    DPRINT_DATA0("Q reader completed dH multicast node={}\\n", node);',
        1,
    )
    qkv_reader = (
        qkv_reader[:q_reader_start] + q_reader_body
        + qkv_reader[q_reader_end:]
    )

    # V is the first projection phase and therefore the useful place to
    # distinguish a compute-output starvation from a writer-barrier failure.
    # Keep the probes inside its function so a 50-core DPRINT run identifies
    # the exact core that failed to retire.  They disappear entirely from
    # ordinary builds where device print is disabled.
    v_writer_start = qkv_writer.index("FORCE_INLINE void v_writer_phase() {")
    v_writer_brace = qkv_writer.index("{", v_writer_start)
    depth = 0
    v_writer_end = -1
    for position in range(v_writer_brace, len(qkv_writer)):
        depth += (qkv_writer[position] == "{") - (qkv_writer[position] == "}")
        if depth == 0:
            v_writer_end = position + 1
            break
    if v_writer_end < 0:
        raise RuntimeError("unterminated V writer function")
    v_writer_body = qkv_writer[v_writer_start:v_writer_end]
    v_writer_body = v_writer_body.replace(
        "    phase_done_cb.wait_front(1);",
        '    DPRINT_DATA1("V writer waiting compute fence\\n");\n'
        "    phase_done_cb.wait_front(1);\n"
        '    DPRINT_DATA1("V writer passed compute fence\\n");',
        1,
    ).replace(
        "    noc_async_atomic_barrier();\n    if (node == 0) {",
        "    noc_async_atomic_barrier();\n"
        '    DPRINT_DATA1("V writer incremented island count\\n");\n'
        "    if (node == 0) {",
        1,
    ).replace(
        "        noc_semaphore_wait_min(count, 50);",
        "        noc_semaphore_wait_min(count, 50);\n"
        '        DPRINT_DATA1("V writer root reached island count\\n");',
        1,
    )
    qkv_writer = (
        qkv_writer[:v_writer_start] + v_writer_body
        + qkv_writer[v_writer_end:]
    )
    k_writer_start = qkv_writer.index("FORCE_INLINE void k_writer_phase() {")
    k_writer_brace = qkv_writer.index("{", k_writer_start)
    depth = 0
    k_writer_end = -1
    for position in range(k_writer_brace, len(qkv_writer)):
        depth += (qkv_writer[position] == "{") - (
            qkv_writer[position] == "}"
        )
        if depth == 0:
            k_writer_end = position + 1
            break
    if k_writer_end < 0:
        raise RuntimeError("unterminated K writer function")
    k_writer_body = qkv_writer[k_writer_start:k_writer_end]
    k_writer_body = k_writer_body.replace(
        "    phase_done_cb.wait_front(1);",
        '    DPRINT_DATA1("K writer waiting compute fence\\n");\n'
        "    phase_done_cb.wait_front(1);\n"
        '    DPRINT_DATA1("K writer passed compute fence\\n");',
        1,
    ).replace(
        "    noc_async_atomic_barrier();\n    if (node == 0) {",
        "    noc_async_atomic_barrier();\n"
        '    DPRINT_DATA1("K writer incremented island count\\n");\n'
        "    if (node == 0) {",
        1,
    ).replace(
        "        noc_semaphore_wait_min(count, 50);",
        "        noc_semaphore_wait_min(count, 50);\n"
        '        DPRINT_DATA1("K writer root reached island count\\n");',
        1,
    )
    qkv_writer = (
        qkv_writer[:k_writer_start] + k_writer_body
        + qkv_writer[k_writer_end:]
    )
    for prefix, label in (("v", "V"), ("k", "K"), ("q", "Q")):
        reader_call = f"    {prefix}_reader_phase();"
        reader_marked = (
            f'    DPRINT_DATA0("full-attention reader {label} enter\\n");\n'
            f"{reader_call}\n"
            f'    DPRINT_DATA0("full-attention reader {label} done\\n");'
        )
        if qkv_reader.count(reader_call) != 1:
            raise RuntimeError(f"QKV {label} reader call marker changed")
        qkv_reader = qkv_reader.replace(reader_call, reader_marked, 1)
        writer_call = f"    {prefix}_writer_phase();"
        writer_marked = (
            f'    DPRINT_DATA1("full-attention writer {label} enter\\n");\n'
            f"{writer_call}\n"
            f'    DPRINT_DATA1("full-attention writer {label} done\\n");'
        )
        if qkv_writer.count(writer_call) != 1:
            raise RuntimeError(f"QKV {label} writer call marker changed")
        qkv_writer = qkv_writer.replace(writer_call, writer_marked, 1)

    # Keep the attention-side credit protocol observable while the full
    # one-call lifecycle is being qualified on silicon.  These calls compile
    # away unless TT_METAL_DPRINT_CORES selects an attention worker.
    attention_reader = '#include "api/debug/dprint.h"\n' + attention[0]
    attention_reader = attention_reader.replace(
        "    noc_semaphore_set(dout_credit, 0);",
        "    noc_semaphore_set(dout_credit, 0);\n"
        '    DPRINT_DATA0("attention dO credit cleared\\n");',
        1,
    )
    ready_wait = "    noc_semaphore_wait(ready_epoch_go, 1);"
    if attention_reader.count(ready_wait) != 2:
        raise RuntimeError("attention ready-epoch wait contract changed")
    attention_reader = attention_reader.replace(
        ready_wait, "/* DPRINT_READY_EPOCH_START */", 1
    ).replace(
        ready_wait, "/* DPRINT_READY_EPOCH_PUBLISH */", 1
    ).replace(
        "/* DPRINT_READY_EPOCH_START */",
        '    DPRINT_DATA0("attention ready epoch waiting start\\n");\n'
        "    noc_semaphore_wait(ready_epoch_go, 1);\n"
        '    DPRINT_DATA0("attention ready epoch got start\\n");',
    ).replace(
        "/* DPRINT_READY_EPOCH_PUBLISH */",
        '    DPRINT_DATA0("attention ready epoch waiting publish\\n");\n'
        "    noc_semaphore_wait(ready_epoch_go, 1);\n"
        '    DPRINT_DATA0("attention ready epoch got publish\\n");',
    ).replace(
        "    noc_semaphore_wait_min(dout_credit, vector_tiles);",
        '    DPRINT_DATA0("attention waiting dO credits={}\\n", '
        "*dout_credit);\n"
        "    noc_semaphore_wait_min(dout_credit, vector_tiles);\n"
        '    DPRINT_DATA0("attention received dO credits={}\\n", '
        "*dout_credit);",
        1,
    )
    attention_compute = '#include "api/debug/dprint.h"\n' + attention[1]
    for label in ("ADV", "ADS", "ADQ", "ADK", "AG0", "AG1", "AG2", "AGD"):
        marker = f'    WAYPOINT("{label}");'
        attention_compute = attention_compute.replace(
            marker,
            marker + f'\n    DPRINT_UNPACK("attention compute {label}\\n");',
        )
    attention_writer = '#include "api/debug/dprint.h"\n' + attention[2]
    for label in ("AW0", "AW1", "AW2", "AH0", "AH1", "AH2", "AH3", "AW4"):
        marker = f'        WAYPOINT("{label}");'
        if marker not in attention_writer:
            marker = f'    WAYPOINT("{label}");'
        attention_writer = attention_writer.replace(
            marker,
            marker + f'\n        DPRINT_DATA1("attention writer {label}\\n");',
        )
    attention = (attention_reader, attention_compute, attention_writer)

    # CB25..56 are added only on the projection island. They increase the
    # compile-time tensor-accessor offset while preserving every local CB ID.
    def widen(source: str) -> str:
        return source.replace(
            f", {_TOTAL_CBS}>()", f", {_FULL_QKV_TOTAL_CBS}>()"
        )

    projection = tuple(widen(source) for source in projection)
    attention = tuple(widen(source) for source in attention)
    sources = (
        _merge_phased_kernel_source(projection[0], qkv_reader),
        qkv_compute,
        _merge_phased_kernel_source(projection[2], qkv_writer),
        attention[0], attention[1], attention[2],
    )

    # Hardware diagnosis intentionally instruments the generated sources
    # above so structural substitutions remain guarded. Production execution
    # must not pay DPRINT traffic or retain its headers. Strip only complete
    # DPRINT statements after all source composition has succeeded.
    def production_source(source: str) -> str:
        source = source.replace('#include "api/debug/dprint.h"\n', "")
        return re.sub(
            r"^[ \t]*DPRINT_(?:PACK|UNPACK|DATA0|DATA1)"
            r"\([\s\S]*?\);[ \t]*\n",
            "",
            source,
            flags=re.MULTILINE,
        )

    return tuple(production_source(source) for source in sources)


def _function_extent(source: str, name: str) -> tuple[int, int]:
    """Return the exact source range occupied by one generated C++ function."""

    marker = f"void {name}() {{"
    start = source.find(marker)
    if start < 0:
        raise RuntimeError(f"generated function {name} is missing")
    brace = source.find("{", start)
    depth = 0
    for position in range(brace, len(source)):
        depth += (source[position] == "{") - (source[position] == "}")
        if depth == 0:
            return start, position + 1
    raise RuntimeError(f"generated function {name} is unterminated")


def _replace_once(source: str, old: str, new: str, contract: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{contract} changed: expected one source marker, found {count}"
        )
    return source.replace(old, new, 1)


def _rewrite_projection_checkpoint_function(
    source: str,
    *,
    function_name: str,
    prefix: str,
    hidden_global_index: int,
    hidden_common_index: int,
    hidden_page_base: int,
    x_from_head_arena: bool = False,
    x_global_index: int = 5,
    x_common_index: int = 5,
    head_page_base: int = 0,
    sequence_tiles: int = 0,
    dimension_tiles: int = 0,
) -> str:
    """Point one accepted projection reader epoch at the forward arena.

    The generated compute and writer kernels are deliberately unchanged.  This
    helper only substitutes the DRAM page source for H (and O's X) while
    retaining the exact FIFO publication order consumed by v90.
    """

    start, end = _function_extent(source, function_name)
    body = source[start:end]
    tile_bytes = f"{prefix}kTileBytes"
    body = _replace_once(
        body,
        "    const uint32_t hidden_addr = "
        f"get_common_arg_val<uint32_t>({hidden_common_index});",
        "    const uint32_t checkpoint_arena_addr = "
        "get_common_arg_val<uint32_t>(19);\n"
        "    constexpr uint32_t checkpoint_hidden_page_base = "
        f"{hidden_page_base};",
        f"{function_name} checkpoint address",
    )
    old_hidden_accessor = (
        "    const auto hidden = TensorAccessor(\n"
        "        TensorAccessorArgs<\n"
        "            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
        f"{hidden_global_index}, {_FULL_QKV_TOTAL_CBS}>(), "
        f"{hidden_common_index}>(), hidden_addr, {tile_bytes});"
    )
    new_hidden_accessor = (
        "    const auto hidden = TensorAccessor(\n"
        "        TensorAccessorArgs<\n"
        "            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
        f"25, {_FULL_QKV_TOTAL_CBS}>(), 19>(), "
        f"checkpoint_arena_addr, {tile_bytes});"
    )
    body = _replace_once(
        body, old_hidden_accessor, new_hidden_accessor,
        f"{function_name} hidden accessor",
    )
    hidden_page = f"{{.page_id = checkpoint_hidden_page_base + m}}"
    body = _replace_once(
        body,
        f"{tile_bytes}, {{.page_id = m}}, {{}});",
        f"{tile_bytes}, {hidden_page}, {{}});",
        f"{function_name} hidden checkpoint read",
    )

    if x_from_head_arena:
        if not sequence_tiles or not dimension_tiles:
            raise ValueError("head-arena mapping requires sequence/dimension tiles")
        body = _replace_once(
            body,
            "    const uint32_t x_addr = "
            f"get_common_arg_val<uint32_t>({x_common_index});",
            "    constexpr uint32_t checkpoint_head_page_base = "
            f"{head_page_base};\n"
            "    constexpr uint32_t checkpoint_sequence_tiles = "
            f"{sequence_tiles};\n"
            "    constexpr uint32_t checkpoint_dimension_tiles = "
            f"{dimension_tiles};",
            f"{function_name} head-output address",
        )
        old_x_accessor = (
            "    const auto x = TensorAccessor(\n"
            "        TensorAccessorArgs<\n"
            "            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
            f"{x_global_index}, {_FULL_QKV_TOTAL_CBS}>(), "
            f"{x_common_index}>(), x_addr, {tile_bytes});"
        )
        new_x_accessor = (
            "    const auto x = TensorAccessor(\n"
            "        TensorAccessorArgs<\n"
            "            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
            f"25, {_FULL_QKV_TOTAL_CBS}>(), 19>(), "
            f"checkpoint_arena_addr, {tile_bytes});"
        )
        body = _replace_once(
            body, old_x_accessor, new_x_accessor,
            f"{function_name} head-output accessor",
        )
        old_page = "{.page_id = m * k_tiles + k}"
        new_page = (
            "{.page_id = checkpoint_head_page_base +\n"
            "                    (k / checkpoint_dimension_tiles) *\n"
            "                        checkpoint_sequence_tiles *\n"
            "                        checkpoint_dimension_tiles +\n"
            "                    m * checkpoint_dimension_tiles +\n"
            "                    (k % checkpoint_dimension_tiles)}"
        )
        body = _replace_once(
            body, old_page, new_page,
            f"{function_name} head-major page mapping",
        )

    return source[:start] + body + source[end:]


def _forward_checkpoint_arena_sources(
    projection_sources: tuple[str, str, str],
    attention_sources: tuple[str, str, str],
    *,
    sequence_length: int,
    hidden_features: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """Make v90 consume the exact forward checkpoint arena directly.

    This is a pure reader-side ABI transformation.  It removes Q/K/V,
    head-output, rank-hidden and probability tensors from the physical data
    path without changing any Tensix arithmetic, reductions, packing or
    writer layout in the accepted backward owner.
    """

    from .full_attention_forward_planner import plan_full_attention_forward

    kv_features = kv_heads * head_dim
    plan = plan_full_attention_forward(
        tokens=sequence_length,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=32,
        qkv_bias=False,
    )
    pages = {
        name: plan.arena.slice(name).page_offset
        for name in (
            "query", "key", "value", "head_output",
            "q_rank_hidden", "k_rank_hidden", "v_rank_hidden",
            "o_rank_hidden", "probability_fp32",
        )
    }
    projection_reader = projection_sources[0]
    projection_reader = _rewrite_projection_checkpoint_function(
        projection_reader,
        function_name="v_reader_phase", prefix="v_",
        hidden_global_index=23, hidden_common_index=17,
        hidden_page_base=pages["v_rank_hidden"],
    )
    projection_reader = _rewrite_projection_checkpoint_function(
        projection_reader,
        function_name="k_reader_phase", prefix="k_",
        hidden_global_index=19, hidden_common_index=13,
        hidden_page_base=pages["k_rank_hidden"],
    )
    projection_reader = _rewrite_projection_checkpoint_function(
        projection_reader,
        function_name="q_reader_phase", prefix="q_",
        hidden_global_index=15, hidden_common_index=9,
        hidden_page_base=pages["q_rank_hidden"],
    )
    projection_reader = _rewrite_projection_checkpoint_function(
        projection_reader,
        function_name="kernel_main", prefix="",
        hidden_global_index=3, hidden_common_index=3,
        hidden_page_base=pages["o_rank_hidden"],
        x_from_head_arena=True,
        x_global_index=5, x_common_index=5,
        head_page_base=pages["head_output"],
        sequence_tiles=plan.sequence_tiles,
        dimension_tiles=head_dim // TILE_SIZE,
    )

    attention_reader = attention_sources[0]
    accessor_start = (
        "    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);"
    )
    accessor_end = (
        "        sin_addr, kTileBytes);"
    )
    start = attention_reader.find(accessor_start)
    end = attention_reader.find(accessor_end, start)
    if start < 0 or end < 0:
        raise RuntimeError("attention checkpoint accessor contract changed")
    end += len(accessor_end)
    direct_accessors = f"""    const uint32_t cos_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sin_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t checkpoint_arena_addr = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t checkpoint_query_page_base = {pages['query']};
    constexpr uint32_t checkpoint_key_page_base = {pages['key']};
    constexpr uint32_t checkpoint_value_page_base = {pages['value']};
    constexpr uint32_t checkpoint_probability_page_base = {pages['probability_fp32']};
    const auto cos_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<10, {_FULL_QKV_TOTAL_CBS}>(), 0>(),
        cos_addr, kTileBytes);
    const auto sin_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<11, {_FULL_QKV_TOTAL_CBS}>(), 1>(),
        sin_addr, kTileBytes);
    const auto checkpoint_arena_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<25, {_FULL_QKV_TOTAL_CBS}>(), 2>(),
        checkpoint_arena_addr, kTileBytes);"""
    attention_reader = (
        attention_reader[:start] + direct_accessors + attention_reader[end:]
    )

    def raw_probability_read(destination: str, logical_page: str) -> str:
        return f"""        const uint32_t {destination}_logical_page = {logical_page};
        const uint32_t {destination}_physical_page =
            checkpoint_probability_page_base + 2 * {destination}_logical_page;
        noc0.async_read(
            checkpoint_arena_tensor,
            CoreLocalMem<uint32_t>({destination}.get_write_ptr() + tile * kProbabilityTileBytes),
            kTileBytes, {{.page_id = {destination}_physical_page}}, {{}});
        noc0.async_read(
            checkpoint_arena_tensor,
            CoreLocalMem<uint32_t>({destination}.get_write_ptr() + tile * kProbabilityTileBytes + kTileBytes),
            kTileBytes, {{.page_id = {destination}_physical_page + 1}}, {{}});"""

    old_p_row = """        noc0.async_read(
            p_tensor,
            CoreLocalMem<uint32_t>(p_row.get_write_ptr() + tile * kProbabilityTileBytes),
            kProbabilityTileBytes,
            {.page_id = head * score_tiles + row * seq_tiles + tile}, {});"""
    old_p_column = """        noc0.async_read(
            p_tensor,
            CoreLocalMem<uint32_t>(p_column.get_write_ptr() + tile * kProbabilityTileBytes),
            kProbabilityTileBytes,
            {.page_id = head * score_tiles + tile * seq_tiles + row}, {});"""
    attention_reader = _replace_once(
        attention_reader, old_p_row,
        raw_probability_read(
            "p_row", "head * score_tiles + row * seq_tiles + tile"
        ),
        "attention probability-row arena read",
    )
    attention_reader = _replace_once(
        attention_reader, old_p_column,
        raw_probability_read(
            "p_column", "head * score_tiles + tile * seq_tiles + row"
        ),
        "attention probability-column arena read",
    )
    attention_reader = _replace_once(
        attention_reader,
        "            q_tensor,\n"
        "            CoreLocalMem<uint32_t>(q_full.get_write_ptr() + tile * kTileBytes),\n"
        "            kTileBytes, {.page_id = head * vector_tiles + tile}, {});",
        "            checkpoint_arena_tensor,\n"
        "            CoreLocalMem<uint32_t>(q_full.get_write_ptr() + tile * kTileBytes),\n"
        "            kTileBytes, {.page_id = checkpoint_query_page_base +\n"
        "                head * vector_tiles + tile}, {});",
        "attention Q arena read",
    )
    attention_reader = _replace_once(
        attention_reader,
        "            k_tensor,\n"
        "            CoreLocalMem<uint32_t>(k.get_write_ptr() + tile * kTileBytes),\n"
        "            kTileBytes, {.page_id = kv_head * vector_tiles + tile}, {});",
        "            checkpoint_arena_tensor,\n"
        "            CoreLocalMem<uint32_t>(k.get_write_ptr() + tile * kTileBytes),\n"
        "            kTileBytes, {.page_id = checkpoint_key_page_base +\n"
        "                kv_head * vector_tiles + tile}, {});",
        "attention K arena read",
    )
    attention_reader = _replace_once(
        attention_reader,
        "                v_tensor,\n"
        "                CoreLocalMem<uint32_t>(v.get_write_ptr() + target * kTileBytes),\n"
        "                kTileBytes, {.page_id = kv_head * vector_tiles + source}, {});",
        "                checkpoint_arena_tensor,\n"
        "                CoreLocalMem<uint32_t>(v.get_write_ptr() + target * kTileBytes),\n"
        "                kTileBytes, {.page_id = checkpoint_value_page_base +\n"
        "                    kv_head * vector_tiles + source}, {});",
        "attention V arena read",
    )
    if any(
        old in attention_reader
        for old in ("p_tensor", "q_tensor", "k_tensor", "v_tensor")
    ):
        raise RuntimeError("legacy checkpoint accessors remain in attention reader")

    return (
        (projection_reader, projection_sources[1], projection_sources[2]),
        (attention_reader, attention_sources[1], attention_sources[2]),
    )


def _direct_checkpoint_arena_abi_sources(
    projection_sources: tuple[str, str, str],
    attention_sources: tuple[str, str, str],
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """Compact the direct-arena custom call to only live tensor operands."""

    def remap(
        sources: tuple[str, str, str],
        mappings: tuple[tuple[int, int, int, int], ...],
    ) -> tuple[str, str, str]:
        common_mapping: dict[int, int] = {}
        for _, old_common, _, new_common in mappings:
            previous = common_mapping.setdefault(old_common, new_common)
            if previous != new_common:
                raise RuntimeError(
                    "direct checkpoint ABI gives common argument "
                    f"{old_common} conflicting destinations "
                    f"{previous}/{new_common}"
                )
        for old_global, old_common, new_global, new_common in mappings:
            pattern = re.compile(
                r"get_tensor_accessor_args_cta_offset<\s*"
                rf"{old_global}\s*,\s*{_FULL_QKV_TOTAL_CBS}\s*>\(\)"
                rf"\s*,\s*{old_common}\b"
            )
            count = sum(len(pattern.findall(source)) for source in sources)
            if count < 1:
                raise RuntimeError(
                    "direct checkpoint ABI accessor changed for "
                    f"tensor {old_global}/common {old_common}"
                )
        rewritten = []
        for source in sources:
            for old_global, old_common, new_global, new_common in mappings:
                pattern = re.compile(
                    r"get_tensor_accessor_args_cta_offset<\s*"
                    rf"{old_global}\s*,\s*{_FULL_QKV_TOTAL_CBS}\s*>\(\)"
                    rf"\s*,\s*{old_common}\b"
                )
                source = pattern.sub(
                    "get_tensor_accessor_args_cta_offset<"
                    f"{new_global}, {_FULL_QKV_TOTAL_CBS}>(), {new_common}",
                    source,
                )
            # TensorAccessorArgs carries the common-argument ordinal in its
            # type, but the generated reader also loads each address into a
            # local variable first.  Both sites must move together.  A single
            # callback pass avoids cascaded substitutions such as 19->14 and
            # then treating that new 14 as an original argument.
            source = re.sub(
                r"get_common_arg_val<uint32_t>\(\s*(\d+)\s*\)",
                lambda match: (
                    "get_common_arg_val<uint32_t>("
                    f"{common_mapping.get(int(match.group(1)), int(match.group(1)))}"
                    ")"
                ),
                source,
            )
            rewritten.append(source)
        remaining = []
        for source in rewritten:
            remaining.extend(
                int(index)
                for index in re.findall(
                    r"get_tensor_accessor_args_cta_offset<\s*(\d+)\s*,\s*"
                    rf"{_FULL_QKV_TOTAL_CBS}\s*>",
                    source,
                )
                if int(index) > 17
            )
        if remaining:
            raise RuntimeError(
                "direct checkpoint ABI left out-of-range tensor accessors: "
                f"{sorted(set(remaining))}"
            )
        return tuple(rewritten)  # type: ignore[return-value]

    projection = remap(
        projection_sources,
        (
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (2, 2, 2, 2),
            (4, 4, 3, 3),
            (12, 6, 6, 4),
            (13, 7, 7, 5),
            (14, 8, 8, 6),
            (16, 10, 9, 7),
            (17, 11, 10, 8),
            (18, 12, 11, 9),
            (20, 14, 12, 10),
            (21, 15, 13, 11),
            (22, 16, 14, 12),
            (24, 18, 15, 13),
            (25, 19, 16, 14),
            (26, 0, 17, 0),
        ),
    )
    attention = remap(
        attention_sources,
        (
            (10, 0, 4, 0),
            (11, 1, 5, 1),
            (25, 2, 16, 2),
            (26, 0, 17, 0),
        ),
    )
    return projection, attention


@lru_cache(maxsize=8)
def build_tt_lang_output_attention_resident_backward(
    *, sequence_length: int, hidden_features: int, query_heads: int,
    kv_heads: int, head_dim: int, physical_rank: int,
    lora_scale: float, attention_scale: float,
    diagnostic_write_dout: bool = False,
    attention_diagnostic: str | None = None,
    resident_qkv_handoff: bool = False,
    resident_qkv_projection: bool = False,
    forward_checkpoint_arena: bool = False,
    artifact_program_sequence: bool = False,
) -> Callable:
    error = output_attention_support_error(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
    )
    if error is not None:
        raise ValueError(error)
    if attention_diagnostic not in (None, "dv", "dout_row"):
        raise ValueError(
            "output+attention diagnostic must be None, 'dv', or 'dout_row'"
        )
    if resident_qkv_projection:
        resident_qkv_handoff = True
    if forward_checkpoint_arena and not resident_qkv_projection:
        raise ValueError(
            "forward checkpoint arena requires the full resident QKV owner"
        )
    if resident_qkv_handoff and attention_diagnostic is not None:
        raise ValueError("resident QKV handoff cannot run a core diagnostic")
    handoff = plan_output_attention_l1_handoff(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        query_heads=query_heads,
        head_dim=head_dim,
    )
    core_plan = plan_attention_block(
        batch_size=1,
        sequence_length=sequence_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    m_tiles = sequence_length // TILE_SIZE
    k_tiles = hidden_features // TILE_SIZE
    n_tiles = k_tiles
    dim_tiles = head_dim // TILE_SIZE
    seq_tiles = m_tiles
    vector_tiles = seq_tiles * dim_tiles
    group = query_heads // kv_heads
    projection_workers = 50
    jobs_per_node = -(-(m_tiles * k_tiles) // projection_workers)
    if resident_qkv_projection:
        jobs_per_node = _complete_projection_jobs_per_node(
            total_jobs=m_tiles * k_tiles,
            projection_workers=projection_workers,
        )
    da_k_tiles_per_node = -(-k_tiles // projection_workers)
    n_tiles_per_node = -(-n_tiles // projection_workers)
    base_phase = exact_base_dx_k_phase(hidden_features, hidden_features)
    assert base_phase is not None

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("fused output+attention requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _program(
        output_dy, o_weight, o_b, o_hidden, o_a, head_output,
        probability, query, key, value, cosine, sine,
        hidden,
        q_weight, q_b, q_hidden, q_a,
        k_weight, k_b, k_hidden, k_a,
        v_weight, v_b, v_hidden, v_a,
        checkpoint_arena,
        arena,
    ):
        b0 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=24)
        b1 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b2 = ttl.make_dataflow_buffer_like(
            probability, shape=(1, 1), block_count=1
        )
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
        b14 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b15 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b16 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b17 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b18 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b19 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b20 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b21 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b22 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b23 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b24 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=9)

        @ttl.datamovement()
        def read():
            with b0.reserve() as block:
                tx = ttl.copy(output_dy[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(o_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(o_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(o_hidden[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(o_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(head_output[0:1, 0:1], block); tx.wait()
            with b2.reserve() as block:
                tx = ttl.copy(probability[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(query[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(key[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(value[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(cosine[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(sine[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(hidden[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_hidden[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_hidden[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_hidden[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(checkpoint_arena[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(25):
                with b0.wait() as item:
                    with b24.reserve() as result:
                        result.store(item)
            with b2.wait() as _probability_item:
                pass
            x, y = ttl.node(dims=2)
            if x + y < 0:
                with b1.wait() as _: pass
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
                with b14.wait() as _: pass
                with b15.wait() as _: pass
                with b16.wait() as _: pass
                with b17.wait() as _: pass
                with b18.wait() as _: pass
                with b19.wait() as _: pass
                with b20.wait() as _: pass
                with b21.wait() as _: pass
                with b22.wait() as _: pass
                with b23.wait() as _: pass

        @ttl.datamovement()
        def write():
            for _ in range(25):
                with b24.wait() as item:
                    tx0 = ttl.copy(item, arena[0:1, 0:1]); tx0.wait()

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _direct_program(
        output_dy, o_weight, o_b, o_a, cosine, sine, hidden,
        q_weight, q_b, q_a,
        k_weight, k_b, k_a,
        v_weight, v_b, v_a,
        checkpoint_arena,
        arena,
    ):
        """Compile the direct-arena owner with its real compact ABI.

        TT-Lang compiles this probe before ``kernel_paths`` are replaced by
        the six production kernels.  Keeping the old 26-input probe here
        caused PJRT to retain ten dead aliases of the large checkpoint arena
        and, after the outer ABI was compacted, made the probe writer address
        nonexistent tensor 26.  The probe still declares the same 25 CBs so
        the descriptor patch below remains identical, but every tensor index
        now belongs to the actual 17-input + one-output custom call.
        """
        b0 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=17)
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
        b14 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b15 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b16 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b17 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b18 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b19 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b20 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b21 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b22 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b23 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b24 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=17)

        @ttl.datamovement()
        def read():
            with b0.reserve() as block:
                tx = ttl.copy(output_dy[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(o_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(o_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(o_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(cosine[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(sine[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(hidden[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(checkpoint_arena[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(17):
                with b0.wait() as item:
                    with b24.reserve() as result:
                        result.store(item)
            # Preserve all production CB descriptors in the probe.  The
            # condition is impossible on the physical non-negative core grid,
            # so it never participates in the substituted runtime kernels.
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
                with b14.wait() as _: pass
                with b15.wait() as _: pass
                with b16.wait() as _: pass
                with b17.wait() as _: pass
                with b18.wait() as _: pass
                with b19.wait() as _: pass
                with b20.wait() as _: pass
                with b21.wait() as _: pass
                with b22.wait() as _: pass
                with b23.wait() as _: pass

        @ttl.datamovement()
        def write():
            for _ in range(17):
                with b24.wait() as item:
                    tx0 = ttl.copy(item, arena[0:1, 0:1]); tx0.wait()

    selected_program = _direct_program if forward_checkpoint_arena else _program

    scale_bf16 = float(
        torch.tensor(lora_scale, dtype=torch.bfloat16).float().item()
    )
    lora_scale_bits = struct.unpack("<I", struct.pack("<f", scale_bf16))[0]
    attention_scale_bits = struct.unpack(
        "<I", struct.pack("<f", float(attention_scale))
    )[0]
    projection_replacements = {
        "@M_TILES@": str(m_tiles),
        "@K_TILES@": str(k_tiles),
        "@N_TILES@": str(n_tiles),
        "@BASE_REDUCTION_TILES@": str(base_phase),
        "@JOBS_PER_NODE@": str(jobs_per_node),
        "@DA_K_TILES_PER_NODE@": str(da_k_tiles_per_node),
        "@N_TILES_PER_NODE@": str(n_tiles_per_node),
        "@TOTAL_JOBS@": str(m_tiles * k_tiles),
        "@REDUCTION_TILES_PER_PHASE@": str(base_phase),
        "@GRID_COLS@": "5",
        "@NUM_N_BLOCKS@": str(n_tiles // 2),
        "@M_TILES_PER_NODE@": "1",
        "@TOTAL_M_TILES@": str(m_tiles),
        "@TOTAL_N_TILES@": str(n_tiles),
        "@REDUCTION_TILES@": str(m_tiles),
        "@SCALE_BITS@": str(lora_scale_bits),
        "@DIM_TILES@": str(dim_tiles),
        "@SEQ_TILES@": str(seq_tiles),
        "@QUERY_HEADS@": str(query_heads),
        "@KV_HEADS@": str(kv_heads),
    }
    attention_replacements = {
        "@SEQ_TILES@": str(seq_tiles),
        "@DIM_TILES@": str(dim_tiles),
        "@QUERY_HEADS@": str(query_heads),
        "@KV_HEADS@": str(kv_heads),
        "@SCALE_BITS@": str(attention_scale_bits),
        "@DIAGNOSTIC_DS@": "0",
        "@DIAGNOSTIC_DV@": str(
            int(attention_diagnostic in ("dv", "dout_row"))
        ),
        "@DIAGNOSTIC_DOUT_HANDOFF@": str(
            int(attention_diagnostic == "dout_row")
        ),
        "@DIAGNOSTIC_PT@": "0",
        "@DIAGNOSTIC_DQ@": "0",
        "@DIAGNOSTIC_DK@": "0",
        "@DIAGNOSTIC_DP@": "0",
        "@DIAGNOSTIC_PRODUCT@": "0",
        "@DIAGNOSTIC_CENTERED@": "0",
        "@DIAGNOSTIC_ROWSUM@": "0",
        "@DIAGNOSTIC_UNSCALED_DS@": "0",
        "@P_INPUT_BF16@": "0",
        "@TILE_BYTES@": "2048",
        "@P_TILE_BYTES@": "4096",
        "@ELEMENT_BYTES@": "2",
        "@BF16@": "1",
        "@FP32_DEST@": "1",
        "@MATMUL_PHASE_TILES@": "1",
        "@TOTAL_CBS@": str(_TOTAL_CBS),
    }
    variant = hashlib.sha256(
        (
            f"output-attention-v90-complete-worker-epochs:{sequence_length}:"
            f"{hidden_features}:"
            f"{query_heads}:{kv_heads}:{head_dim}:{physical_rank}:"
            f"{scale_bf16:.17g}:{attention_scale:.17g}:"
            f"{int(diagnostic_write_dout)}:{attention_diagnostic or 'full'}:"
            f"qkv-handoff={int(resident_qkv_handoff)}:"
            f"qkv-projection={int(resident_qkv_projection)}"
            f":forward-arena={int(forward_checkpoint_arena)}"
        ).encode("ascii")
    ).hexdigest()[:16]

    def _invoke_operation(program_args):
        _clear_ttl_compile_wrapper_cache(selected_program)
        result = selected_program(*program_args)
        closure = selected_program.__closure__ or ()
        for name, cell in zip(selected_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != _PROJECTION_CBS:
                    raise RuntimeError(
                        "output+attention probe expected 25 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                template = compiled.cb_configs[24]
                projection_shapes = _projection_cb_shapes(
                    m_tiles=m_tiles,
                    n_tiles=n_tiles,
                    base_phase=base_phase,
                    jobs_per_node=jobs_per_node,
                    da_k_tiles_per_node=da_k_tiles_per_node,
                    n_tiles_per_node=n_tiles_per_node,
                )
                for index, (shape, blocks) in projection_shapes.items():
                    compiled.cb_configs[index].shape = shape
                    compiled.cb_configs[index].block_count = blocks
                for index in (2, 7, 13, 20):
                    compiled.cb_configs[index].tensor = _FP32_DTYPE_ONLY_TENSOR

                attention_blocks = _attention_cb_blocks(
                    seq_tiles=seq_tiles,
                    dim_tiles=dim_tiles,
                    vector_tiles=vector_tiles,
                    group=group,
                )
                local_id_map = {2: 25, 3: 26, 25: 2, 26: 3}
                for original_id, blocks in enumerate(attention_blocks):
                    config = copy(template)
                    config.shape = (1, 1)
                    config.block_count = max(1, blocks)
                    config._cb_index = local_id_map.get(original_id, original_id)
                    compiled.cb_configs.append(config)
                from ttl.dataflow_buffer import CompilerAllocatedDFBConfig

                for original_id in (0, 1, 12, 13, 14, 15, 18):
                    config = CompilerAllocatedDFBConfig(
                        dfb_index=local_id_map.get(original_id, original_id),
                        num_tiles=1,
                        data_format="float32",
                        block_count=max(1, attention_blocks[original_id]),
                    )
                    config._cb_index = local_id_map.get(
                        original_id, original_id
                    )
                    compiled.cb_configs[_PROJECTION_CBS + original_id] = config

                if resident_qkv_projection:
                    # V, K and Q reuse these FIFO IDs sequentially. Consuming
                    # a CB advances its pointers but does not rewind its
                    # physical allocation, so the full owner needs capacity
                    # for the complete publication history, not just Q's
                    # largest live window. This is the same plan already
                    # qualified by the isolated exact QKV owner.
                    sequential_capacities = (
                        _full_serial_qkv_cb_capacities(
                            m_tiles=m_tiles,
                            hidden_tiles=k_tiles,
                            kv_tiles=kv_heads * dim_tiles,
                            jobs_per_node=jobs_per_node,
                        )
                    )
                    for index, tiles in sequential_capacities.items():
                        compiled.cb_configs[index].shape = (1, 1)
                        compiled.cb_configs[index].block_count = tiles
                    # Compute-only exact-add and K/V->Q handoff scratch on the
                    # projection island. Attention owns the same local IDs on
                    # its disjoint cores, so no hardware CB collision exists.
                    for cb_index in (25, 26):
                        config = copy(template)
                        config.shape = (1, 1)
                        config.block_count = max(2, jobs_per_node)
                        config._cb_index = cb_index
                        compiled.cb_configs.append(config)
                    # Dedicated FP32 partial-reduction FIFO for Q. O/V/K keep
                    # their already-balanced CB2 history; Q starts clean.
                    for cb_index in (27,):
                        config = CompilerAllocatedDFBConfig(
                            dfb_index=cb_index,
                            num_tiles=1,
                            data_format="float32",
                            block_count=2,
                        )
                        config._cb_index = cb_index
                        compiled.cb_configs.append(config)
                    # Phase-local L1 words for K/Q compute->writer completion.
                    # They are addressed through CB allocation metadata but
                    # never use FIFO reserve/push/wait/pop operations.
                    for cb_index in (28, 29):
                        config = copy(template)
                        config.shape = (1, 1)
                        config.block_count = 1
                        config._cb_index = cb_index
                        compiled.cb_configs.append(config)
                    # K accumulated dX (CB30) and Q's exact-add scratch
                    # (CB31). Both hold BF16 tiles and avoid read->write
                    # turn-around on CB24/CB26 across projection phases.
                    for cb_index in (30, 31):
                        config = copy(template)
                        config.shape = (1, 1)
                        config.block_count = max(2, jobs_per_node)
                        config._cb_index = cb_index
                        compiled.cb_configs.append(config)
                    # Clean Q LoRA dH/dB gradient-input FIFOs. These mirror
                    # projection CB4/CB9's BF16 format but hold only Q's one
                    # phase, not the O/V/K publication history.
                    for cb_index, tiles in ((32, n_tiles), (33, m_tiles)):
                        config = copy(template)
                        config.shape = (1, 1)
                        config.block_count = tiles
                        config._cb_index = cb_index
                        compiled.cb_configs.append(config)
                    # Q dB/dA final publications use clean one-phase FIFOs.
                    # This distinguishes correct Q math from a stale fourth
                    # output epoch on the shared CB15/CB22 rings.
                    for cb_index in (34, 35):
                        config = copy(template)
                        config.shape = (1, 1)
                        config.block_count = 2
                        config._cb_index = cb_index
                        compiled.cb_configs.append(config)
                    # Clean Q LoRA B and saved-rank-activation inputs. These
                    # are the independent non-dY operands of dA and dB.
                    for cb_index, tiles in ((36, n_tiles), (37, m_tiles)):
                        config = copy(template)
                        config.shape = (1, 1)
                        config.block_count = tiles
                        config._cb_index = cb_index
                        compiled.cb_configs.append(config)
                    # The remaining Q-only LoRA workspace. FP32 partials copy
                    # their source config so unpack/destination format stays
                    # identical; all windows are single-phase capacities.
                    q_workspace = (
                        (38, 6, n_tiles),
                        (39, 7, 1),
                        (40, 8, 2),
                        (41, 11, m_tiles),
                        (42, 12, m_tiles),
                        (43, 13, 1),
                        (44, 14, 1),
                    )
                    for cb_index, source_index, tiles in q_workspace:
                        config = copy(compiled.cb_configs[source_index])
                        config.shape = (1, 1)
                        config.block_count = tiles
                        config._cb_index = cb_index
                        compiled.cb_configs.append(config)
                    # The final dHidden bug only appears when a native owner
                    # feeds another native owner below it: Q is the fourth
                    # serial projection epoch, so its frozen-base reduction
                    # and tail must not inherit O/V/K FIFO counters either.
                    # Copying the original configs preserves formats and the
                    # exact reduction/packing schedule while giving the
                    # complete Q projection a clean physical L1 lifetime.
                    q_projection_workspace = (
                        (45, 0),
                        (46, 1),
                        (47, 3),
                        (48, 2),
                        (49, 16),
                        (50, 17),
                        (51, 18),
                        (52, 19),
                        (53, 20),
                        (54, 21),
                        (55, 23),
                    )
                    for cb_index, source_index in q_projection_workspace:
                        config = copy(compiled.cb_configs[source_index])
                        config._cb_index = cb_index
                        compiled.cb_configs.append(config)
                    # Compute consumes the V/K accumulation through CB24, but
                    # the writer RISC does not share that consumer pointer and
                    # would otherwise export the first V tile still at its
                    # local front. Q publishes only the completed five-edge
                    # sum to this terminal-only FIFO, which the writer owns.
                    config = copy(compiled.cb_configs[24])
                    config._cb_index = 56
                    compiled.cb_configs.append(config)
                    sources = _full_attention_qkv_sources(
                        projection_replacements=projection_replacements,
                        attention_replacements=attention_replacements,
                        sequence_tiles=seq_tiles,
                        hidden_tiles=k_tiles,
                        kv_tiles=kv_heads * dim_tiles,
                        dimension_tiles=dim_tiles,
                        query_heads=query_heads,
                        kv_heads=kv_heads,
                        scale_bits=lora_scale_bits,
                        base_phase=base_phase,
                        output_tensor_index=26,
                    )
                    projection_sources = sources[:3]
                    attention_sources = sources[3:]
                else:
                    projection_sources = _projection_sources(
                        replacements=projection_replacements,
                        output_tensor_index=26,
                        diagnostic_write_dout=diagnostic_write_dout,
                        resident_qkv_handoff=resident_qkv_handoff,
                        core_output_row_offset=(
                            sequence_length + TILE_SIZE + hidden_features
                        ) // TILE_SIZE,
                    )
                    attention_sources = _attention_sources(
                        replacements=attention_replacements,
                        output_tensor_index=26,
                        output_row_offset=(
                            sequence_length + TILE_SIZE + hidden_features
                        ) // TILE_SIZE,
                        output_arena_col_tiles=k_tiles,
                        inverse_rope=attention_diagnostic is None,
                        resident_qkv_handoff=resident_qkv_handoff,
                    )
                if forward_checkpoint_arena:
                    projection_sources, attention_sources = (
                        _forward_checkpoint_arena_sources(
                            projection_sources,
                            attention_sources,
                            sequence_length=sequence_length,
                            hidden_features=hidden_features,
                            query_heads=query_heads,
                            kv_heads=kv_heads,
                            head_dim=head_dim,
                        )
                    )
                    projection_sources, attention_sources = (
                        _direct_checkpoint_arena_abi_sources(
                            projection_sources, attention_sources
                        )
                    )
                generated = Path(compiled.kernel_paths[0][0]).parent
                specs = (
                    ("output_attention_projection_reader.cpp", "noc", projection_sources[0]),
                    ("output_attention_projection_compute.cpp", "compute", projection_sources[1]),
                    ("output_attention_projection_writer.cpp", "noc", projection_sources[2]),
                    ("output_attention_core_reader.cpp", "noc", attention_sources[0]),
                    ("output_attention_core_compute.cpp", "compute", attention_sources[1]),
                    ("output_attention_core_writer.cpp", "noc", attention_sources[2]),
                )
                paths = []
                for filename, kind, source in specs:
                    path = generated / filename
                    path.write_text(source, encoding="utf-8")
                    dump_root = os.environ.get(
                        "TT_FINETUNE_SOURCE_DUMP_DIR"
                    )
                    if dump_root:
                        dump = Path(dump_root) / (
                            f"full_attention_{sequence_length}x"
                            f"{hidden_features}_q{query_heads}_kv{kv_heads}_d"
                            f"{head_dim}"
                        )
                        dump.mkdir(parents=True, exist_ok=True)
                        (dump / filename).write_text(source, encoding="utf-8")
                    paths.append((str(path), kind))

                import ttnn

                attention_compute = ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=True,
                    fp32_dest_acc_en=True,
                )
                full_fp32 = {0, 1, 13, 14, 15}
                for cb_index in range(64):
                    attention_compute.unpack_to_dest_mode.append(
                        ttnn.UnpackToDestMode.UnpackToDestFp32
                        if cb_index in full_fp32
                        else ttnn.UnpackToDestMode.Default
                    )
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
                    attention_compute,
                    ttnn.WriterConfigDescriptor(),
                ]
                if forward_checkpoint_arena:
                    compiled.kernel_tensor_indices = [
                        [
                            0, 1, 2, 3, 6,
                            7, 8, 9,
                            10, 11, 12,
                            13, 14, 15,
                            16,
                        ],
                        [], [17],
                        [4, 5, 16],
                        [], [17],
                    ]
                else:
                    compiled.kernel_tensor_indices = [
                        (
                            list(range(6)) + list(range(12, 25))
                            if resident_qkv_projection
                            else list(range(6))
                        ),
                        [], [26],
                        [6, 7, 8, 9, 10, 11],
                        [], [26],
                    ]
                compiled.kernel_core_ranges = [
                    _PROJECTION_RANGE, _PROJECTION_RANGE, _PROJECTION_RANGE,
                    _ATTENTION_RANGE, _ATTENTION_RANGE, _ATTENTION_RANGE,
                ]
                compiled.kernel_noc_roles = [
                    "reader", None, "writer", "reader", None, "writer",
                ]
                compiled.cb_core_ranges = (
                    [_PROJECTION_RANGE] * _PROJECTION_CBS
                    + [
                        (
                            _ATTENTION_RANGE
                        )
                        for original_id in range(_ATTENTION_CBS)
                    ]
                    + (
                        [_PROJECTION_RANGE] * 32
                        if resident_qkv_projection
                        else []
                    )
                )
                dump_root = os.environ.get("TT_FINETUNE_SOURCE_DUMP_DIR")
                if dump_root:
                    dump = Path(dump_root) / (
                        f"full_attention_{sequence_length}x"
                        f"{hidden_features}_q{query_heads}_kv{kv_heads}_d"
                        f"{head_dim}"
                    )
                    metadata = {
                        "cb_configs": [
                            {
                                key: str(value)
                                for key, value in vars(config).items()
                            }
                            for config in compiled.cb_configs
                        ],
                        "cb_core_ranges": compiled.cb_core_ranges,
                        "kernel_tensor_indices": (
                            compiled.kernel_tensor_indices
                        ),
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
                    for semaphore in range(16)
                ]
                compiled.num_pipe_sync_semaphores = 16
                compiled.kernel_arg_specs = [[] for _ in range(6)]
                if artifact_program_sequence:
                    compiled.wrap_as_program_sequence = True
            break
        return result

    if forward_checkpoint_arena:
        @tt_lang_operation(
            operation_id=(
                f"tt_finetune.output_attention_backward.{variant}"
            ),
            arg_roles=("in",) * 17 + ("out",),
            version_tag=f"output-attention-backward-{variant}",
        )
        def operation(
            output_dy, o_weight, o_b, o_a, cosine, sine, hidden,
            q_weight, q_b, q_a,
            k_weight, k_b, k_a,
            v_weight, v_b, v_a,
            checkpoint_arena, arena,
        ):
            return _invoke_operation((
                output_dy, o_weight, o_b, o_a, cosine, sine, hidden,
                q_weight, q_b, q_a,
                k_weight, k_b, k_a,
                v_weight, v_b, v_a,
                checkpoint_arena,
                arena,
            ))
    else:
        @tt_lang_operation(
            operation_id=(
                f"tt_finetune.output_attention_backward.{variant}"
            ),
            arg_roles=("in",) * 26 + ("out",),
            version_tag=f"output-attention-backward-{variant}",
        )
        def operation(
            output_dy, o_weight, o_b, o_hidden, o_a, head_output,
            probability, query, key, value, cosine, sine,
            hidden,
            q_weight, q_b, q_hidden, q_a,
            k_weight, k_b, k_hidden, k_a,
            v_weight, v_b, v_hidden, v_a,
            checkpoint_arena,
            arena,
        ):
            return _invoke_operation((
                output_dy, o_weight, o_b, o_hidden, o_a, head_output,
                probability, query, key, value, cosine, sine,
                hidden,
                q_weight, q_b, q_hidden, q_a,
                k_weight, k_b, k_hidden, k_a,
                v_weight, v_b, v_hidden, v_a,
                checkpoint_arena,
                arena,
            ))

    operation.handoff_plan = handoff
    operation.attention_plan = core_plan
    operation.arena_shape = (
        full_attention_backward_arena_shape(
            tokens=sequence_length,
            hidden_features=hidden_features,
            kv_features=kv_heads * head_dim,
        )
        if resident_qkv_projection
        else output_attention_arena_shape(
            tokens=sequence_length,
            hidden_features=hidden_features,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
    )
    operation.intermediate_dram_outputs = 0
    operation.experimental_backend = "output_attention_resident_rope_v2"
    operation.resident_qkv_handoff = resident_qkv_handoff
    operation.resident_qkv_projection = resident_qkv_projection
    operation.artifact_program_sequence = bool(artifact_program_sequence)
    # Artifact-v4 parent owners compile this already-qualified program as one
    # physical child. Exposing the selected direct-checkpoint program avoids
    # reconstructing or wrapping its numerical schedule above TT-Metal.
    operation._ttl_program = selected_program
    return operation
