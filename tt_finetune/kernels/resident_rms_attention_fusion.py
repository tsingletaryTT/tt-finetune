"""Physical RMSNorm-backward -> attention-backward resident handoff.

This module fuses the accepted compact RMSNorm backward and the accepted
single-program attention backward without changing either arithmetic body.
The final rounded RMSNorm ``dY`` row remains in a dedicated L1 circular
buffer.  Projection readers fetch those tiles directly over NoC, then a
device-side credit barrier releases the producer.  No TensorAccessor is used
for the handoff.

The transformation is intentionally performed on qualified serialized child
programs.  It is private infrastructure for the experimental full attention
block owner; the safe artifact-v4 sequence remains available as fallback.
"""

from __future__ import annotations

import copy
import os
import re
from typing import Mapping, Sequence

from .resident_layer_source_composer import (
    ResidentKernelPhase,
    compose_resident_kernel_phases,
)


_PROJECTION_RANGE = {"start": [0, 0], "end": [4, 9]}
_ATTENTION_RANGE = {"start": [5, 0], "end": [10, 9]}
_BF16_BYTES = 2048
_FP32_BYTES = 4096


def _dtype(config: Mapping[str, object]) -> str:
    value = str(config["data_format"])
    if value == "BFloat16":
        return "bf16"
    if value == "Float32":
        return "fp32"
    raise ValueError(f"unsupported resident CB format: {value}")


def _is_range(config: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    return config.get("core_range") == expected


def _replace_exact_once(source: str, old: str, new: str, *, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"{label} marker changed (found {source.count(old)})")
    return source.replace(old, new, 1)


def _projection_descriptor_layout(
    rms_program: Mapping[str, object],
    attention_program: Mapping[str, object],
) -> tuple[list[dict], dict[int, int], dict[int, int], int, int]:
    """Build a fixed-ID phase alias plan for the projection core island.

    Attention's hardware CB IDs are preserved because some qualified phase
    bodies use literal IDs.  RMS buffers reuse compatible attention IDs when
    possible.  Attention's 57 IDs provide enough distinct same-format
    lifetimes for every dead RMS scratch buffer.  Only the live handoff output
    needs another ID because it overlaps the first attention reader epoch.
    """

    attention_configs = list(attention_program["cb_configs"])
    projection_entries = [
        (index, copy.deepcopy(config))
        for index, config in enumerate(attention_configs)
        if _is_range(config, _PROJECTION_RANGE)
    ]
    island_entries = [
        (index, copy.deepcopy(config))
        for index, config in enumerate(attention_configs)
        if _is_range(config, _ATTENTION_RANGE)
    ]
    if len(projection_entries) != 57 or len(island_entries) != 38:
        raise ValueError(
            "accepted attention descriptor split changed: "
            f"projection={len(projection_entries)}, island={len(island_entries)}"
        )

    by_hw: dict[int, dict] = {}
    attention_arg_to_hw: dict[int, int] = {}
    for old_arg, config in projection_entries:
        hw = int(config["buffer_index"])
        if hw in by_hw:
            raise ValueError(f"duplicate projection CB id {hw}")
        if not 0 <= hw < 57:
            raise ValueError(f"unexpected projection CB id {hw}")
        config["alias_slot"] = hw
        by_hw[hw] = config
        attention_arg_to_hw[old_arg] = hw

    # CB25 is a raw count-word compute->writer phase fence throughout the
    # accepted projection program.  No producer reserves/pushes tile payloads
    # into it, so its inherited four-page allocation is unnecessary.  One
    # aligned page preserves the exact protocol and funds a deeper streaming
    # RMS handoff without increasing peak L1.
    if _dtype(by_hw[25]) != "bf16":
        raise ValueError("attention phase fence CB25 format changed")
    by_hw[25]["total_size"] = int(by_hw[25]["page_size"])
    by_hw[25]["num_tiles"] = 1
    by_hw[25]["block_count"] = 1

    rms_configs = copy.deepcopy(list(rms_program["cb_configs"]))
    if len(rms_configs) != 21:
        raise ValueError("compact RMSNorm backward must expose 21 CBs")
    # The standalone child inherits FP32 descriptors for five one-page
    # lifecycle FIFOs from the pre-compaction graph.  They carry count tokens,
    # never numerical tiles; the qualified compact planner therefore models
    # them as BF16 one-page slots.  Normalize that latent metadata here so it
    # does not consume scarce FP32 hardware IDs in the fused program.
    for control_index in range(13, 18):
        rms_configs[control_index].update(
            data_format="BFloat16",
            page_size=_BF16_BYTES,
            total_size=_BF16_BYTES,
            num_tiles=1,
            block_count=1,
        )

    # Scratch lifetimes are strictly before attention and therefore reuse
    # attention's hardware IDs/backing slots.  Unpack modes are merged into
    # the one compute kernel below.  The exact silicon gate decides whether a
    # particular same-format assignment is numerically compatible.
    # Every RMS scratch lifetime ends before attention consumes its first
    # input.  Do not allocate a synthetic cross-RISC release FIFO here:
    # compute-side CircularBuffer APIs only address hardware IDs 0..31, and a
    # wait on such a FIFO would park UNPACK without parking MATH/PACK.  The
    # real attention input CB dependency is the all-TRISC fence; the BRISC
    # semaphore below prevents its producer epoch from retiring early.
    # The four RMS tail cores have their own disjoint compute kernel and may
    # therefore apply RMS's Fp32 unpack modes without perturbing attention's
    # arithmetic cores.  Storage can alias any compatible, sufficiently large
    # attention slot because the reader/writer lifetime fence remains global.
    dedicated_scratch: set[int] = set()
    live_output = 18
    free_hw = iter(range(57, 64))
    used_existing: set[int] = set()
    rms_to_hw: dict[int, int] = {}

    def install_new(rms_index: int, *, live: bool) -> int:
        try:
            hw = next(free_hw)
        except StopIteration as exc:  # pragma: no cover - shape gate
            raise ValueError("RMS/attention fusion exhausted 64 CB IDs") from exc
        source = rms_configs[rms_index]
        config = copy.deepcopy(source)
        config["buffer_index"] = hw
        config["core_range"] = copy.deepcopy(_PROJECTION_RANGE)
        if live:
            # RMS publishes a small tile block and its writer drains directly
            # into the compiler-owned distributed L1 handoff tensor.  Keep a
            # dedicated six-page FIFO on every projection core for producer
            # flow control and the BRISC/NCRISC rendezvous; the full hidden row
            # must never be replicated in every core's static CB region.
            config["total_size"] = 6 * int(config["page_size"])
            config["num_tiles"] = 6
            config["block_count"] = 6
            config["alias_slot"] = hw
        else:
            candidates = sorted(
                (candidate for candidate in by_hw if candidate not in used_existing),
                key=lambda candidate: (
                    # Preserve the scarce attention FP32 IDs for the five
                    # small RMS FP32 buffers that can reuse them directly.
                    0 if _dtype(by_hw[candidate]) == "bf16" else 1,
                    max(
                        0,
                        int(source["total_size"])
                        - int(by_hw[candidate]["total_size"]),
                    ),
                    -int(by_hw[candidate]["total_size"]),
                    candidate,
                ),
            )
            alias_hw = candidates[0]
            used_existing.add(alias_hw)
            capacity = max(
                int(by_hw[alias_hw]["total_size"]),
                int(source["total_size"]),
            )
            # Artifact-v3 aliases name one physical backing allocation.  All
            # logical formats on that slot must describe the same byte size.
            by_hw[alias_hw]["total_size"] = capacity
            by_hw[alias_hw]["num_tiles"] = (
                capacity // int(by_hw[alias_hw]["page_size"])
            )
            by_hw[alias_hw]["block_count"] = by_hw[alias_hw]["num_tiles"]
            config["total_size"] = capacity
            config["num_tiles"] = capacity // int(config["page_size"])
            config["block_count"] = config["num_tiles"]
            config["alias_slot"] = alias_hw
        by_hw[hw] = config
        return hw

    for rms_index in sorted(dedicated_scratch):
        rms_to_hw[rms_index] = install_new(rms_index, live=False)
    rms_to_hw[live_output] = install_new(live_output, live=True)

    # Reuse distinct same-format attention IDs for the remaining simultaneous
    # RMS buffers.  Capacity is enlarged when necessary; the alias backing is
    # still one allocation and attention's literal hardware ID remains valid.
    for rms_index, rms_config in enumerate(rms_configs):
        if rms_index in rms_to_hw:
            continue
        candidates = [
            hw for hw, config in by_hw.items()
            if hw < 57
            and hw not in used_existing
            and _dtype(config) == _dtype(rms_config)
        ]
        if not candidates:
            raise ValueError(f"no compatible CB for RMS index {rms_index}")
        hw = min(
            candidates,
            key=lambda candidate: (
                max(
                    0,
                    int(rms_config["total_size"])
                    - int(by_hw[candidate]["total_size"]),
                ),
                -int(by_hw[candidate]["total_size"]),
                candidate,
            ),
        )
        used_existing.add(hw)
        capacity = max(
            int(by_hw[hw]["total_size"]), int(rms_config["total_size"])
        )
        by_hw[hw]["total_size"] = capacity
        by_hw[hw]["num_tiles"] = capacity // int(by_hw[hw]["page_size"])
        by_hw[hw]["block_count"] = by_hw[hw]["num_tiles"]
        rms_to_hw[rms_index] = hw

    if set(rms_to_hw) != set(range(21)):
        raise AssertionError("incomplete RMS CB assignment")
    if len(set(rms_to_hw.values())) != 21:
        raise AssertionError("simultaneous RMS CBs must have unique IDs")

    projection_configs = [by_hw[hw] for hw in sorted(by_hw)]
    projection_pos = {
        int(config["buffer_index"]): position
        for position, config in enumerate(projection_configs)
    }
    attention_arg_map = {
        old_arg: projection_pos[hw]
        for old_arg, hw in attention_arg_to_hw.items()
    }
    rms_arg_map = {
        rms_index: projection_pos[hw] for rms_index, hw in rms_to_hw.items()
    }

    alias_offset = 58
    island_configs = []
    island_arg_map: dict[int, int] = {}
    for local, (old_arg, config) in enumerate(island_entries):
        config["alias_slot"] = alias_offset + local
        island_arg_map[old_arg] = len(projection_configs) + local
        island_configs.append(config)

    all_configs = projection_configs + island_configs
    attention_arg_map.update(island_arg_map)
    if os.environ.get("TT_FINETUNE_FUSED_RMS_DUMP_CB_MAP") == "1":
        print(
            "FUSED_RMS_SOURCE_CONFIGS",
            {
                "rms": [
                    {
                        "arg": index,
                        "format": str(config["data_format"]),
                        "bytes": int(config["total_size"]),
                    }
                    for index, config in enumerate(rms_configs)
                ],
                "attention_projection": [
                    {
                        "arg": old_arg,
                        "hw": int(config["buffer_index"]),
                        "format": str(config["data_format"]),
                        "bytes": int(config["total_size"]),
                    }
                    for old_arg, config in projection_entries
                ],
            },
            flush=True,
        )
        print(
            "FUSED_RMS_CB_MAP",
            {
                "rms_to_hw": rms_to_hw,
                "attention_arg_to_hw": attention_arg_to_hw,
                "output_hw": rms_to_hw[live_output],
            },
            flush=True,
        )
        print(
            "FUSED_RMS_CB_CONFIGS",
            [
                {
                    "hw": int(config["buffer_index"]),
                    "alias": int(config["alias_slot"]),
                    "format": str(config["data_format"]),
                    "bytes": int(config["total_size"]),
                }
                for config in projection_configs
            ],
            flush=True,
        )
    return (
        all_configs,
        rms_arg_map,
        attention_arg_map,
        rms_to_hw[live_output],
        len(projection_configs),
    )


def _remove_rms_reader_release(source: str) -> str:
    marker = '''    // A following compact program may start its NCRISC as soon as this
    // reader exits, while this program's BRISC is still draining output NoC
    // writes. Keep the reader epoch alive until the writer publishes its
    // local completion token.
    CircularBuffer reader_release(cb_reader_release);
    reader_release.wait_front(1);
    reader_release.pop_front(1);
'''
    return _replace_exact_once(
        source, marker, "", label="RMS reader release"
    )


def _remove_rms_compute_release(source: str) -> str:
    """Move the compact RMS tail release into the composed phase wrapper.

    The standalone compact kernel only runs arithmetic on ``token_tiles``
    compute cores, so its original release wait parks four TRISCs and lets the
    other 46 enter the aliased attention CB epoch immediately.  The composed
    owner needs the same wait on every projection core.  Removing this local
    tail and re-emitting it after ``run()`` gives all 50 cores one lifecycle.
    """

    marker = '''    // Match the reader/writer tail fence: the next physical program must not
    // reuse this core's compute RISC before BRISC has made output visible.
    CircularBuffer compute_release(get_compile_time_arg_val(14));
    compute_release.wait_front(1);
    compute_release.pop_front(1);
'''
    return _replace_exact_once(
        source, marker, "", label="RMS compute release"
    )


def _move_rms_phase_to_tail(source: str, *, token_tiles: int) -> str:
    """Run compact RMS only on projection cores unused by LoRA output work.

    A physical TRISC kernel may perform its hardware startup only once.  The
    first ``token_tiles`` attention workers immediately execute the LoRA dH
    and dB paths, so making those same cores start in RMS mode creates an
    unsupported engine transition inside the fused kernel.  The last workers
    in the 5x10 projection island have no output-projection jobs for the
    accepted shapes.  Put the four RMS token rows there and use NoC for the
    resident handoff, leaving the attention-critical cores in one matmul
    engine epoch from the beginning.
    """

    worker_base = 50 - token_tiles
    marker = '''    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= token_tiles) return;'''
    replacement = f'''    constexpr uint32_t rms_worker_base = {worker_base};
    const uint32_t worker_node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();
    if (worker_node < rms_worker_base) return;
    const uint32_t node = worker_node - rms_worker_base;
    if (node >= token_tiles) return;'''
    return _replace_exact_once(
        source, marker, replacement, label="RMS tail-worker assignment"
    )


def _prepare_tail_rms_reader_controller(
    source: str, *, worker_base: int
) -> str:
    """Clear the producer-arrival counter before any RMS reader can finish."""

    marker = f'''    constexpr uint32_t rms_worker_base = {worker_base};
    const uint32_t worker_node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();'''
    replacement = marker + r'''
    if (worker_node == rms_worker_base) {
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(13)), 0);
    }'''
    return _replace_exact_once(
        source, marker, replacement, label="RMS tail controller preflight"
    )


def _rms_reader_handoff_barrier(
    *, hidden_tiles: int, output_hw_id: int, token_tiles: int,
    worker_base: int,
) -> str:
    source = r'''constexpr uint32_t rms_ready_count_sem = 0;
constexpr uint32_t rms_go_sem = 1;
constexpr uint32_t rms_reset_ring_sem = 2;
constexpr uint32_t rms_consumed_count_sem = 3;
constexpr uint32_t rms_producer_arrival_sem = 13;
constexpr uint32_t rms_worker_base = @WORKER_BASE@;
constexpr uint32_t rms_token_tiles = @TOKEN_TILES@;
constexpr uint8_t rms_projection_x[5] = {1, 2, 3, 4, 5};
const uint32_t rms_phase_node =
    get_absolute_logical_y() * 5 + get_absolute_logical_x();
for (uint32_t semaphore = 0; semaphore < 13; ++semaphore) {
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(semaphore)), 0);
}
volatile tt_l1_ptr uint32_t* rms_reset_ring =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(rms_reset_ring_sem));
const uint32_t rms_controller_x =
    rms_projection_x[rms_worker_base % 5];
const uint32_t rms_controller_y = 2 + rms_worker_base / 5;
if (rms_phase_node >= rms_worker_base &&
    rms_phase_node < rms_worker_base + rms_token_tiles) {
    noc_semaphore_inc(
        get_noc_addr(
            rms_controller_x, rms_controller_y,
            get_semaphore(rms_producer_arrival_sem)), 1);
    noc_async_atomic_barrier();
}
if (rms_phase_node == rms_worker_base) {
    volatile tt_l1_ptr uint32_t* producer_arrival =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(rms_producer_arrival_sem));
    noc_semaphore_wait_min(producer_arrival, rms_token_tiles);
    const uint32_t first = (rms_worker_base + 1) % 50;
    noc_semaphore_inc(
        get_noc_addr(
            rms_projection_x[first % 5], 2 + first / 5,
            get_semaphore(rms_reset_ring_sem)), 1);
    noc_async_atomic_barrier();
    noc_semaphore_wait_min(rms_reset_ring, 1);
    noc_semaphore_set(rms_reset_ring, 2);
    for (uint32_t target = 0; target < 50; ++target) {
        if (target == rms_worker_base) continue;
        noc_async_write(
            get_semaphore(rms_reset_ring_sem),
            get_noc_addr(rms_projection_x[target % 5], 2 + target / 5,
                         get_semaphore(rms_reset_ring_sem)),
            sizeof(uint32_t));
    }
    noc_async_write_barrier();
} else {
    noc_semaphore_wait_min(rms_reset_ring, 1);
    const uint32_t next = (rms_phase_node + 1) % 50;
    noc_semaphore_inc(
        get_noc_addr(rms_projection_x[next % 5], 2 + next / 5,
                     get_semaphore(rms_reset_ring_sem)), 1);
    noc_async_atomic_barrier();
}
noc_semaphore_wait_min(rms_reset_ring, 2);
volatile tt_l1_ptr uint32_t* rms_go =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(rms_go_sem));
// BRISC streams every produced tile into the distributed L1 tensor and only
// then publishes generation one.  Waiting here keeps the attention reader on
// a real data-availability dependency without retaining a complete row in a
// per-core CB.
noc_semaphore_wait_min(rms_go, 1);'''
    return (
        source.replace("@WORKER_BASE@", str(worker_base))
        .replace("@TOKEN_TILES@", str(token_tiles))
        .replace("@HIDDEN_TILES@", str(hidden_tiles))
        .replace("@OUTPUT_HW_ID@", str(output_hw_id))
    )


def _replace_rms_writer(
    source: str, *, handoff_hw_id: int, handoff_tensor_arg: int,
    handoff_tensor_index: int, total_cbs: int, worker_base: int = 0,
) -> str:
    hidden_match = re.search(
        r"constexpr uint32_t hidden_tiles = (\d+);", source
    )
    token_match = re.search(
        r"constexpr uint32_t token_tiles = (\d+);", source
    )
    if hidden_match is None or token_match is None:
        raise ValueError("expanded RMS writer shape constants changed")
    hidden_tiles = hidden_match.group(1)
    token_tiles = token_match.group(1)
    marker = "void kernel_main() {"
    if source.count(marker) != 1:
        raise ValueError("RMS writer kernel marker changed")
    prefix = source.split(marker, 1)[0]
    result = prefix + r'''void kernel_main() {
    constexpr uint32_t cb_output = get_compile_time_arg_val(18);
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t token_tiles = @TOKEN_TILES@;
    constexpr uint32_t worker_base = @WORKER_BASE@;
    constexpr uint32_t ready_count_sem = 0;
    constexpr uint32_t go_sem = 1;
    constexpr uint32_t reset_ring_sem = 2;
    constexpr uint32_t consumed_count_sem = 3;
    constexpr uint8_t projection_x[5] = {1, 2, 3, 4, 5};
    const uint32_t worker_node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();
    const bool producer =
        worker_node >= worker_base && worker_node < worker_base + token_tiles;
    const uint32_t node = worker_node - worker_base;
    const uint32_t controller_x = projection_x[worker_base % 5];
    const uint32_t controller_y = 2 + worker_base / 5;
    volatile tt_l1_ptr uint32_t* go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(go_sem));

    if (producer) {
    CircularBuffer output(cb_output);
    volatile tt_l1_ptr uint32_t* reset_ring =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(reset_ring_sem));
    noc_semaphore_wait_min(reset_ring, 2);

    // Drain the one-page RMS output FIFO as compute publishes it and scatter
    // the complete row into compiler-owned distributed L1.  This replaces a
    // hidden_tiles-deep CB replicated on all 50 projection cores with one
    // page per core while preserving the exact BF16 producer boundary.
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t handoff_addr =
        get_common_arg_val<uint32_t>(@HANDOFF_TENSOR_ARG@);
    const auto handoff_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                @HANDOFF_TENSOR_INDEX@, @TOTAL_CBS@>(),
            @HANDOFF_TENSOR_ARG@>(),
        handoff_addr, tile_bytes);
    Noc handoff_noc(1);
    constexpr uint32_t handoff_block_tiles = 6;
    for (uint32_t block = 0; block < hidden_tiles;
         block += handoff_block_tiles) {
        const uint32_t block_tiles =
            (hidden_tiles - block < handoff_block_tiles)
                ? hidden_tiles - block : handoff_block_tiles;
        output.wait_front(block_tiles);
        for (uint32_t local = 0; local < block_tiles; ++local) {
            handoff_noc.async_write(
                CoreLocalMem<uint32_t>(
                    output.get_read_ptr() + local * tile_bytes),
                handoff_tensor, tile_bytes, {},
                {.page_id = node * hidden_tiles + block + local});
        }
        handoff_noc.async_write_barrier();
        output.pop_front(block_tiles);
    }
    noc_semaphore_inc(
        get_noc_addr(
            controller_x, controller_y, get_semaphore(ready_count_sem)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* ready_count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(ready_count_sem));
        noc_semaphore_wait_min(ready_count, token_tiles);
        noc_semaphore_set(go, 1);
        for (uint32_t target = 0; target < 50; ++target) {
            if (target == worker_base) continue;
            noc_async_write(
                get_semaphore(go_sem),
                get_noc_addr(projection_x[target % 5], 2 + target / 5,
                             get_semaphore(go_sem)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait_min(go, 1);

    if (node == 0) {
        volatile tt_l1_ptr uint32_t* consumed =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(consumed_count_sem));
        noc_semaphore_wait_min(consumed, 50);
        volatile tt_l1_ptr uint32_t* ready_count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(ready_count_sem));
        noc_semaphore_set(ready_count, 0);
        noc_semaphore_set(go, 2);
        for (uint32_t target = 1; target < token_tiles; ++target) {
            const uint32_t target_worker = worker_base + target;
            noc_async_write(
                get_semaphore(go_sem),
                get_noc_addr(
                    projection_x[target_worker % 5],
                    2 + target_worker / 5,
                    get_semaphore(go_sem)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait_min(go, 2);
    noc_semaphore_inc(
        get_noc_addr(
            controller_x, controller_y, get_semaphore(ready_count_sem)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* ready_count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(ready_count_sem));
        noc_semaphore_wait_min(ready_count, token_tiles);
        noc_semaphore_set(ready_count, 0);
        noc_semaphore_set(go, 3);
        for (uint32_t target = 0; target < 50; ++target) {
            if (target == worker_base) continue;
            noc_async_write(
                get_semaphore(go_sem),
                get_noc_addr(projection_x[target % 5], 2 + target / 5,
                             get_semaphore(go_sem)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    // Close the writer epoch here, before the wrapper can start attention and
    // before any reader is allowed to reset semaphore one.  This single wait
    // covers both producer and non-producer BRISCs; an outer wait would race
    // the reader's semaphore-lifetime transition.
    noc_semaphore_wait_min(go, 3);
    } else {
        // Inactive projection writers join the same release generation as
        // the four producer writers before leaving the RMS phase.
        noc_semaphore_wait_min(go, 3);
    }

    // Keep lifecycle state outside attention's semaphore IDs 0..8.  Every
    // BRISC first acknowledges release generation three at node zero.  Only
    // after all 50 acknowledgements does node zero publish writer-done.
    // Semaphores 9..12 remain live until the next invocation's globally
    // ordered reset, so a fast reader cannot erase a remote writer's credit.
    constexpr uint32_t writer_ack_sem = 0;
    constexpr uint32_t writer_done_sem = 1;
    constexpr uint32_t reader_reset_count_sem = 2;
    constexpr uint32_t attention_start_sem = 3;
    constexpr uint32_t handoff_cb = @HANDOFF_HW_ID@;
    noc_semaphore_inc(
        get_noc_addr(
            controller_x, controller_y, get_semaphore(writer_ack_sem)),
        1);
    noc_async_atomic_barrier();
    volatile tt_l1_ptr uint32_t* writer_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(writer_done_sem));
    if (worker_node == worker_base) {
        volatile tt_l1_ptr uint32_t* writer_ack =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(writer_ack_sem));
        noc_semaphore_wait_min(writer_ack, 50);
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(reader_reset_count_sem)), 0);
        noc_semaphore_set(writer_done, 4);
        for (uint32_t target = 0; target < 50; ++target) {
            if (target == worker_base) continue;
            // Every writer has already observed release generation three.
            // Advance the remote word atomically to generation four instead
            // of copying the controller's local semaphore value; this avoids
            // a source-word publication race on a controller that is no
            // longer projection node zero.
            noc_semaphore_inc(
                get_noc_addr(
                    projection_x[target % 5], 2 + target / 5,
                    get_semaphore(writer_done_sem)),
                1);
        }
        noc_async_atomic_barrier();
    }
    volatile tt_l1_ptr uint32_t* attention_start =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(attention_start_sem));
    noc_semaphore_wait_min(attention_start, 1);
    CircularBuffer handoff(handoff_cb);
    volatile tt_l1_ptr uint32_t* handoff_word =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            handoff.get_write_ptr());
    *handoff_word = 1;
    while (*handoff_word != 2) {}
    *handoff_word = 3;
}
'''
    return (
        result.replace("@HIDDEN_TILES@", hidden_tiles)
        .replace("@TOKEN_TILES@", token_tiles)
        .replace("@WORKER_BASE@", str(worker_base))
        .replace("@HANDOFF_HW_ID@", str(handoff_hw_id))
        .replace("@HANDOFF_TENSOR_ARG@", str(handoff_tensor_arg))
        .replace("@HANDOFF_TENSOR_INDEX@", str(handoff_tensor_index))
        .replace("@TOTAL_CBS@", str(total_cbs))
    )


def _resident_dy_helper(
    *, output_hw_id: int, hidden_tiles: int, worker_base: int
) -> str:
    return f'''\nnamespace resident_post_rms_handoff {{
constexpr uint32_t kOutputCb = {output_hw_id};
constexpr uint32_t kHiddenTiles = {hidden_tiles};
constexpr uint32_t kWorkerBase = {worker_base};
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kPhysicalX[5] = {{1, 2, 3, 4, 5}};
FORCE_INLINE uint64_t noc_address(uint32_t page_id) {{
    const uint32_t token = page_id / kHiddenTiles;
    const uint32_t tile = page_id - token * kHiddenTiles;
    const uint32_t worker = kWorkerBase + token;
    CircularBuffer output(kOutputCb);
    return get_noc_addr(
        kPhysicalX[worker % 5], 2 + worker / 5,
        output.get_read_ptr() + tile * kTileBytes);
}}
}}  // namespace resident_post_rms_handoff
\n'''


def _writer_handoff_retired_barrier() -> str:
    """The RMS writer now retires all 50 BRISCs inside its own phase.

    Waiting again after ``run()`` races the reader's legal transition to the
    attention semaphore epoch: the reader observes generation three and then
    resets semaphore one while the outer writer wrapper is trying to observe
    the same generation a second time.  Keep this hook for source-composer
    symmetry, but emit no second wait.
    """

    return ""


def _compute_handoff_retired_barrier(
    *, product_cb_arg: int, scaler_cb_arg: int, token_tiles: int,
    worker_base: int = 0,
) -> str:
    """Retire persistent RMS scratch before attention reuses its CB slot.

    ``calculate_and_prepare_reduce_scaler`` intentionally leaves its constant
    tile resident until standalone program teardown.  RMS reduction also
    leaves Blackhole's MATH reduction mode and PACK edge mask configured for
    the standalone program lifetime.  The fused program has no teardown
    between phases.  Explicitly restore the reduction engines and pop the
    scaler on the active RMS cores before attention reuses those engines and
    aliases the scratch CBs.  There is deliberately no separate
    compute-release FIFO.  CircularBuffer waits execute only on UNPACK, so
    they cannot fence the three compute RISCs.  Instead the first attention
    matmul's ordinary input-CB and tile-register handshakes provide the
    all-engine dependency after the reader/BRISC NoC protocol.
    """

    return rf'''constexpr uint32_t rms_compute_worker_base = {worker_base};
const uint32_t rms_compute_node =
    get_absolute_logical_y() * 5 + get_absolute_logical_x();
if (rms_compute_node >= rms_compute_worker_base &&
    rms_compute_node < rms_compute_worker_base + {token_tiles}) {{
    // Standalone RMS teardown normally retires this state at program exit.
    // Without an explicit retirement, only these RMS-active cores retain the
    // reduction PACK mask and later deadlock in output-projection LoRA dB.
    reduce_uninit(get_compile_time_arg_val({product_cb_arg}));
    CircularBuffer rms_reduce_scaler(get_compile_time_arg_val({scaler_cb_arg}));
    rms_reduce_scaler.pop_front(1);
}}
// The reader publishes attention's real input CBs only after writer-done
// generation four, which itself follows the final handoff pop.  Those input
// waits are the cross-RISC/all-TRISC compute fence; a synthetic control CB is
// deliberately avoided because its FIFO state is not a BRISC/TRISC barrier.'''


def _remove_rms_compute_startup(source: str) -> str:
    """Let the composed owner perform the one legal hardware startup."""

    return _replace_exact_once(
        source,
        "    compute_kernel_hw_startup(cb_x, cb_scaler, cb_output);\n",
        "",
        label="compact RMS hardware startup",
    )


def _prepend_composed_compute_startup(
    source: str, *,
    dy_cb_arg: int,
    weight_cb_arg: int,
    partial_cb_arg: int,
    rms_x_cb_arg: int,
    rms_scaler_cb_arg: int,
    rms_output_cb_arg: int,
    rms_worker_base: int,
) -> str:
    """Start every TRISC once in the first operation it actually executes.

    The compact RMS phase runs only on the tail token workers.  Starting those
    cores with attention's reverse-matmul descriptors and relying on ordinary
    operation reconfiguration is not numerically equivalent to the qualified
    standalone RMS kernel: its first BF16 row-broadcast/reduction chain inherits
    the wrong initial UNPACK/PACK contract.  Conversely, the other projection
    cores must retain attention startup because they never execute RMS.

    This branch is core-static.  Every physical compute core still executes
    exactly one public ``compute_kernel_hw_startup`` call.  The accepted
    mid-kernel attention reconfiguration below transitions the RMS tail cores
    after their output has been retired.
    """

    call = "    resident_layer_phase_post_rms_backward::run();"
    startup = f'''    // One physical compute kernel permits one hardware startup.  Select
    // the configuration of the first operation owned by this particular
    // core; the branch is invariant for the complete program invocation.
    constexpr uint32_t composed_rms_worker_base = {rms_worker_base};
    const uint32_t composed_worker =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();
    if (composed_worker >= composed_rms_worker_base) {{
        compute_kernel_hw_startup(
            get_compile_time_arg_val({rms_x_cb_arg}),
            get_compile_time_arg_val({rms_scaler_cb_arg}),
            get_compile_time_arg_val({rms_output_cb_arg}));
    }} else {{
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            get_compile_time_arg_val({dy_cb_arg}),
            get_compile_time_arg_val({weight_cb_arg}),
            get_compile_time_arg_val({partial_cb_arg}));
    }}
{call}'''
    return _replace_exact_once(
        source, call, startup, label="composed compute startup"
    )


def _inject_resident_rms_export_diagnostic(
    source: str,
    *,
    arena_tensor_arg: int,
    arena_tensor_index: int,
    total_cbs: int,
    hidden_tiles: int,
    token_tiles: int,
    kv_tiles: int,
) -> str:
    """Mirror live resident RMS tiles into unused packed-arena padding.

    This diagnostic executes after all attention readers consumed the handoff
    but before its writer pops the CB.  The mirror uses columns beyond the four
    logical KV tiles of the final dK/dV LoRA-gradient sections, so production
    writers never overwrite it and no phase boundary is introduced.
    """

    marker = '''    noc_semaphore_wait_min(go, 2);
    output.pop_front(hidden_tiles);'''
    export = f'''    noc_semaphore_wait_min(go, 2);
    {{
        constexpr uint32_t rms_export_tile_bytes = 2048;
        constexpr uint32_t rms_export_hidden_tiles = {hidden_tiles};
        constexpr uint32_t rms_export_token_tiles = {token_tiles};
        constexpr uint32_t rms_export_kv_tiles = {kv_tiles};
        constexpr uint32_t rms_export_padding_tiles =
            rms_export_hidden_tiles - rms_export_kv_tiles;
        constexpr uint32_t rms_export_kdb_tile_row =
            rms_export_token_tiles + 4 + 2 * rms_export_hidden_tiles;
        const uint32_t rms_export_addr =
            get_common_arg_val<uint32_t>({arena_tensor_arg});
        const auto rms_export_tensor = TensorAccessor(
            TensorAccessorArgs<
                tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                    {arena_tensor_index}, {total_cbs}>(), 0>(),
            rms_export_addr, rms_export_tile_bytes);
        Noc rms_export_noc(1);
        for (uint32_t tile = 0; tile < hidden_tiles; ++tile) {{
            const uint32_t linear = node * hidden_tiles + tile;
            const uint32_t padding_row =
                linear / rms_export_padding_tiles;
            const uint32_t padding_column =
                rms_export_kv_tiles + linear % rms_export_padding_tiles;
            const uint32_t page =
                (rms_export_kdb_tile_row + padding_row) *
                    rms_export_hidden_tiles + padding_column;
            rms_export_noc.async_write(
                CoreLocalMem<uint32_t>(
                    output.get_read_ptr() + tile * rms_export_tile_bytes),
                rms_export_tensor, rms_export_tile_bytes, {{}},
                {{.page_id = page}});
        }}
        rms_export_noc.async_write_barrier();
    }}
    output.pop_front(hidden_tiles);'''
    return _replace_exact_once(
        source, marker, export, label="resident RMS export diagnostic"
    )


def _replace_attention_dy_reads(
    source: str, *, output_hw_id: int, hidden_tiles: int,
    handoff_hw_id: int, worker_base: int = 0,
    distributed_l1: bool = False,
) -> str:
    if not distributed_l1:
        setup = re.compile(
            r"\s*const uint32_t dy_addr = get_common_arg_val<uint32_t>\(0\);\n"
        )
        source, count = setup.subn("\n", source, count=1)
        if count != 1:
            raise ValueError("attention dY address marker changed")
        accessor = re.compile(
            r"\s*const auto dy = TensorAccessor\(\n"
            r"\s*TensorAccessorArgs<\n"
            r"\s*tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
            r"\s*\d+,\s*\d+>\(\),\s*0>\(\), dy_addr, kTileBytes\);\n"
        )
        source, count = accessor.subn("\n", source, count=1)
        if count != 1:
            raise ValueError("attention dY TensorAccessor marker changed")

        read = re.compile(
            r"noc0\.async_read\(\n"
            r"\s*dy,\n"
            r"\s*CoreLocalMem<uint32_t>\((?P<destination>[\s\S]*?)\),\n"
            r"\s*kTileBytes,\s*\{\.page_id = (?P<page>[^}]+)\}, \{\}\);"
        )

        def replace(match: re.Match[str]) -> str:
            destination = match.group("destination").strip()
            page = match.group("page").strip()
            return (
                "noc_async_read(\n"
                f"                    resident_post_rms_handoff::noc_address({page}),\n"
                f"                    {destination},\n"
                "                    kTileBytes);"
            )

        source, count = read.subn(replace, source)
        if count != 3:
            raise ValueError(f"expected three attention dY streams, found {count}")

    reset = '''    // Semaphore addresses may contain state from an unrelated prior program.
    // Reset every flag on every invocation before any producer can publish.
    for (uint32_t semaphore = 0; semaphore < 2 * m_tiles + 1; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }

'''
    if source.count(reset) != 1:
        raise ValueError(
            f"output-projection semaphore reset changed (found {source.count(reset)})"
        )
    source = source.replace(reset, "", 1)

    marker = "    // Wait for every dH producer, then multicast its tile to the same fixed\n"
    ack = r'''    noc_semaphore_inc(
        get_noc_addr(1, 2, get_semaphore(3)), 1);
    noc_async_atomic_barrier();

    // Handoff semaphores 9..12 are not used by the attention owner.  Writer
    // node zero publishes semaphore 10 only after all 50 BRISCs have observed
    // release generation three, so it is now safe to clear attention's 0..8
    // state on this reader core.
    constexpr uint32_t writer_done_sem = 1;
    constexpr uint32_t reader_reset_count_sem = 2;
    constexpr uint32_t attention_start_sem = 3;
    volatile tt_l1_ptr uint32_t* writer_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(writer_done_sem));
    while (*writer_done != 4) {}
    // Semaphores 0..3 still carry the writer acknowledgement, done, reset
    // count and start generations.  Clear only attention-owned 4..8 here.
    // The local BRISC/NCRISC rendezvous below clears 0..3 after the start
    // generation has been observed by both data-movement RISCs.
    for (uint32_t semaphore = 4; semaphore < 2 * m_tiles + 1; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }

    // Count reset completion centrally.  Both NCRISC and BRISC wait for the
    // final semaphore-12 broadcast, so neither RISC can enter attention while
    // another core is still clearing a reused semaphore.
    noc_semaphore_inc(
        get_noc_addr(
            @CONTROLLER_X@, @CONTROLLER_Y@,
            get_semaphore(reader_reset_count_sem)), 1);
    noc_async_atomic_barrier();
    volatile tt_l1_ptr uint32_t* attention_start =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(attention_start_sem));
    if (node == @WORKER_BASE@) {
        volatile tt_l1_ptr uint32_t* reader_reset_count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(reader_reset_count_sem));
        noc_semaphore_wait_min(reader_reset_count, kWorkerNodes);
        noc_semaphore_set(attention_start, 1);
        constexpr uint8_t handoff_x[5] = {1, 2, 3, 4, 5};
        for (uint32_t target = 0; target < kWorkerNodes; ++target) {
            if (target == @WORKER_BASE@) continue;
            noc_async_write(
                get_semaphore(attention_start_sem),
                get_noc_addr(
                    handoff_x[target % 5], 2 + target / 5,
                    get_semaphore(attention_start_sem)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait_min(attention_start, 1);

    // Both data-movement RISCs must retire the temporary handoff semaphore
    // epoch before attention reuses IDs 9..12.  The RMS output backing is
    // dedicated (not aliased to attention) and has already been consumed, so
    // use one local word there as a three-state BRISC/NCRISC rendezvous.
    CircularBuffer handoff(@HANDOFF_HW_ID@);
    volatile tt_l1_ptr uint32_t* handoff_word =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            handoff.get_write_ptr());
    while (*handoff_word != 1) {}
    for (uint32_t semaphore = 0; semaphore < 4; ++semaphore) {
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore)), 0);
    }
    *handoff_word = 2;
    while (*handoff_word != 3) {}

'''
    ack = ack.replace("@HANDOFF_HW_ID@", str(handoff_hw_id))
    physical_x = (1, 2, 3, 4, 5)[worker_base % 5]
    physical_y = 2 + worker_base // 5
    ack = (
        ack.replace("@CONTROLLER_X@", str(physical_x))
        .replace("@CONTROLLER_Y@", str(physical_y))
        .replace("@WORKER_BASE@", str(worker_base))
    )
    ack = ack.replace(
        "get_noc_addr(1, 2, get_semaphore(3))",
        f"get_noc_addr({physical_x}, {physical_y}, get_semaphore(3))",
        1,
    )
    if source.count(marker) != 4:
        raise ValueError(
            "attention phase barrier topology changed "
            f"(found {source.count(marker)})"
        )
    prefix, suffix = source.rsplit(marker, 1)
    source = prefix + ack + marker + suffix
    kernel = "void kernel_main() {"
    if not distributed_l1:
        source = _replace_exact_once(
            source,
            kernel,
            _resident_dy_helper(
                output_hw_id=output_hw_id,
                hidden_tiles=hidden_tiles,
                worker_base=worker_base,
            ) + kernel,
            label="attention reader kernel",
        )
    return source


def _inject_attention_dy_read_export_diagnostic(
    source: str,
    *,
    arena_tensor_arg: int,
    arena_tensor_index: int,
    total_cbs: int,
    hidden_tiles: int,
    token_tiles: int,
    kv_tiles: int,
) -> str:
    """Export the local dH dY stream after its resident NoC fetch."""

    marker = '''        dh_b.push_back(n_tiles);
        dh_dy.push_back(n_tiles);
    }'''
    export = f'''        dh_b.push_back(n_tiles);
        dh_dy.push_back(n_tiles);
        constexpr uint32_t dy_export_tile_bytes = 2048;
        constexpr uint32_t dy_export_hidden_tiles = {hidden_tiles};
        constexpr uint32_t dy_export_token_tiles = {token_tiles};
        constexpr uint32_t dy_export_kv_tiles = {kv_tiles};
        constexpr uint32_t dy_export_padding_tiles =
            dy_export_hidden_tiles - dy_export_kv_tiles;
        constexpr uint32_t dy_export_kdb_tile_row =
            dy_export_token_tiles + 4 + 2 * dy_export_hidden_tiles;
        const uint32_t dy_export_addr =
            get_common_arg_val<uint32_t>({arena_tensor_arg});
        const auto dy_export_tensor = TensorAccessor(
            TensorAccessorArgs<
                tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                    {arena_tensor_index}, {total_cbs}>(), 0>(),
            dy_export_addr, dy_export_tile_bytes);
        for (uint32_t tile = 0; tile < n_tiles; ++tile) {{
            const uint32_t linear = node * n_tiles + tile;
            const uint32_t padding_row = linear / dy_export_padding_tiles;
            const uint32_t padding_column =
                dy_export_kv_tiles + linear % dy_export_padding_tiles;
            const uint32_t page =
                (dy_export_kdb_tile_row + padding_row) *
                    dy_export_hidden_tiles + padding_column;
            noc0.async_write(
                CoreLocalMem<uint32_t>(
                    dh_dy.get_read_ptr() + tile * dy_export_tile_bytes),
                dy_export_tensor, dy_export_tile_bytes, {{}},
                {{.page_id = page}});
        }}
        noc0.async_write_barrier();
    }}'''
    resident_read = "resident_post_rms_handoff::noc_address"
    read_position = source.find(resident_read)
    marker_position = source.find(marker, read_position)
    if read_position < 0 or marker_position < 0:
        raise ValueError("attention dY read export diagnostic marker changed")
    return (
        source[:marker_position]
        + export
        + source[marker_position + len(marker):]
    )


def _replace_first_attention_compute_startup(
    source: str, *, rms_worker_base: int
) -> str:
    """Reconfigure the first attention matmul after RMS hardware startup.

    Metalium requires ``compute_kernel_hw_startup`` exactly once at the start
    of a physical compute kernel.  The accepted attention owner has its own
    startup schedule, but after physical composition RMS has already started
    the engine.  Only the first output-projection phase must therefore use
    the mid-kernel-safe data-format and packer reconfiguration APIs.  Later
    V/K phases keep the previously qualified attention schedule.
    """

    signature = '''void resident_down_phase_one(
    uint32_t projection_n_tiles, uint32_t projection_workers) {'''
    replacement = '''void resident_down_phase_one(
    uint32_t projection_n_tiles, uint32_t projection_workers,
    bool reconfigure_after_resident_phase = false) {'''
    source = _replace_exact_once(
        source, signature, replacement, label="resident-down phase signature"
    )
    function_start = source.index(replacement)
    function_brace = source.index("{", function_start)
    depth = 0
    function_end = -1
    for position in range(function_brace, len(source)):
        depth += (source[position] == "{") - (source[position] == "}")
        if depth == 0:
            function_end = position + 1
            break
    if function_end < 0:
        raise ValueError("unterminated resident-down phase function")
    function_source = source[function_start:function_end]
    startup = '''    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_dy, cb_weight, cb_partial);'''
    reconfigure = '''    if (reconfigure_after_resident_phase) {
        // RMS already performed the one legal public hardware startup for
        // this physical TRISC kernel.  Its packs are drained at the resident
        // handoff, so explicitly begin the next engine epoch here.  A
        // dtype-only reconfiguration is insufficient: RMS and matmul are both
        // BF16 at this boundary, making reconfig_data_format a no-op while the
        // UNPACK tile descriptor still describes RMS's asymmetric source
        // state.  Reconfigure all three engines in the same reverse operand
        // order as compute_kernel_hw_startup<SrcOrder::Reverse>.
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(
            cb_weight, cb_dy)));
        MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
        MATH((llk_math_hw_configure<DST_ACCUM_MODE>(
            cb_weight, cb_dy)));
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_partial)));
        PACK((llk_pack_init<PackMode::Default>(cb_partial)));
        PACK((llk_pack_dest_init<
            DST_ACCUM_MODE, PackMode::Default>(cb_partial)));
        ComputeKernelSentinel::instance()
            .set_srca(cb_weight).set_srcb(cb_dy).set_pack(cb_partial);
    } else {
        // These cores were already started in this exact attention matmul
        // configuration by the composed kernel entry.  A second public
        // startup is illegal, and replaying the RMS-transition sequence here
        // perturbs the accepted reduction/packing state.
    }'''
    function_source = _replace_exact_once(
        function_source,
        startup,
        reconfigure,
        label="resident-down first startup",
    )
    source = source[:function_start] + function_source + source[function_end:]
    first_call = re.compile(
        r'(WAYPOINT\("PB0"\);\n\s*)'
        r'resident_down_phase_one\(([^;]+)\);'
    )
    source, count = first_call.subn(
        lambda match: (
            match.group(1)
            + "resident_down_phase_one("
            + match.group(2)
            + ", (get_absolute_logical_y() * 5 + "
            + f"get_absolute_logical_x()) >= {rms_worker_base});"
        ),
        source,
        count=1,
    )
    if count != 1:
        raise ValueError("attention output-projection phase call changed")
    return source


def _common_map(indices: Sequence[int], combined: list[int]) -> dict[int, int]:
    result = {}
    for local, tensor in enumerate(indices):
        tensor = int(tensor)
        if tensor not in combined:
            combined.append(tensor)
        result[local] = combined.index(tensor)
    return result


def _remap_single_phase_source(
    source: str,
    *,
    cb_map: Mapping[int, int],
    total_cbs: int,
) -> str:
    compile_arg = re.compile(r"get_compile_time_arg_val\((\d+)\)")

    def replace_arg(match: re.Match[str]) -> str:
        old = int(match.group(1))
        if old not in cb_map:
            raise ValueError(f"no descriptor mapping for compile arg {old}")
        return f"get_compile_time_arg_val({cb_map[old]})"

    source = compile_arg.sub(replace_arg, source)
    source = re.sub(
        r"get_tensor_accessor_args_cta_offset<(\s*\d+\s*),\s*\d+>",
        rf"get_tensor_accessor_args_cta_offset<\1, {total_cbs}>",
        source,
    )
    return source


def _add_handoff_waypoints(source: str, *, processor: int) -> str:
    """Add temporary silicon-localization markers to a fused kernel.

    The markers are emitted only when ``TT_FINETUNE_FUSED_RMS_WAYPOINTS=1``
    is present while the operation is captured.  They deliberately surround
    lifecycle waits rather than arithmetic so Watcher identifies the blocked
    RISC without perturbing reduction or BF16 packing order.
    """

    include = '#include "api/debug/waypoint.h"\n'
    if include not in source:
        source = include + source

    if processor == 0:  # NCRISC / reader
        rms_scaler = '''    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW>();'''
        source = _replace_exact_once(
            source,
            rms_scaler,
            '    WAYPOINT("RSCL");\n' + rms_scaler + '\n    WAYPOINT("RSCF");',
            label="RMS reader scaler diagnostic",
        )
        rms_read_barrier = '''    noc0.async_read_barrier();
    x.push_back(hidden_tiles);'''
        source = _replace_exact_once(
            source,
            rms_read_barrier,
            '    WAYPOINT("RNB0");\n'
            '    noc0.async_read_barrier();\n'
            '    WAYPOINT("RNB1");\n'
            '    x.push_back(hidden_tiles);',
            label="RMS reader NoC diagnostic",
        )
        consumed_ack = re.compile(
            r"    noc_semaphore_inc\(\n"
            r"        get_noc_addr\(\d+, \d+, get_semaphore\(3\)\), 1\);"
        )
        source, count = consumed_ack.subn(
            '    WAYPOINT("RA1");\n' + r"\g<0>", source, count=1
        )
        if count != 1:
            raise ValueError("RMS-consumed diagnostic acknowledgement changed")
        replacements = (
            (
                "    resident_layer_phase_post_rms_backward::run();",
                '    WAYPOINT("RR0");\n'
                "    resident_layer_phase_post_rms_backward::run();\n"
                '    WAYPOINT("RR1");',
            ),
            (
                "    noc_semaphore_wait_min(rms_reset_ring, 2);",
                '    WAYPOINT("RR2");\n'
                "    noc_semaphore_wait_min(rms_reset_ring, 2);\n"
                '    WAYPOINT("RR3");',
            ),
            (
                "    noc_semaphore_wait_min(rms_go, 1);",
                '    WAYPOINT("RR4");\n'
                "    noc_semaphore_wait_min(rms_go, 1);\n"
                '    WAYPOINT("RR5");',
            ),
            (
                "    resident_layer_phase_attention_backward::run();",
                '    WAYPOINT("RA0");\n'
                "    resident_layer_phase_attention_backward::run();\n"
                '    WAYPOINT("RAF");',
            ),
            (
                "    while (*writer_done != 4) {}",
                '    WAYPOINT("RD0");\n'
                "    while (*writer_done != 4) {}\n"
                '    WAYPOINT("RD1");',
            ),
            (
                "        noc_semaphore_wait_min(reader_reset_count, kWorkerNodes);",
                '        WAYPOINT("RC0");\n'
                "        noc_semaphore_wait_min(reader_reset_count, kWorkerNodes);\n"
                '        WAYPOINT("RC1");',
            ),
            (
                "    noc_semaphore_wait_min(attention_start, 1);",
                '    WAYPOINT("RS0");\n'
                "    noc_semaphore_wait_min(attention_start, 1);\n"
                '    WAYPOINT("RS1");',
            ),
            (
                "    noc_semaphore_wait_min(output_phase_done, 2);",
                '    WAYPOINT("OE0");\n'
                "    noc_semaphore_wait_min(output_phase_done, 2);\n"
                '    WAYPOINT("OE1");',
            ),
            (
                "    noc_semaphore_wait(entry_start, 1);",
                '    WAYPOINT("QE0");\n'
                "    noc_semaphore_wait(entry_start, 1);\n"
                '    WAYPOINT("QE1");',
            ),
            (
                "    noc_semaphore_wait_min(phase_ready, 2);",
                '    WAYPOINT("PI2");\n'
                "    noc_semaphore_wait_min(phase_ready, 2);\n"
                '    WAYPOINT("PI3");',
            ),
        )
    elif processor == 1:  # TRISC / compute
        replacements = (
            (
                "    resident_layer_phase_post_rms_backward::run();",
                '    WAYPOINT("CR0");\n'
                "    resident_layer_phase_post_rms_backward::run();\n"
                '    WAYPOINT("CR1");',
            ),
            (
                "    resident_layer_phase_attention_backward::run();",
                '    WAYPOINT("CA0");\n'
                "    resident_layer_phase_attention_backward::run();\n"
                '    WAYPOINT("CAF");',
            ),
            (
                "        rms_reduce_scaler.pop_front(1);",
                '        WAYPOINT("SP0");\n'
                "        rms_reduce_scaler.pop_front(1);\n"
                '        WAYPOINT("SP1");',
            ),
            (
                "        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(\n"
                "            cb_weight, cb_dy)));",
                '        WAYPOINT("CX0");\n'
                "        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(\n"
                "            cb_weight, cb_dy)));\n"
                '        WAYPOINT("CX1");',
            ),
            (
                "        MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));",
                '        WAYPOINT("CX2");\n'
                "        MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));\n"
                '        WAYPOINT("CX3");',
            ),
            (
                "        MATH((llk_math_hw_configure<DST_ACCUM_MODE>(\n"
                "            cb_weight, cb_dy)));",
                "        MATH((llk_math_hw_configure<DST_ACCUM_MODE>(\n"
                "            cb_weight, cb_dy)));\n"
                '        WAYPOINT("CX4");',
            ),
            (
                "        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_partial)));",
                '        WAYPOINT("CX5");\n'
                "        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_partial)));\n"
                '        WAYPOINT("CX6");',
            ),
            (
                "        PACK((llk_pack_init<PackMode::Default>(cb_partial)));",
                "        PACK((llk_pack_init<PackMode::Default>(cb_partial)));\n"
                '        WAYPOINT("CX7");',
            ),
            (
                "        PACK((llk_pack_dest_init<\n"
                "            DST_ACCUM_MODE, PackMode::Default>(cb_partial)));",
                "        PACK((llk_pack_dest_init<\n"
                "            DST_ACCUM_MODE, PackMode::Default>(cb_partial)));\n"
                '        WAYPOINT("CX8");',
            ),
        )
    elif processor == 2:  # BRISC / writer
        replacements = (
            (
                "    resident_layer_phase_post_rms_backward::run();",
                '    WAYPOINT("WR0");\n'
                "    resident_layer_phase_post_rms_backward::run();\n"
                '    WAYPOINT("WRF");',
            ),
            (
                "        output.wait_front(block_tiles);",
                '        WAYPOINT("WR1");\n'
                "        output.wait_front(block_tiles);\n"
                '        WAYPOINT("WR2");',
            ),
            (
                "        noc_semaphore_wait_min(consumed, 50);",
                '        WAYPOINT("WR3");\n'
                "        noc_semaphore_wait_min(consumed, 50);\n"
                '        WAYPOINT("WR4");',
            ),
            (
                "        handoff_noc.async_write_barrier();\n"
                "        output.pop_front(block_tiles);",
                "        handoff_noc.async_write_barrier();\n"
                '        WAYPOINT("WR5");\n'
                "        output.pop_front(block_tiles);\n"
                '        WAYPOINT("WR6");',
            ),
            (
                "    noc_semaphore_wait_min(reset_ring, 2);",
                '    WAYPOINT("WRS0");\n'
                "    noc_semaphore_wait_min(reset_ring, 2);\n"
                '    WAYPOINT("WRS1");',
            ),
            (
                "        noc_semaphore_wait_min(writer_ack, 50);",
                '        WAYPOINT("WK0");\n'
                "        noc_semaphore_wait_min(writer_ack, 50);\n"
                '        WAYPOINT("WK1");',
            ),
            (
                "    noc_semaphore_wait_min(attention_start, 1);",
                '    WAYPOINT("WS0");\n'
                "    noc_semaphore_wait_min(attention_start, 1);\n"
                '    WAYPOINT("WS1");',
            ),
            (
                "    resident_layer_phase_attention_backward::run();",
                '    WAYPOINT("WA0");\n'
                "    resident_layer_phase_attention_backward::run();\n"
                '    WAYPOINT("WAF");',
            ),
        )
    else:  # pragma: no cover - topology is checked by the caller
        raise ValueError(f"invalid fused processor index {processor}")

    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(
                f"fused waypoint marker changed for processor {processor}: "
                f"{old!r} occurred {source.count(old)} times"
            )
        source = source.replace(old, new, 1)

    # Semaphore helpers overwrite the Watcher waypoint with the generic
    # ``NSMW`` marker while they spin.  In diagnostic builds only, express the
    # same volatile-L1 predicate directly so the marker immediately before the
    # wait remains visible and identifies the exact blocked protocol edge.
    if processor == 0:
        diagnostic_waits = (
            (
                '    WAYPOINT("RR2");\n'
                "    noc_semaphore_wait_min(rms_reset_ring, 2);",
                '    WAYPOINT("RR2");\n'
                "    while (*rms_reset_ring < 2) {}",
            ),
            (
                '    WAYPOINT("RR4");\n'
                "    noc_semaphore_wait_min(rms_go, 1);",
                '    WAYPOINT("RR4");\n'
                "    while (*rms_go < 1) {}",
            ),
            (
                '        WAYPOINT("RC0");\n'
                "        noc_semaphore_wait_min(reader_reset_count, kWorkerNodes);",
                '        WAYPOINT("RC0");\n'
                "        while (*reader_reset_count < kWorkerNodes) {}",
            ),
            (
                '    WAYPOINT("RS0");\n'
                "    noc_semaphore_wait_min(attention_start, 1);",
                '    WAYPOINT("RS0");\n'
                "    while (*attention_start < 1) {}",
            ),
            (
                '    WAYPOINT("OE0");\n'
                "    noc_semaphore_wait_min(output_phase_done, 2);",
                '    WAYPOINT("OE0");\n'
                "    while (*output_phase_done < 2) {}",
            ),
            (
                '    WAYPOINT("QE0");\n'
                "    noc_semaphore_wait(entry_start, 1);",
                '    WAYPOINT("QE0");\n'
                "    while (*entry_start != 1) {}",
            ),
            (
                '    WAYPOINT("PI2");\n'
                "    noc_semaphore_wait_min(phase_ready, 2);",
                '    WAYPOINT("PI2");\n'
                "    while (*phase_ready < 2) {}",
            ),
        )
        for old, new in diagnostic_waits:
            if source.count(old) != 1:
                raise ValueError(
                    "resident-down diagnostic wait marker changed: "
                    f"{old!r} occurred {source.count(old)} times"
                )
            source = source.replace(old, new, 1)
        phase_ready_wait = "        noc_semaphore_wait_min(phase_ready, 1);"
        if source.count(phase_ready_wait) != 2:
            raise ValueError(
                "QKV phase-init generation-one waits changed: "
                f"found {source.count(phase_ready_wait)}"
            )
        source = source.replace(
            phase_ready_wait,
            '        WAYPOINT("PI1");\n'
            "        while (*phase_ready < 1) {}",
        )
    elif processor == 2:
        diagnostic_waits = (
            (
                '        WAYPOINT("WR3");\n'
                "        noc_semaphore_wait_min(consumed, 50);",
                '        WAYPOINT("WR3");\n'
                "        while (*consumed < 50) {}",
            ),
            (
                '        WAYPOINT("WK0");\n'
                "        noc_semaphore_wait_min(writer_ack, 50);",
                '        WAYPOINT("WK0");\n'
                "        while (*writer_ack < 50) {}",
            ),
            (
                '    WAYPOINT("WS0");\n'
                "    noc_semaphore_wait_min(attention_start, 1);",
                '    WAYPOINT("WS0");\n'
                "    while (*attention_start < 1) {}",
            ),
        )
        for old, new in diagnostic_waits:
            if source.count(old) != 1:
                raise ValueError(
                    "resident-down writer diagnostic wait marker changed: "
                    f"{old!r} occurred {source.count(old)} times"
                )
            source = source.replace(old, new, 1)
        # Locate the output writer by its existing OP0 marker.  Source order
        # is not a lifecycle contract after physical phase composition, so a
        # source-wide "first occurrence" can accidentally instrument V.
        op_marker = source.index('WAYPOINT("OP0")')
        op_start = source.rfind("void run() {", 0, op_marker)
        if op_start < 0:
            raise ValueError("output writer run function changed")
        op_brace = source.index("{", op_start)
        depth = 0
        op_end = -1
        for position in range(op_brace, len(source)):
            depth += (source[position] == "{") - (source[position] == "}")
            if depth == 0:
                op_end = position + 1
                break
        if op_end < 0:
            raise ValueError("unterminated output writer run function")
        op_source = source[op_start:op_end]
        for old, new in (
            (
                "        db.wait_front(1);",
                '        WAYPOINT("ODB0");\n'
                "        while (!db.pages_available_at_front(1)) {}",
            ),
            (
                "        da.wait_front(1);",
                '        WAYPOINT("ODA0");\n'
                "        while (!da.pages_available_at_front(1)) {}",
            ),
            (
                "        final_dx.wait_front(1);",
                '        WAYPOINT("ODX0");\n'
                "        while (!final_dx.pages_available_at_front(1)) {}",
            ),
        ):
            if op_source.count(old) != 1:
                raise ValueError(f"output writer diagnostic marker changed: {old!r}")
            op_source = op_source.replace(old, new, 1)
        source = source[:op_start] + op_source + source[op_end:]

    if processor == 1:
        # Localize the first post-RMS matmul handshake without marking the
        # many later projection phases.  A direct source-wide replacement is
        # unsafe because the accepted attention owner intentionally repeats
        # this reduction template for O/V/K/Q.
        signature = "__attribute__((optimize(\"Os\"), noinline)) void resident_down_phase_one("
        function_start = source.index(signature)
        function_brace = source.index("{", function_start)
        depth = 0
        function_end = -1
        for position in range(function_brace, len(source)):
            depth += (source[position] == "{") - (source[position] == "}")
            if depth == 0:
                function_end = position + 1
                break
        if function_end < 0:
            raise ValueError("unterminated diagnostic resident-down phase")
        function_source = source[function_start:function_end]
        detailed = (
            (
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
                '    WAYPOINT("MI0");',
            ),
            (
                "            dy.wait_front(reduction_tiles_per_phase);\n"
                "            weight.wait_front(reduction_tiles_per_phase);",
                '            WAYPOINT("IW0");\n'
                "            dy.wait_front(reduction_tiles_per_phase);\n"
                '            WAYPOINT("IW1");\n'
                "            weight.wait_front(reduction_tiles_per_phase);\n"
                '            WAYPOINT("IW2");',
            ),
            (
                "            tile_regs_acquire();",
                '            WAYPOINT("AC0");\n'
                "            tile_regs_acquire();\n"
                '            WAYPOINT("AC1");',
            ),
            (
                "                matmul_block(\n"
                "                    cb_dy,\n"
                "                    cb_weight,\n"
                "                    inner,\n"
                "                    inner,\n"
                "                    0,\n"
                "                    false,\n"
                "                    1,\n"
                "                    1,\n"
                "                    reduction_tiles_per_phase);",
                '                WAYPOINT("MM0");\n'
                "                matmul_block(\n"
                "                    cb_dy,\n"
                "                    cb_weight,\n"
                "                    inner,\n"
                "                    inner,\n"
                "                    0,\n"
                "                    false,\n"
                "                    1,\n"
                "                    1,\n"
                "                    reduction_tiles_per_phase);\n"
                '                WAYPOINT("MM1");',
            ),
            (
                "            tile_regs_commit();",
                "            tile_regs_commit();\n"
                '            WAYPOINT("CM1");',
            ),
            (
                "            tile_regs_wait();",
                '            WAYPOINT("TW0");\n'
                "            tile_regs_wait();\n"
                '            WAYPOINT("TW1");',
            ),
            (
                "            PACK((pack_reconfig_data_format(target_cb)));\n"
                "            pack_tile_block(0, target_cb, 1);",
                '            WAYPOINT("PK0");\n'
                "            PACK((pack_reconfig_data_format(target_cb)));\n"
                "            pack_tile_block(0, target_cb, 1);\n"
                '            WAYPOINT("PK1");',
            ),
            (
                "            tile_regs_release();",
                "            tile_regs_release();\n"
                '            WAYPOINT("RL1");',
            ),
        )
        for old, new in detailed:
            if function_source.count(old) != 1:
                raise ValueError(
                    "first attention matmul diagnostic marker changed: "
                    f"{old!r} occurred {function_source.count(old)} times"
                )
            function_source = function_source.replace(old, new, 1)
        local_brace = function_source.index("{")
        function_source = (
            function_source[: local_brace + 1]
            + "\n    const uint32_t diagnostic_projection_epoch = "
            "diagnostic_shared_projection_epoch++;"
            + function_source[local_brace + 1 :]
        )
        function_source = function_source.replace(
            '    WAYPOINT("MI0");',
            '    if (diagnostic_projection_epoch == 0) WAYPOINT("OMI0");\n'
            '    else if (diagnostic_projection_epoch == 1) WAYPOINT("VMI0");\n'
            '    else WAYPOINT("KMI0");',
            1,
        ).replace(
            '            WAYPOINT("IW0");',
            '            if (diagnostic_projection_epoch == 0) WAYPOINT("OIW0");\n'
            '            else if (diagnostic_projection_epoch == 1) WAYPOINT("VIW0");\n'
            '            else WAYPOINT("KIW0");',
            1,
        ).replace(
            '            WAYPOINT("IW1");',
            '            if (diagnostic_projection_epoch == 0) WAYPOINT("OIW1");\n'
            '            else if (diagnostic_projection_epoch == 1) WAYPOINT("VIW1");\n'
            '            else WAYPOINT("KIW1");',
            1,
        ).replace(
            '            WAYPOINT("IW2");',
            '            if (diagnostic_projection_epoch == 0) WAYPOINT("OIW2");\n'
            '            else if (diagnostic_projection_epoch == 1) WAYPOINT("VIW2");\n'
            '            else WAYPOINT("KIW2");',
            1,
        )
        source = (
            source[:function_start]
            + "static uint32_t diagnostic_shared_projection_epoch = 0;\n"
            + function_source
            + source[function_end:]
        )

        # Q owns a cloned frozen-base reduction because it is the fourth
        # projection epoch.  Give that clone distinct diagnostic markers so a
        # stale CB lifecycle cannot be mistaken for the shared O/V/K body.
        q_signature = (
            '__attribute__((optimize("Os"), noinline)) void '
            'q_resident_down_phase_one('
        )
        q_start = source.index(q_signature)
        q_brace = source.index("{", q_start)
        depth = 0
        q_end = -1
        for position in range(q_brace, len(source)):
            depth += (source[position] == "{") - (source[position] == "}")
            if depth == 0:
                q_end = position + 1
                break
        if q_end < 0:
            raise ValueError("unterminated diagnostic Q reduction")
        q_source = source[q_start:q_end]
        q_local_brace = q_source.index("{")
        q_source = (
            q_source[: q_local_brace + 1]
            + '\n    WAYPOINT("QB0");'
            + q_source[q_local_brace + 1 :]
        )
        q_detailed = (
            (
                "            dy.wait_front(reduction_tiles_per_phase);\n"
                "            weight.wait_front(reduction_tiles_per_phase);",
                '            WAYPOINT("QW0");\n'
                "            dy.wait_front(reduction_tiles_per_phase);\n"
                "            weight.wait_front(reduction_tiles_per_phase);\n"
                '            WAYPOINT("QW1");',
            ),
            (
                "            tile_regs_acquire();",
                '            WAYPOINT("QA0");\n'
                "            tile_regs_acquire();\n"
                '            WAYPOINT("QA1");',
            ),
            (
                "                partial.wait_front(1);",
                '                WAYPOINT("QP0");\n'
                "                partial.wait_front(1);\n"
                '                WAYPOINT("QP1");',
            ),
        )
        for old, new in q_detailed:
            if q_source.count(old) != 1:
                raise ValueError(
                    "Q diagnostic marker changed: "
                    f"{old!r} occurred {q_source.count(old)} times"
                )
            q_source = q_source.replace(old, new, 1)
        source = source[:q_start] + q_source + source[q_end:]
        # The fused owner has historically reached the first output
        # projection tail and then parked before its resident dH window became
        # visible.  Keep the diagnostic markers inside the generic O tail
        # only; V/K/Q have independent helpers with similar source text.
        tail_signature = (
            '__attribute__((optimize("Os"), noinline)) '
            'void resident_down_tail() {'
        )
        tail_start = source.index(tail_signature)
        tail_brace = source.index("{", tail_start)
        depth = 0
        tail_end = -1
        for position in range(tail_brace, len(source)):
            depth += (source[position] == "{") - (source[position] == "}")
            if depth == 0:
                tail_end = position + 1
                break
        if tail_end < 0:
            raise ValueError("unterminated diagnostic resident-down tail")
        tail_source = source[tail_start:tail_end]
        tail_markers = (
            (
                "    resident_dh.wait_front(m_tiles);",
                '    WAYPOINT("CD0");\n'
                "    resident_dh.wait_front(m_tiles);\n"
                '    WAYPOINT("CD1");',
            ),
            (
                "        run_da(\n"
                "            cb_dh, cb_x, cb_transposed_x, cb_da_partial,\n"
                "            cb_da_pretranspose, cb_da, m_tiles,\n"
                "            da_k_tiles_per_node, k_tiles, node);",
                '        WAYPOINT("DA0");\n'
                "        run_da(\n"
                "            cb_dh, cb_x, cb_transposed_x, cb_da_partial,\n"
                "            cb_da_pretranspose, cb_da, m_tiles,\n"
                "            da_k_tiles_per_node, k_tiles, node);\n"
                '        WAYPOINT("DA1");',
            ),
            (
                "        run_resident_down_lora_dx(\n"
                "            cb_dh, cb_a, cb_lora_dx,\n"
                "            jobs_per_node, total_jobs, k_tiles, node);",
                '        WAYPOINT("LD0");\n'
                "        run_resident_down_lora_dx(\n"
                "            cb_dh, cb_a, cb_lora_dx,\n"
                "            jobs_per_node, total_jobs, k_tiles, node);\n"
                '        WAYPOINT("LD1");',
            ),
            (
                "        run_bf16_add(\n"
                "            cb_base, cb_lora_dx, cb_final_dx,\n"
                "            jobs_per_node, total_jobs, node);",
                '        WAYPOINT("AD0");\n'
                "        run_bf16_add(\n"
                "            cb_base, cb_lora_dx, cb_final_dx,\n"
                "            jobs_per_node, total_jobs, node);\n"
                '        WAYPOINT("AD1");',
            ),
        )
        for old, new in tail_markers:
            if tail_source.count(old) != 1:
                raise ValueError(
                    "resident-down tail diagnostic marker changed: "
                    f"{old!r} occurred {tail_source.count(old)} times"
                )
            tail_source = tail_source.replace(old, new, 1)
        source = source[:tail_start] + tail_source + source[tail_end:]
    elif processor == 0:
        # Mark the generic O reader's handoff only.  The repeated dH template
        # also appears in V/K/Q, so select the final occurrence, which is the
        # O projection that executes first in the composed attention owner.
        reader_markers = (
            (
                "        dh_output.wait_front(1);",
                '        WAYPOINT("RD0");\n'
                "        dh_output.wait_front(1);\n"
                '        WAYPOINT("RD1");',
            ),
            (
                "        noc_semaphore_wait(go, 1);",
                '        WAYPOINT("RG0");\n'
                "        noc_semaphore_wait(go, 1);\n"
                '        WAYPOINT("RG1");',
            ),
            (
                "        noc_semaphore_wait(ready, 1);",
                '        WAYPOINT("RY0");\n'
                "        noc_semaphore_wait(ready, 1);\n"
                '        WAYPOINT("RY1");',
            ),
            (
                "    dh_resident.push_back(m_tiles);",
                "    dh_resident.push_back(m_tiles);\n"
                '    WAYPOINT("RP1");',
            ),
            (
                "    noc_semaphore_wait(phase_done, 1);",
                '    WAYPOINT("RF0");\n'
                "    noc_semaphore_wait(phase_done, 1);\n"
                '    WAYPOINT("RF1");',
            ),
        )
        for old, new in reader_markers:
            if old not in source:
                raise ValueError(
                    f"resident-down reader diagnostic marker changed: {old!r}"
                )
            prefix, suffix = source.rsplit(old, 1)
            source = prefix + new + suffix
        for old, new in (
            (
                '        WAYPOINT("RG0");\n'
                "        noc_semaphore_wait(go, 1);",
                '        WAYPOINT("RG0");\n'
                "        while (*go != 1) {}",
            ),
            (
                '        WAYPOINT("RY0");\n'
                "        noc_semaphore_wait(ready, 1);",
                '        WAYPOINT("RY0");\n'
                "        while (*ready != 1) {}",
            ),
            (
                '    WAYPOINT("RF0");\n'
                "    noc_semaphore_wait(phase_done, 1);",
                '    WAYPOINT("RF0");\n'
                "    while (*phase_done != 1) {}",
            ),
        ):
            if source.count(old) != 1:
                raise ValueError(
                    "resident-down diagnostic wait marker changed: "
                    f"{old!r} occurred {source.count(old)} times"
                )
            source = source.replace(old, new, 1)
    return source


def fuse_post_rms_backward_into_attention_program(
    rms_program: Mapping[str, object],
    attention_program: Mapping[str, object],
    *,
    hidden_tiles: int,
    kv_tiles: int | None = None,
    handoff_tensor_index: int = 27,
) -> dict:
    """Return one physical artifact-v3 RMS->attention backward program."""

    if int(rms_program.get("format_version", 0)) != 1:
        raise ValueError("RMS child must be the accepted compact v1 artifact")
    if int(attention_program.get("format_version", 0)) != 2:
        raise ValueError("attention child must be the accepted segmented v2 artifact")
    configs, rms_cb_map, attention_cb_map, output_hw, projection_count = (
        _projection_descriptor_layout(rms_program, attention_program)
    )
    total_cbs = len(configs)

    rms_kernels = list(rms_program["kernels"])
    attention_kernels = list(attention_program["kernels"])
    if len(rms_kernels) != 3 or len(attention_kernels) != 6:
        raise ValueError("resident RMS/attention kernel topology changed")

    fused_kernels = []
    export_rms_diagnostic = (
        os.environ.get("TT_FINETUNE_FUSED_RMS_EXPORT") == "1"
    )
    export_dy_read_diagnostic = (
        os.environ.get("TT_FINETUNE_FUSED_DY_READ_EXPORT") == "1"
    )
    if (export_rms_diagnostic or export_dy_read_diagnostic) and kv_tiles is None:
        raise ValueError("diagnostic resident export requires kv_tiles")
    for index, (rms_kernel, attention_kernel) in enumerate(
        zip(rms_kernels, attention_kernels[:3])
    ):
        attention_only_source: str | None = None
        combined_tensors: list[int] = []
        rms_common = _common_map(rms_kernel["tensor_indices"], combined_tensors)
        attention_common = _common_map(
            attention_kernel["tensor_indices"], combined_tensors
        )
        rms_source = str(rms_kernel["cpp_source"])
        attention_source = str(attention_kernel["cpp_source"])
        rms_source = rms_source.replace(
            "constexpr uint32_t total_cbs = 21;",
            f"constexpr uint32_t total_cbs = {total_cbs};",
        )
        token_match = re.search(
            r"constexpr uint32_t token_tiles = (\d+);", rms_source
        )
        if token_match is None:
            raise ValueError("expanded RMS token constant changed")
        rms_token_tiles = int(token_match.group(1))
        rms_worker_base = 50 - rms_token_tiles
        if index == 0:
            rms_source = _remove_rms_reader_release(rms_source)
            rms_source = _move_rms_phase_to_tail(
                rms_source, token_tiles=rms_token_tiles
            )
            rms_source = _prepare_tail_rms_reader_controller(
                rms_source, worker_base=rms_worker_base
            )
            attention_source = _replace_attention_dy_reads(
                attention_source,
                output_hw_id=output_hw,
                hidden_tiles=hidden_tiles,
                handoff_hw_id=output_hw,
                worker_base=rms_worker_base,
                distributed_l1=True,
            )
            if export_dy_read_diagnostic:
                if 22 not in combined_tensors:
                    combined_tensors.append(22)
                arena_tensor_arg = max(attention_common, default=-1) + 1
                attention_common[arena_tensor_arg] = combined_tensors.index(22)
                attention_source = _inject_attention_dy_read_export_diagnostic(
                    attention_source,
                    arena_tensor_arg=arena_tensor_arg,
                    arena_tensor_index=22,
                    total_cbs=total_cbs,
                    hidden_tiles=hidden_tiles,
                    token_tiles=rms_token_tiles,
                    kv_tiles=int(kv_tiles),
                )
        elif index == 1:
            rms_source = _remove_rms_compute_release(rms_source)
            rms_source = _move_rms_phase_to_tail(
                rms_source, token_tiles=rms_token_tiles
            )
            rms_source = _remove_rms_compute_startup(rms_source)
            # The first 46 projection cores never execute RMS arithmetic.
            # Preserve their accepted attention compute source and static
            # unpack configuration in separate, disjoint compute kernels.
            attention_only_source = compose_resident_kernel_phases(
                (
                    ResidentKernelPhase(
                        name="attention_backward",
                        source=attention_source,
                        cb_arguments=attention_cb_map,
                        tensor_indices={
                            int(tensor): int(tensor)
                            for tensor in attention_kernel["tensor_indices"]
                        },
                        common_arguments=attention_common,
                    ),
                ),
                total_cbs=total_cbs,
            )
            attention_source = _replace_first_attention_compute_startup(
                attention_source, rms_worker_base=rms_worker_base
            )
        elif index == 2:
            # The RMS writer materializes its streamed row into the same
            # compiler-owned distributed L1 tensor consumed as attention dY.
            # Give the RMS phase a synthetic common-argument position for ID27;
            # the composer remaps it into this kernel's compact ABI.
            handoff_tensor_arg = max(rms_common, default=-1) + 1
            if handoff_tensor_index not in combined_tensors:
                combined_tensors.append(handoff_tensor_index)
            rms_common[handoff_tensor_arg] = combined_tensors.index(
                handoff_tensor_index
            )
            rms_source = _replace_rms_writer(
                rms_source,
                handoff_hw_id=output_hw,
                handoff_tensor_arg=handoff_tensor_arg,
                handoff_tensor_index=handoff_tensor_index,
                total_cbs=total_cbs,
                worker_base=rms_worker_base,
            )

        if index == 0:
            phase_barrier = _rms_reader_handoff_barrier(
                hidden_tiles=hidden_tiles,
                output_hw_id=output_hw,
                token_tiles=rms_token_tiles,
                worker_base=rms_worker_base,
            )
        elif index == 1:
            phase_barrier = _compute_handoff_retired_barrier(
                product_cb_arg=rms_cb_map[5],
                scaler_cb_arg=rms_cb_map[6],
                token_tiles=rms_token_tiles,
                worker_base=rms_worker_base,
            )
        else:
            phase_barrier = _writer_handoff_retired_barrier()

        source = compose_resident_kernel_phases(
            (
                ResidentKernelPhase(
                    name="post_rms_backward",
                    source=rms_source,
                    cb_arguments=rms_cb_map,
                    tensor_indices={
                        int(tensor): int(tensor)
                        for tensor in (
                            list(rms_kernel["tensor_indices"])
                            + ([handoff_tensor_index] if index == 2 else [])
                        )
                    },
                    common_arguments=rms_common,
                ),
                ResidentKernelPhase(
                    name="attention_backward",
                    source=attention_source,
                    cb_arguments=attention_cb_map,
                    tensor_indices={
                        int(tensor): int(tensor)
                        for tensor in (
                            list(attention_kernel["tensor_indices"])
                            + ([22] if index == 0 and export_dy_read_diagnostic else [])
                        )
                    },
                    common_arguments=attention_common,
                ),
            ),
            total_cbs=total_cbs,
            barriers_after={"post_rms_backward": phase_barrier},
        )
        if (
            index == 2
            and export_rms_diagnostic
        ):
            try:
                arena_tensor_arg = combined_tensors.index(22)
            except ValueError as exc:
                raise ValueError(
                    "attention writer no longer exposes packed arena tensor 22"
                ) from exc
            source = _inject_resident_rms_export_diagnostic(
                source,
                arena_tensor_arg=arena_tensor_arg,
                arena_tensor_index=22,
                total_cbs=total_cbs,
                hidden_tiles=hidden_tiles,
                token_tiles=rms_token_tiles,
                kv_tiles=int(kv_tiles),
            )
        if index == 1:
            source = _prepend_composed_compute_startup(
                source,
                dy_cb_arg=attention_cb_map[0],
                weight_cb_arg=attention_cb_map[1],
                partial_cb_arg=attention_cb_map[2],
                rms_x_cb_arg=rms_cb_map[0],
                rms_scaler_cb_arg=rms_cb_map[6],
                rms_output_cb_arg=rms_cb_map[18],
                rms_worker_base=rms_worker_base,
            )
        if os.environ.get("TT_FINETUNE_FUSED_RMS_WAYPOINTS") == "1":
            source = _add_handoff_waypoints(source, processor=index)
        kernel = copy.deepcopy(attention_kernel)
        kernel["cpp_source"] = source
        kernel["tensor_indices"] = combined_tensors
        if index == 1:
            modes = ["Default"] * 64
            rms_modes = rms_kernel["kernel_config"].get(
                "unpack_to_dest_modes", []
            )
            for old_cb, mode in enumerate(rms_modes):
                if mode != "Default":
                    hw = int(configs[rms_cb_map[old_cb]]["buffer_index"])
                    modes[hw] = str(mode)
            kernel["kernel_config"]["unpack_to_dest_modes"] = modes
            if attention_only_source is None:
                raise AssertionError("missing accepted attention compute source")
            # Tail workers 46..49 are logical cores (1..4, 9).  They own the
            # RMS arithmetic and only participate in attention's count/fence
            # tail, where no projection math jobs are assigned.  The other
            # two rectangular ranges retain the exact accepted compute body
            # and kernel configuration.
            kernel["core_range"] = {"start": [1, 9], "end": [4, 9]}
            fused_kernels.append(kernel)
            for core_range in (
                {"start": [0, 0], "end": [4, 8]},
                {"start": [0, 9], "end": [0, 9]},
            ):
                accepted_kernel = copy.deepcopy(attention_kernel)
                accepted_kernel["cpp_source"] = attention_only_source
                accepted_kernel["tensor_indices"] = combined_tensors
                accepted_kernel["core_range"] = core_range
                accepted_modes = list(
                    accepted_kernel["kernel_config"].get(
                        "unpack_to_dest_modes", []
                    )
                )
                accepted_kernel["kernel_config"][
                    "unpack_to_dest_modes"
                ] = accepted_modes + ["Default"] * (64 - len(accepted_modes))
                fused_kernels.append(accepted_kernel)
        else:
            fused_kernels.append(kernel)

    for kernel in attention_kernels[3:]:
        remapped = copy.deepcopy(kernel)
        remapped["cpp_source"] = _remap_single_phase_source(
            str(remapped["cpp_source"]),
            cb_map=attention_cb_map,
            total_cbs=total_cbs,
        )
        fused_kernels.append(remapped)

    semaphores = copy.deepcopy(attention_program.get("semaphores", []))
    if {int(item["id"]) for item in semaphores} != set(range(16)):
        raise ValueError("accepted attention must own semaphore IDs 0..15")

    return {
        "format_version": 3,
        "kernels": fused_kernels,
        "core_range": copy.deepcopy(attention_program["core_range"]),
        "cb_configs": configs,
        "num_tensors": int(attention_program["num_tensors"]),
        "num_pipe_sync_semaphores": int(
            attention_program["num_pipe_sync_semaphores"]
        ),
        "semaphores": semaphores,
        "resident_handoff": {
            "kind": "post_rms_dy_cb_noc",
            "projection_cb_descriptors": projection_count,
            "output_cb_id": output_hw,
            "producer_cores": 4,
            "consumer_cores": 50,
            "intermediate_dram_bytes": 0,
        },
    }
