"""One-program exact V -> K -> Q LoRA projection backward.

This is the projection half of the full resident attention owner.  The first
qualification form accepts dV/dK/dQ as ordinary operands so the sequential
projection schedule and its five BF16 dHidden edges can be proven in
isolation.  The output-attention owner replaces those three reads with the
already-qualified L1/NoC attention handoff; the compute and writer sources
are shared unchanged.

Only final tensors leave the program: dHidden and Q/K/V dA/dB.  V and K
input-gradient accumulators remain in CB24 across phases.
"""

from __future__ import annotations

import hashlib
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .native_frozen_lora_autograd import exact_base_dx_k_phase
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID
from .tt_lang_frozen_linear_lora_backward import (
    _shift_compile_time_cb_indices,
)
from .tt_lang_lora import TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR
from .tt_lang_output_attention_resident_backward import (
    _projection_cb_shapes,
)


ttl = None
tt_lang_operation = None

_CBS = 27
_GRID = {"start": [0, 0], "end": [4, 9]}
_ALL_GRID = {"start": [0, 0], "end": [10, 9]}
_WORKERS = 50


def _parallel_qkv_sequential_cb_capacities(
    *, m_tiles: int, hidden_tiles: int, kv_tiles: int,
    jobs_per_node: int,
) -> dict[int, int]:
    """Tile capacities for FIFOs reused by the K and Q phases.

    Draining a circular buffer does not rewind its physical write pointer.
    A later multi-tile ``reserve_back`` therefore needs capacity for every
    prior publication in the program, not merely the maximum live window.
    """

    return {
        4: kv_tiles + hidden_tiles,
        5: (
            kv_tiles + jobs_per_node
            + hidden_tiles + jobs_per_node
        ),
        # Scaled dY and LoRA dX share CB6 in both phases.
        6: (
            kv_tiles + jobs_per_node
            + hidden_tiles + jobs_per_node
        ),
        # Cores 0..M-1 publish one K dH tile and one Q dH tile through CB8
        # before Q reuses it as the exact-add scratch.  Keep that scratch
        # window contiguous for every job stride (Qwen uses three).
        8: 2 + jobs_per_node,
        9: 4 * m_tiles,
        10: 2 * m_tiles,
        # K's member-zero exact-add tail advances CB11 after dB scaling.
        11: 2 * m_tiles + jobs_per_node,
    }


def qkv_resident_backward_arena_shape(
    *, tokens: int, hidden_features: int, kv_features: int,
) -> tuple[int, int]:
    """Return the final-only QKV projection arena shape."""

    return (
        tokens + 3 * TILE_SIZE + hidden_features + 2 * kv_features,
        hidden_features,
    )


def qkv_resident_backward_support_error(
    *, tokens: int, hidden_features: int, kv_features: int,
    physical_rank: int,
) -> str | None:
    if any(
        value <= 0 or value % TILE_SIZE
        for value in (tokens, hidden_features, kv_features, physical_rank)
    ):
        return "QKV resident dimensions must be positive and tile aligned"
    if physical_rank != TILE_SIZE:
        return "QKV resident backward requires physical rank 32"
    if tokens // TILE_SIZE > 8:
        return "QKV resident semaphore schedule requires M <= 8 tiles"
    q_phase = exact_base_dx_k_phase(hidden_features, hidden_features)
    kv_phase = exact_base_dx_k_phase(hidden_features, kv_features)
    if q_phase is None or kv_phase is None:
        return "no exact frozen-base reduction schedule for Q/K/V shapes"
    if q_phase != kv_phase:
        return "first QKV owner requires one shared exact reduction phase"
    if (hidden_features // TILE_SIZE) % 2:
        return "exact LoRA dH requires an even Q output tile count"
    if (kv_features // TILE_SIZE) % 2:
        return "exact LoRA dH requires an even K/V output tile count"
    return None


def _replace_all(source: str, replacements: dict[str, str]) -> str:
    for marker, value in replacements.items():
        source = source.replace(marker, value)
    unresolved = re.findall(r"@[A-Z0-9_]+@", source)
    if unresolved:
        raise RuntimeError(f"unexpanded QKV resident markers: {unresolved}")
    return source


def _resident_attention_dy_reader_source(
    source: str,
    *,
    sequence_tiles: int,
    hidden_tiles: int,
    dimension_tiles: int,
    query_heads: int,
    kv_heads: int,
    semaphore_base: int = 0,
) -> str:
    """Replace Q/K/V DRAM dY reads with the accepted attention L1 handoff.

    The serial 50-core projection reader is deliberately retained for this
    first whole-attention program: its arithmetic schedule is already exact.
    Only the source address changes.  Attention owners publish the three CB
    base addresses through semaphores 11..13 on projection node zero; every
    projection consumer then pulls the exact tile directly over NoC.

    Projection-internal semaphores are shifted above the attention protocol
    so both phases can coexist in one TT-Metal program.
    """

    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("invalid resident attention head geometry")
    if query_heads * dimension_tiles != hidden_tiles:
        raise ValueError("query heads do not cover the hidden tile width")
    if min(sequence_tiles, hidden_tiles, dimension_tiles) <= 0:
        raise ValueError("resident attention tile dimensions must be positive")

    # Move only the pre-existing projection protocol.  The handoff prologue
    # inserted below intentionally continues to use attention semaphores 0..4.
    if semaphore_base:
        source = source.replace(
            "get_semaphore(", f"get_semaphore({semaphore_base} + "
        )

    phase_geometry = {
        "q": (11, hidden_tiles, True),
        "k": (12, kv_heads * dimension_tiles, False),
        "v": (13, kv_heads * dimension_tiles, False),
    }
    for prefix, (address_semaphore, output_tiles, is_query) in (
        phase_geometry.items()
    ):
        address_helper = f"""
namespace {{
constexpr uint8_t {prefix}_kAttentionPhysicalX[11] = {{
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13}};
FORCE_INLINE uint64_t {prefix}_resident_dy_noc_addr(uint32_t page_id) {{
    constexpr uint32_t sequence_tiles = {sequence_tiles};
    constexpr uint32_t output_tiles = {output_tiles};
    constexpr uint32_t dimension_tiles = {dimension_tiles};
    constexpr uint32_t query_heads = {query_heads};
    constexpr uint32_t kv_heads = {kv_heads};
    constexpr uint32_t group = query_heads / kv_heads;
    const uint32_t token = page_id / output_tiles;
    const uint32_t output = page_id - token * output_tiles;
    const uint32_t owner_head = output / dimension_tiles;
    const uint32_t source_node =
        {"owner_head * sequence_tiles + token" if is_query else "owner_head * group * sequence_tiles + token"};
    const uint32_t source_tile = output - owner_head * dimension_tiles;
    const uint32_t base =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore({address_semaphore}));
    const uint32_t logical_x = 5 + source_node % 6;
    return get_noc_addr(
        {prefix}_kAttentionPhysicalX[logical_x],
        {prefix}_physical_y(source_node / 6),
        base + source_tile * {prefix}_kTileBytes);
}}
}}  // namespace

"""
        marker = f"FORCE_INLINE void {prefix}_reader_phase() {{"
        if source.count(marker) != 1:
            raise RuntimeError(f"{prefix} projection reader marker changed")
        source = source.replace(marker, address_helper + marker, 1)

        dy_address = re.compile(
            rf"\n    const uint32_t dy_addr = .*?;\n",
        )
        phase_start = source.index(marker)
        next_starts = [
            source.find(f"FORCE_INLINE void {other}_reader_phase()", phase_start + 1)
            for other in ("v", "k", "q")
        ]
        next_starts = [position for position in next_starts if position >= 0]
        phase_end = min(next_starts) if next_starts else source.index(
            "void kernel_main()", phase_start
        )
        phase = source[phase_start:phase_end]
        phase, address_count = dy_address.subn("\n", phase, count=1)
        dy_accessor = re.compile(
            rf"\n    const auto dy = TensorAccessor\(.*?"
            rf"dy_addr, {prefix}_kTileBytes\);\n",
            re.DOTALL,
        )
        phase, accessor_count = dy_accessor.subn("\n", phase, count=1)
        dy_read = re.compile(
            rf"noc0\.async_read\(\s*dy,\s*"
            rf"CoreLocalMem<uint32_t>\((.*?)\),\s*"
            rf"{prefix}_kTileBytes,\s*"
            rf"\{{\.page_id = (.*?)\}}, \{{\}}\);",
            re.DOTALL,
        )

        def replace_dy_read(match: re.Match[str]) -> str:
            destination = match.group(1).strip()
            page_id = match.group(2).strip()
            return (
                "noc_async_read(\n"
                f"                    {prefix}_resident_dy_noc_addr({page_id}),\n"
                f"                    {destination},\n"
                f"                    {prefix}_kTileBytes);"
            )

        phase, read_count = dy_read.subn(replace_dy_read, phase)
        if (address_count, accessor_count, read_count) != (1, 1, 3):
            raise RuntimeError(
                f"{prefix} resident dY rewrite changed: "
                f"address={address_count}, accessor={accessor_count}, "
                f"reads={read_count}"
            )
        source = source[:phase_start] + phase + source[phase_end:]

    old_main = """void kernel_main() {
    v_reader_phase();
    k_reader_phase();
    q_reader_phase();
}"""
    handoff_main = f"""void kernel_main() {{
    constexpr uint32_t kProjectionWorkers = 50;
    constexpr uint32_t kActiveAttentionNodes =
        {query_heads} * {sequence_tiles};
    constexpr uint8_t kProjectionPhysicalX[5] = {{1, 2, 3, 4, 5}};
    const uint32_t node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();
    volatile tt_l1_ptr uint32_t* attention_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* handoff_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    if (node == 0) {{
        noc_semaphore_wait_min(attention_ready, kActiveAttentionNodes);
        noc_semaphore_set(attention_ready, 0);
        for (uint32_t target = 1; target < kProjectionWorkers; ++target) {{
            for (uint32_t word = 11; word <= 13; ++word) {{
                noc_async_write(
                    get_semaphore(word),
                    get_noc_addr(
                        kProjectionPhysicalX[target % 5],
                        2 + target / 5, get_semaphore(word)),
                    sizeof(uint32_t));
            }}
        }}
        noc_async_write_barrier();
        noc_semaphore_set(handoff_go, 1);
        for (uint32_t target = 1; target < kProjectionWorkers; ++target) {{
            noc_async_write(
                get_semaphore(4),
                get_noc_addr(
                    kProjectionPhysicalX[target % 5],
                    2 + target / 5, get_semaphore(4)),
                sizeof(uint32_t));
        }}
        noc_async_write_barrier();
    }}
    noc_semaphore_wait(handoff_go, 1);
    noc_semaphore_set(handoff_go, 0);
    v_reader_phase();
    k_reader_phase();
    q_reader_phase();
}}"""
    if source.count(old_main) != 1:
        raise RuntimeError("QKV projection reader main changed")
    return source.replace(old_main, handoff_main, 1)


def _resident_attention_qkv_writer_source(
    source: str, *, semaphore_base: int = 0,
    gradient_row_offset: int = 0,
) -> str:
    """Shift projection barriers and release retained attention gradients."""

    if semaphore_base:
        source = source.replace(
            "get_semaphore(", f"get_semaphore({semaphore_base} + "
        )
    if gradient_row_offset:
        # The final dHidden remains at row zero, reusing the output
        # projection's intentionally unwritten dO region. Only the six LoRA
        # gradient regions move past o_dA/o_dB.
        page = re.compile(r"\{\.page_id = (?!job\})([^}]*)\}")
        source, count = page.subn(
            lambda match: (
                "{.page_id = "
                f"{gradient_row_offset} * arena_cols + "
                f"({match.group(1).strip()})"
                "}"
            ),
            source,
        )
        if count != 6:
            raise RuntimeError(
                f"QKV final writer page contract changed: {count} pages"
            )
    old_main = """void kernel_main() {
    v_writer_phase();
    k_writer_phase();
    q_writer_phase();
}"""
    new_main = """void kernel_main() {
    v_writer_phase();
    k_writer_phase();
    q_writer_phase();
    // Release the attention owners only after every final Q/K/V projection
    // output on this core has reached the output arena.
    noc_semaphore_inc(get_noc_addr(6, 2, get_semaphore(10)), 1);
    noc_async_atomic_barrier();
}"""
    if source.count(old_main) != 1:
        raise RuntimeError("QKV projection writer main changed")
    return source.replace(old_main, new_main, 1)


def _merge_phased_kernel_source(base: str, extension: str) -> str:
    """Run two already-qualified phase bodies on one physical RISC kernel.

    TT-Metal cannot schedule two reader (or two writer/compute) kernels on the
    same core at once.  A whole-block owner therefore merges their helper
    functions and concatenates their ``kernel_main`` bodies instead of merely
    placing two kernel descriptors on the same core range.
    """

    marker = "void kernel_main() {"

    def split(source: str) -> tuple[str, str]:
        if source.count(marker) != 1 or not source.rstrip().endswith("}"):
            raise RuntimeError("phased kernel main contract changed")
        start = source.index(marker)
        body_start = start + len(marker)
        end = source.rstrip().rfind("}")
        return source[:start], source[body_start:end]

    base_prelude, base_body = split(base)
    extension_prelude, extension_body = split(extension)
    return (
        base_prelude.rstrip()
        + "\n\n"
        + extension_prelude.rstrip()
        + "\n\nvoid kernel_main() {\n    {"
        + base_body.rstrip()
        + "\n    }\n    {"
        + extension_body.rstrip()
        + "\n    }\n}\n"
    )


def _noinline_functions(
    source: str,
    names: tuple[str, ...],
    *,
    optimize: str | None = None,
) -> str:
    """Keep a large multi-phase Tensix owner within the config-text budget.

    The accepted projection sources use ``FORCE_INLINE`` because their
    standalone kernels have only one call site.  A whole-attention owner has
    four projection call sites, so retaining that policy clones the complete
    matmul/transpose bodies and can exceed Blackhole's 70,656-byte kernel
    config buffer.  Function calls do not change CB ordering or arithmetic;
    they only stop the C++ compiler from duplicating identical control code.
    """

    for name in names:
        markers = (
            f"FORCE_INLINE void {name}(",
            f"ALWI void {name}(",
        )
        matches = [marker for marker in markers if marker in source]
        if len(matches) != 1 or source.count(matches[0]) != 1:
            raise RuntimeError(f"size-optimized function marker changed: {name}")
        attributes = "noinline"
        if optimize is not None:
            attributes = f'optimize("{optimize}"), noinline'
        source = source.replace(
            matches[0], f"__attribute__(({attributes})) void {name}(", 1
        )
    return source


def _share_function_definitions(
    owner: str, consumer: str, names: tuple[str, ...]
) -> str:
    """Remove byte-equivalent helper definitions from a merged consumer.

    Both qualified projection generators embed the same exact-add/dA helper
    library. Once the kernels share one physical TRISC program, retaining two
    copies wastes scarce kernel text. Calls in ``consumer`` resolve to the
    definition in ``owner`` after :func:`_merge_phased_kernel_source` joins
    their preludes.
    """

    def function(source: str, name: str) -> tuple[int, int, str]:
        match = re.search(
            rf"^(?:FORCE_INLINE|ALWI) void {re.escape(name)}\(",
            source,
            re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"shared function marker changed: {name}")
        brace = source.index("{", match.start())
        depth = 0
        for index in range(brace, len(source)):
            depth += (source[index] == "{") - (source[index] == "}")
            if depth == 0:
                end = index + 1
                if end < len(source) and source[end] == "\n":
                    end += 1
                return match.start(), end, source[match.start():end]
        raise RuntimeError(f"unterminated shared function: {name}")

    def canonical(value: str) -> str:
        value = re.sub(r"//[^\n]*", "", value)
        return re.sub(r"\s+", "", value)

    removals: list[tuple[int, int]] = []
    for name in names:
        _, _, owner_body = function(owner, name)
        start, end, consumer_body = function(consumer, name)
        if canonical(owner_body) != canonical(consumer_body):
            raise RuntimeError(f"shared function arithmetic changed: {name}")
        removals.append((start, end))
    for start, end in sorted(removals, reverse=True):
        consumer = consumer[:start] + consumer[end:]
    return consumer


def _alias_equivalent_function(
    owner: str,
    consumer: str,
    *,
    owner_name: str,
    consumer_name: str,
) -> str:
    """Make a differently named but identical phase use the owner's body."""

    def function(source: str, name: str) -> tuple[int, int, str]:
        match = re.search(
            rf"^(?:FORCE_INLINE|ALWI) void {re.escape(name)}\(",
            source,
            re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"aliased function marker changed: {name}")
        brace = source.index("{", match.start())
        depth = 0
        for index in range(brace, len(source)):
            depth += (source[index] == "{") - (source[index] == "}")
            if depth == 0:
                end = index + 1
                if end < len(source) and source[end] == "\n":
                    end += 1
                return match.start(), end, source[match.start():end]
        raise RuntimeError(f"unterminated aliased function: {name}")

    _, _, owner_body = function(owner, owner_name)
    start, end, consumer_body = function(consumer, consumer_name)
    normalized_consumer = re.sub(
        rf"\b{re.escape(consumer_name)}\b", owner_name, consumer_body
    )
    canonical = lambda value: re.sub(r"\s+", "", value)  # noqa: E731
    if canonical(owner_body) != canonical(normalized_consumer):
        raise RuntimeError(
            f"aliased function arithmetic changed: {consumer_name}"
        )
    consumer = consumer[:start] + consumer[end:]
    return re.sub(
        rf"\b{re.escape(consumer_name)}\b", owner_name, consumer
    )


def _share_projection_phase_one_by_width(
    source: str,
    *,
    hidden_tiles: int,
    kv_tiles: int,
    workers: int,
) -> str:
    """Use one exact phase-one implementation for hidden and KV widths.

    The serial QKV builder emits a hidden-width output/Q implementation and a
    second K/V implementation.  After CB remapping they differ only in the N
    loop bounds; the K/V ``accumulate`` argument is consumed by its tail and
    is dead in phase one.  Passing those bounds to one ordinary device
    function removes a real TRISC text duplicate while retaining the same
    loop order, matmul reduction blocks, and BF16 pack sites.
    """

    def function(value: str, name: str) -> tuple[int, int, str]:
        match = re.search(
            rf"^(?:__attribute__\(\([^)]*(?:\)[^)]*)?\)\)|FORCE_INLINE|ALWI) "
            rf"void {re.escape(name)}\(",
            value,
            re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"width-shared function marker changed: {name}")
        brace = value.index("{", match.start())
        depth = 0
        for index in range(brace, len(value)):
            depth += (value[index] == "{") - (value[index] == "}")
            if depth == 0:
                end = index + 1
                if end < len(value) and value[end] == "\n":
                    end += 1
                return match.start(), end, value[match.start():end]
        raise RuntimeError(f"unterminated width-shared function: {name}")

    owner_lora = "lora_backward_phase"
    owner_base = "resident_down_phase_one"
    kv_lora = "kv_lora_backward_phase"
    kv_base = "kv_resident_down_phase_one"

    # Prove the generated static bodies still differ only in width, name, and
    # the unused K/V phase-one argument before deleting either definition.
    _, _, owner_lora_body = function(source, owner_lora)
    _, _, kv_lora_body = function(source, kv_lora)
    normalized_kv_lora = re.sub(rf"\b{kv_lora}\b", owner_lora, kv_lora_body)
    normalized_kv_lora = normalized_kv_lora.replace(
        f"num_n_blocks = {kv_tiles // 2}",
        f"num_n_blocks = {hidden_tiles // 2}",
    ).replace(
        f"total_n_tiles = {kv_tiles}",
        f"total_n_tiles = {hidden_tiles}",
    )
    canonical = lambda value: re.sub(r"\s+", "", value)  # noqa: E731
    if canonical(owner_lora_body) != canonical(normalized_kv_lora):
        raise RuntimeError("K/V LoRA phase-one arithmetic changed")

    _, _, owner_base_body = function(source, owner_base)
    _, _, kv_base_body = function(source, kv_base)
    normalized_kv_base = re.sub(rf"\b{kv_base}\b", owner_base, kv_base_body)
    normalized_kv_base = re.sub(rf"\b{kv_lora}\b", owner_lora, normalized_kv_base)
    normalized_kv_base = normalized_kv_base.replace("(bool accumulate)", "()")
    normalized_kv_base = normalized_kv_base.replace(
        f"n_tiles = {kv_tiles}", f"n_tiles = {hidden_tiles}"
    )
    if canonical(owner_base_body) != canonical(normalized_kv_base):
        raise RuntimeError("K/V frozen-base phase-one arithmetic changed")

    owner_lora_decl = owner_lora_body[:owner_lora_body.index("{")].rstrip()
    if not owner_lora_decl.endswith(f"void {owner_lora}()"):
        raise RuntimeError("hidden LoRA phase-one declaration changed")
    owner_lora_runtime = (
        owner_lora_decl[:-2]
        + "(\n    uint32_t projection_n_tiles, "
        "uint32_t projection_workers)"
    )
    if source.count(owner_lora_decl) != 1:
        raise RuntimeError("hidden LoRA phase-one declaration changed")
    source = source.replace(owner_lora_decl, owner_lora_runtime, 1)
    static_lora_bounds = (
        f"    constexpr uint32_t num_n_blocks = {hidden_tiles // 2};\n"
        "    constexpr uint32_t n_tiles =\n"
        "        num_n_blocks * reduction_block_tiles;"
    )
    runtime_lora_bounds = (
        "    const uint32_t num_n_blocks =\n"
        "        projection_n_tiles / reduction_block_tiles;\n"
        "    const uint32_t n_tiles = projection_n_tiles;"
    )
    if source.count(static_lora_bounds) != 1:
        raise RuntimeError("hidden LoRA N-block schedule changed")
    source = source.replace(static_lora_bounds, runtime_lora_bounds, 1)
    static_db_bounds = (
        "    constexpr uint32_t n_tiles_per_node = "
        f"{-(-hidden_tiles // workers)};\n"
        f"    constexpr uint32_t total_n_tiles = {hidden_tiles};"
    )
    runtime_db_bounds = (
        "    const uint32_t n_tiles_per_node =\n"
        "        (projection_n_tiles + projection_workers - 1) /\n"
        "        projection_workers;\n"
        "    const uint32_t total_n_tiles = projection_n_tiles;"
    )
    if source.count(static_db_bounds) != 1:
        raise RuntimeError("hidden LoRA dB worker schedule changed")
    source = source.replace(static_db_bounds, runtime_db_bounds, 1)

    owner_base_decl = owner_base_body[:owner_base_body.index("{")].rstrip()
    if not owner_base_decl.endswith(f"void {owner_base}()"):
        raise RuntimeError("hidden frozen-base phase-one declaration changed")
    owner_base_runtime = (
        owner_base_decl[:-2]
        + "(\n    uint32_t projection_n_tiles, "
        "uint32_t projection_workers)"
    )
    if source.count(owner_base_decl) != 1:
        raise RuntimeError("hidden frozen-base phase-one declaration changed")
    source = source.replace(owner_base_decl, owner_base_runtime, 1)
    hidden_n_tiles = f"    constexpr uint32_t n_tiles = {hidden_tiles};"
    if source.count(hidden_n_tiles) != 1:
        raise RuntimeError("hidden frozen-base N schedule changed")
    source = source.replace(
        hidden_n_tiles,
        "    const uint32_t n_tiles = projection_n_tiles;",
        1,
    )
    static_reduction_blocks = (
        "    constexpr uint32_t reduction_blocks =\n"
        "        n_tiles / reduction_tiles_per_phase;"
    )
    runtime_reduction_blocks = (
        "    const uint32_t reduction_blocks =\n"
        "        n_tiles / reduction_tiles_per_phase;"
    )
    if source.count(static_reduction_blocks) < 1:
        raise RuntimeError("hidden frozen-base reduction schedule changed")
    source = source.replace(
        static_reduction_blocks, runtime_reduction_blocks, 1
    )
    source = source.replace(
        f"    {owner_lora}();",
        f"    {owner_lora}(projection_n_tiles, projection_workers);",
        1,
    )

    # Remove the now redundant K/V implementations from the merged program.
    removals = []
    for name in (kv_lora, kv_base):
        start, end, _ = function(source, name)
        removals.append((start, end))
    for start, end in sorted(removals, reverse=True):
        source = source[:start] + source[end:]

    source = source.replace(
        f"{kv_base}(accumulate);",
        f"{owner_base}({kv_tiles}, {workers});",
    )
    source = source.replace(
        f"{owner_base}();",
        f"{owner_base}({hidden_tiles}, {workers});",
    )
    if re.search(
        rf"^\s*(?:{kv_lora}|{kv_base})\(", source, re.MULTILINE
    ):
        raise RuntimeError("obsolete K/V phase-one reference survived sharing")
    return source


def _remap_qkv_reader_abi(
    source: str,
    *,
    tensor_map: dict[int, tuple[int, int]],
    total_cbs: int,
) -> str:
    """Remap surviving QKV reader operands into a larger fused custom call."""

    original_cbs = _CBS

    def common(match: re.Match[str]) -> str:
        old = int(match.group(1))
        if old not in tensor_map:
            raise RuntimeError(f"unmapped QKV reader common argument {old}")
        return f"get_common_arg_val<uint32_t>({tensor_map[old][0]})"

    source = re.sub(
        r"get_common_arg_val<uint32_t>\((\d+)\)", common, source
    )

    def accessor(match: re.Match[str]) -> str:
        old_tensor = int(match.group(1))
        old_common = int(match.group(2))
        if old_tensor != old_common or old_tensor not in tensor_map:
            raise RuntimeError(
                "unexpected QKV reader tensor accessor mapping "
                f"{old_tensor}/{old_common}"
            )
        common_position, tensor_index = tensor_map[old_tensor]
        return (
            "get_tensor_accessor_args_cta_offset<"
            f"{tensor_index}, {total_cbs}>(), {common_position}"
        )

    source = re.sub(
        rf"get_tensor_accessor_args_cta_offset<(\d+), {original_cbs}>\(\),\s*(\d+)",
        accessor,
        source,
    )
    if f", {original_cbs}>()" in source:
        raise RuntimeError("unmapped QKV reader tensor accessor remains")
    return source


def _remap_qkv_writer_abi(
    source: str, *, output_tensor_index: int, total_cbs: int,
) -> str:
    """Point every prefixed final writer at the fused arena operand."""

    source = re.sub(
        rf"get_tensor_accessor_args_cta_offset<\d+, {_CBS}>\(\),\s*0",
        (
            "get_tensor_accessor_args_cta_offset<"
            f"{output_tensor_index}, {total_cbs}>(), 0"
        ),
        source,
    )
    if f", {_CBS}>()" in source:
        raise RuntimeError("unmapped QKV writer tensor accessor remains")
    return source


def _phase_replacements(
    *, m_tiles: int, k_tiles: int, n_tiles: int, scale_bits: int,
    base_phase: int,
) -> dict[str, str]:
    return {
        "@M_TILES@": str(m_tiles),
        "@K_TILES@": str(k_tiles),
        "@N_TILES@": str(n_tiles),
        "@BASE_REDUCTION_TILES@": str(base_phase),
        "@JOBS_PER_NODE@": str(-(-(m_tiles * k_tiles) // _WORKERS)),
        "@DA_K_TILES_PER_NODE@": str(-(-k_tiles // _WORKERS)),
        "@N_TILES_PER_NODE@": str(-(-n_tiles // _WORKERS)),
        "@TOTAL_JOBS@": str(m_tiles * k_tiles),
        "@REDUCTION_TILES_PER_PHASE@": str(base_phase),
        "@GRID_COLS@": "5",
        "@NUM_N_BLOCKS@": str(n_tiles // 2),
        "@M_TILES_PER_NODE@": "1",
        "@TOTAL_M_TILES@": str(m_tiles),
        "@TOTAL_N_TILES@": str(n_tiles),
        "@REDUCTION_TILES@": str(m_tiles),
        "@SCALE_BITS@": str(scale_bits),
    }


def _prefix(source: str, prefix: str, names: tuple[str, ...]) -> str:
    for name in names:
        source = re.sub(rf"\b{name}\b", f"{prefix}_{name}", source)
    return source


_PHASE_COMPUTE_NAMES = (
    "lora_backward_phase",
    "resident_down_phase_one",
    "run_resident_down_lora_dx",
    "resident_down_tail",
)


def _phase_compute_source(
    *, name: str, replacements: dict[str, str], accumulate: bool | None,
    include_helpers: bool,
) -> str:
    cpp = Path(__file__).with_name("cpp")
    lora = _shift_compile_time_cb_indices(
        (cpp / "lora_backward_arena_exact_compute.cpp").read_text(), 4
    ).replace("void kernel_main()", "FORCE_INLINE void lora_backward_phase()", 1)
    base = (cpp / "frozen_linear_backward_base_phase.cpp").read_text().replace(
        "void kernel_main()", "FORCE_INLINE void resident_down_phase_one()", 1
    )
    if accumulate is None:
        base = base.replace(
            "FORCE_INLINE void resident_down_phase_one()",
            "FORCE_INLINE void resident_down_phase_one(bool accumulate)",
            1,
        )
        base = base.replace(
            "    lora_backward_phase();",
            "    lora_backward_phase();",
            1,
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
        "            // Keep dH resident for LoRA dX.\n",
    )
    tail = (cpp / "resident_down_backward_tail.cpp").read_text()
    if accumulate is not False:
        declaration = (
            "    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(24);"
        )
        tail = tail.replace(
            declaration,
            declaration
            + "\n    constexpr uint32_t cb_accum_scratch = "
            "get_compile_time_arg_val(8);",
            1,
        )
        old = """        run_bf16_add(
            cb_base, cb_lora_dx, cb_final_dx,
            jobs_per_node, total_jobs, node);"""
        accumulated = """        // Preserve the framework's two new BF16 edges:
        // accumulator + LoRA, followed by + frozen-base.
        run_bf16_add(
            cb_final_dx, cb_lora_dx, cb_accum_scratch,
            jobs_per_node, total_jobs, node);
        run_bf16_add(
            cb_accum_scratch, cb_base, cb_final_dx,
            jobs_per_node, total_jobs, node);"""
        if accumulate is None:
            new = """        if (accumulate) {
""" + accumulated + """
        } else {
            run_bf16_add(
                cb_base, cb_lora_dx, cb_final_dx,
                jobs_per_node, total_jobs, node);
        }"""
            tail = tail.replace(
                "FORCE_INLINE void resident_down_tail() {",
                "FORCE_INLINE void resident_down_tail(bool accumulate) {",
                1,
            )
        else:
            new = accumulated
        if tail.count(old) != 1:
            raise RuntimeError("resident final-add marker changed")
        tail = tail.replace(old, new, 1)
    # The exact dA/add helper code is shape-generic: all dimensions are
    # runtime function arguments.  Define it once in the V phase instead of
    # asking the Metal compiler to optimize three identical copies.
    compute = (
        lora + "\n" + base + "\n"
        + (helper + "\n" if include_helpers else "")
        + tail
    )
    compute = compute.replace(
        "constexpr uint32_t grid_cols = 11;",
        "constexpr uint32_t grid_cols = 5;",
    )
    marker = "    resident_down_tail();\n}"
    completion = """    resident_down_tail();
    CircularBuffer resident_dh(get_compile_time_arg_val(16));
    resident_dh.pop_front(@M_TILES@);
    // CB7 is empty after dH and becomes a count-only compute->writer fence.
    CircularBuffer phase_done(get_compile_time_arg_val(7));
    phase_done.reserve_back(1);
    phase_done.push_back(1);
}"""
    if compute.count(marker) != 1:
        raise RuntimeError("resident compute completion marker changed")
    compute = compute.replace(marker, completion, 1)
    if accumulate is None:
        compute = compute.replace(
            "    resident_down_phase_one();",
            "    resident_down_phase_one(accumulate);",
            1,
        )
        compute = compute.replace(
            "    resident_down_tail();",
            "    resident_down_tail(accumulate);",
            1,
        )
    compute = _prefix(compute, name, _PHASE_COMPUTE_NAMES)
    compute = compute.replace(
        "void kernel_main()",
        f"static __attribute__((noinline, noclone)) void "
        f"{name}_compute_phase("
        + ("bool accumulate" if accumulate is None else "")
        + ")",
        1,
    )
    return _replace_all(compute, replacements)


def _remap_reader(
    source: str, *, name: str, tensor_map: tuple[int, ...],
) -> str:
    """Map conceptual dY/W/B/H/A/X operands into the grouped ABI."""

    source = source.replace(
        "constexpr uint32_t kGridCols = 11;\n"
        "constexpr uint32_t kWorkerNodes = 110;",
        "constexpr uint32_t kGridCols = 5;\n"
        "constexpr uint32_t kWorkerNodes = 50;",
    ).replace(
        "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
        "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};",
        "constexpr uint8_t kPhysicalX[kGridCols] = {1, 2, 3, 4, 5};",
    )
    # Remap in one pass.  Sequential substitutions are incorrect for a
    # permutation: e.g. Q's (0->1, 1->2, ..., 5->0) otherwise cascades until
    # every accessor points at operand zero.
    source = re.sub(
        r"get_common_arg_val<uint32_t>\((\d+)\)",
        lambda match: (
            "get_common_arg_val<uint32_t>("
            f"{tensor_map[int(match.group(1))]})"
        ),
        source,
    )
    source = re.sub(
        r"get_tensor_accessor_args_cta_offset<(\d+), 25>\(\),\s*(\d+)",
        lambda match: (
            "get_tensor_accessor_args_cta_offset<"
            f"{tensor_map[int(match.group(1))]}, 25>(), "
            f"{tensor_map[int(match.group(2))]}"
        ),
        source,
    )
    source = _prefix(
        source,
        name,
        ("kGridCols", "kWorkerNodes", "kTileBytes", "kPhysicalX", "physical_y"),
    )
    source = source.replace(
        "void kernel_main()", f"FORCE_INLINE void {name}_reader_phase()", 1
    )
    if name != "v":
        # The preceding phase's writer clears the complete resident-down
        # semaphore protocol on every core *before* its global barrier.  Do
        # not clear it again here: a fast reader can otherwise publish a dH
        # ready flag which a slower reader then erases while entering this
        # phase.  That cross-phase reset race deadlocked the K phase on
        # silicon even though V completed correctly.
        reset = """    // Semaphore addresses may contain state from an unrelated prior program.
    // Reset every flag on every invocation before any producer can publish.
    for (uint32_t semaphore = 0; semaphore < 2 * m_tiles + 1; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }

"""
        if source.count(reset) != 1:
            raise RuntimeError("resident reader semaphore reset marker changed")
        source = source.replace(reset, "", 1)
    if name == "v":
        node_marker = (
            "    const uint32_t node = logical_y * "
            f"{name}_kGridCols + logical_x;\n"
        )
        initialization = """    // These two IDs are outside resident-down's
    // 0..8 protocol. Clear stale state once, before the first phase.
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9)), 0);
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10)), 0);
"""
        if source.count(node_marker) != 1:
            raise RuntimeError("compact reader node marker changed")
        source = source.replace(node_marker, node_marker + initialization, 1)
    # Writer node zero publishes this only after all 50 writers have drained
    # the phase. A local per-core fence is insufficient because the next
    # resident reader resets semaphores used by cross-core dH multicast.
    end = source.rfind("\n}")
    if end < 0:
        raise RuntimeError("resident reader closing brace changed")
    wait = """
    volatile tt_l1_ptr uint32_t* phase_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
    noc_semaphore_wait(phase_done, 1);
    noc_semaphore_set(phase_done, 0);
"""
    return source[:end] + wait + source[end:]


def _phase_reader_source(
    *, name: str, replacements: dict[str, str],
    tensor_map: tuple[int, ...],
) -> str:
    cpp = Path(__file__).with_name("cpp")
    source = (cpp / "resident_down_backward_reader.cpp").read_text()
    source = _remap_reader(source, name=name, tensor_map=tensor_map)
    return _replace_all(source, replacements)


def _phase_writer_source(
    *, name: str, replacements: dict[str, str], arena_index: int,
    da_row: int, db_row: int, write_dx: bool,
) -> str:
    cpp = Path(__file__).with_name("cpp")
    source = (cpp / "resident_down_backward_writer.cpp").read_text()
    source = source.replace(
        "get_absolute_logical_y() * 11 + get_absolute_logical_x()",
        "get_absolute_logical_y() * 5 + get_absolute_logical_x()",
    ).replace(
        "get_tensor_accessor_args_cta_offset<6, 25>(), 0>()",
        f"get_tensor_accessor_args_cta_offset<{arena_index}, 25>(), 0>()",
    )
    source = source.replace(
        ".page_id = (m_tiles + 1 + n) * arena_cols",
        f".page_id = ({db_row} + n) * arena_cols",
    ).replace(
        ".page_id = m_tiles * arena_cols + k",
        f".page_id = {da_row} * arena_cols + k",
    )
    if not write_dx:
        start = source.index("    const uint32_t first_job = node * jobs_per_node;")
        end = source.rindex("\n}")
        source = source[:start] + source[end:]
    source = _prefix(source, name, ("kTileBytes",))
    source = source.replace(
        "void kernel_main()", f"FORCE_INLINE void {name}_writer_phase()", 1
    )
    end = source.rfind("\n}")
    signal = """
    // A count-only CB fence guarantees compute has completed the phase.
    CircularBuffer phase_done_cb(get_compile_time_arg_val(7));
    phase_done_cb.wait_front(1);
    phase_done_cb.pop_front(1);
    // Clear this core's complete resident-down semaphore protocol while all
    // readers are still held behind the previous phase.  The global writer
    // barrier below proves every core has completed this reset before any
    // reader is released into the next phase.
    constexpr uint32_t resident_m_tiles = @M_TILES@;
    for (uint32_t semaphore = 0;
         semaphore < 2 * resident_m_tiles + 1; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }
    // A complete device-side barrier is required here. A fast core must not
    // enter the next reader and clear semaphores while a slow core is still
    // finishing this phase's cross-core dH multicast.
    constexpr uint32_t phase_count_semaphore = 9;
    constexpr uint32_t phase_go_semaphore = 10;
    noc_semaphore_inc(
        get_noc_addr(1, 2, get_semaphore(phase_count_semaphore)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(phase_count_semaphore));
        noc_semaphore_wait_min(count, 50);
        noc_semaphore_set(count, 0);
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(phase_go_semaphore));
        noc_semaphore_set(go, 1);
        for (uint32_t target = 1; target < 50; ++target) {
            noc_semaphore_inc(
                get_noc_addr(
                    1 + target % 5, 2 + target / 5,
                    get_semaphore(phase_go_semaphore)),
                1);
        }
        noc_async_atomic_barrier();
    }
"""
    source = source[:end] + signal + source[end:]
    return _replace_all(source, replacements)


def _compose_qkv_sources(
    *, m_tiles: int, hidden_tiles: int, kv_tiles: int,
    scale_bits: int, base_phase: int, arena_index: int,
    phase_count: int = 3,
) -> tuple[str, str, str]:
    """Compose the independently qualified projection phases in V/K/Q order."""

    if phase_count not in (1, 2, 3):
        raise ValueError("QKV diagnostic phase count must be 1, 2, or 3")

    q = _phase_replacements(
        m_tiles=m_tiles, k_tiles=hidden_tiles, n_tiles=hidden_tiles,
        scale_bits=scale_bits, base_phase=base_phase,
    )
    kv = _phase_replacements(
        m_tiles=m_tiles, k_tiles=hidden_tiles, n_tiles=kv_tiles,
        scale_bits=scale_bits, base_phase=base_phase,
    )
    # ABI: hidden, Q(dY/W/B/H/A), K(...), V(...), arena.
    maps = {
        "v": (11, 12, 13, 14, 15, 0),
        "k": (6, 7, 8, 9, 10, 0),
        "q": (1, 2, 3, 4, 5, 0),
    }
    phase_names = ("v", "k", "q")[:phase_count]
    readers = [
        _phase_reader_source(
            name=name, replacements=q if name == "q" else kv,
            tensor_map=maps[name],
        )
        for name in phase_names
    ]
    # V and K have identical dimensions and numerical kernels. Define that
    # phase body once and select only the final accumulation tree at the two
    # call sites. This avoids a pathological duplicate-template compile while
    # preserving every BF16 pack boundary.
    computes = [
        _phase_compute_source(
            name="kv", replacements=kv, accumulate=None,
            include_helpers=True,
        ),
    ] + ([
        _phase_compute_source(
            name="q", replacements=q, accumulate=True,
            include_helpers=False,
        ),
    ] if phase_count == 3 else [])
    q_da = m_tiles
    k_da = q_da + 1
    v_da = k_da + 1
    q_db = v_da + 1
    k_db = q_db + hidden_tiles
    v_db = k_db + kv_tiles
    all_writers = [
        _phase_writer_source(
            name="v", replacements=kv, arena_index=arena_index,
            da_row=v_da, db_row=v_db, write_dx=phase_count == 1,
        ),
        _phase_writer_source(
            name="k", replacements=kv, arena_index=arena_index,
            da_row=k_da, db_row=k_db, write_dx=phase_count == 2,
        ),
        _phase_writer_source(
            name="q", replacements=q, arena_index=arena_index,
            da_row=q_da, db_row=q_db, write_dx=True,
        ),
    ]
    writers = all_writers[:phase_count]
    reader_calls = (
        "    v_reader_phase();\n"
        + ("    k_reader_phase();\n" if phase_count >= 2 else "")
        + ("    q_reader_phase();\n" if phase_count >= 3 else "")
    )
    reader = "\n".join(readers) + """
void kernel_main() {
""" + reader_calls + """}
}
""".replace("}\n}\n", "}\n")
    compute_calls = (
        "    kv_compute_phase(false);\n"
        + ("    kv_compute_phase(true);\n" if phase_count >= 2 else "")
        + ("    q_compute_phase();\n" if phase_count >= 3 else "")
    )
    compute = "\n".join(computes) + """
void kernel_main() {
""" + compute_calls + "}\n"
    writer_calls = (
        "    v_writer_phase();\n"
        + ("    k_writer_phase();\n" if phase_count >= 2 else "")
        + ("    q_writer_phase();\n" if phase_count >= 3 else "")
    )
    writer = "\n".join(writers) + """
void kernel_main() {
""" + writer_calls + "}\n"
    # TensorAccessor compile-time metadata starts after all CB descriptors.
    # The source templates were authored for the original 25-CB projection.
    reader = reader.replace(", 25>", f", {_CBS}>")
    writer = writer.replace(", 25>", f", {_CBS}>")
    return reader, compute, writer


def _compose_parallel_kv_q_sources(
    *, m_tiles: int, hidden_tiles: int, kv_tiles: int,
    scale_bits: int, base_phase: int, arena_index: int,
) -> tuple[str, str, str]:
    """Compose qualified 55+55 K/V followed by a 50-core Q tail.

    The grouped K/V member-zero result remains in CB12.  For the currently
    qualified SmolLM2 and Qwen token grids, K/V and Q use the same job owner
    and local slot, so Q consumes that tile without a NoC or DRAM copy.
    """

    from .tt_lang_full_mlp_backward import _grouped_sources

    jobs = m_tiles * hidden_tiles
    kv_jobs_per_node = -(-jobs // 55)
    q_jobs_per_node = -(-jobs // 50)
    if kv_jobs_per_node != q_jobs_per_node:
        raise ValueError(
            "parallel QKV direct-CB schedule requires matching K/V and Q "
            "job strides"
        )
    grouped_replacements = {
        "@M_TILES@": str(m_tiles),
        "@K_TILES@": str(hidden_tiles),
        "@N_TILES@": str(kv_tiles),
        "@BASE_REDUCTION_TILES@": str(base_phase),
        "@JOBS_PER_NODE@": str(kv_jobs_per_node),
        "@TOTAL_JOBS@": str(jobs),
        "@REDUCTION_TILES_PER_PHASE@": str(base_phase),
        "@GRID_COLS@": "11",
        "@NUM_N_BLOCKS@": str(kv_tiles // 2),
        "@M_TILES_PER_NODE@": "1",
        "@TOTAL_M_TILES@": str(m_tiles),
        "@N_TILES_PER_NODE@": str(-(-kv_tiles // 55)),
        "@TOTAL_N_TILES@": str(kv_tiles),
        "@REDUCTION_TILES@": str(m_tiles),
        "@SCALE_BITS@": str(scale_bits),
        "@MEMBER_NODES@": "55",
        "@DX_JOBS_PER_NODE@": str(kv_jobs_per_node),
        "@TOTAL_DX_JOBS@": str(jobs),
        "@K_TILES_PER_NODE@": "1",
        "@TOTAL_K_TILES@": str(hidden_tiles),
        "@RANK_TILES@": "1",
    }
    kv_compute, kv_reader, kv_writer = _grouped_sources(
        replacements=grouped_replacements, total_cbs=_CBS
    )
    # _grouped_sources is also used by the 100-core MLP island. Restore the
    # full 55+55 geometry for attention projection ownership.
    for old, new in (
        ("constexpr uint32_t grid_cols = 10;", "constexpr uint32_t grid_cols = 11;"),
        ("constexpr uint32_t member_nodes = 50;", "constexpr uint32_t member_nodes = 55;"),
        ("constexpr uint32_t kGridCols = 10;", "constexpr uint32_t kGridCols = 11;"),
        ("constexpr uint32_t kMemberNodes = 50;", "constexpr uint32_t kMemberNodes = 55;"),
        (
            "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
            "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};",
            "constexpr uint8_t kPhysicalX[kGridCols] = {\n"
            "    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};",
        ),
    ):
        kv_compute = kv_compute.replace(old, new)
        kv_reader = kv_reader.replace(old, new)
        kv_writer = kv_writer.replace(old, new)

    request_write = """            volatile tt_l1_ptr uint32_t* request_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(10));
            noc_semaphore_set(request_sem, 1);
            noc_async_write(
                get_semaphore(10),
                get_noc_addr(
                    kPhysicalX[logical_x],
                    physical_y(peer_y),
                    get_semaphore(10)),
                sizeof(uint32_t));
            noc_async_write_barrier();
            noc_semaphore_set(request_sem, 0);"""
    request_atomic = """            noc_semaphore_inc(
                get_noc_addr(
                    kPhysicalX[logical_x],
                    physical_y(peer_y),
                    get_semaphore(10)),
                1);
            noc_async_atomic_barrier();"""
    if kv_reader.count(request_write) != 1:
        raise RuntimeError("grouped peer request protocol changed")
    kv_reader = kv_reader.replace(request_write, request_atomic, 1)

    response_write = """            volatile tt_l1_ptr uint32_t* response_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(9));
            noc_semaphore_set(response_sem, valid_jobs);
            noc_async_write(
                get_semaphore(9),
                get_noc_addr(
                    kPhysicalX[logical_x],
                    physical_y(peer_logical_y),
                    get_semaphore(9)),
                sizeof(uint32_t));
            noc_async_write_barrier();
            noc_semaphore_set(response_sem, 0);"""
    response_atomic = """            noc_semaphore_inc(
                get_noc_addr(
                    kPhysicalX[logical_x],
                    physical_y(peer_logical_y),
                    get_semaphore(9)),
                valid_jobs);
            noc_async_atomic_barrier();"""
    if kv_writer.count(response_write) != 1:
        raise RuntimeError("grouped peer response protocol changed")
    kv_writer = kv_writer.replace(response_write, response_atomic, 1)

    kv_reader = kv_reader.replace(
        "for (uint32_t semaphore = 0; semaphore < 11; ++semaphore)",
        "for (uint32_t semaphore = 0; semaphore < 13; ++semaphore)",
        1,
    )

    # Grouped ABI member zero=K, member one=V, then shared hidden input.
    grouped_map = {**{index: index + 6 for index in range(10)}, 10: 0}
    kv_reader = re.sub(
        r"get_common_arg_val<uint32_t>\((\d+)\)",
        lambda match: (
            "get_common_arg_val<uint32_t>("
            f"{grouped_map[int(match.group(1))]})"
        ),
        kv_reader,
    )
    kv_reader = re.sub(
        r"get_tensor_accessor_args_cta_offset<(\d+), 25>\(\),\s*(\d+)",
        lambda match: (
            "get_tensor_accessor_args_cta_offset<"
            f"{grouped_map[int(match.group(1))]}, 25>(), "
            f"{grouped_map[int(match.group(1))]}"
        ),
        kv_reader,
    )
    kv_writer = kv_writer.replace(
        "get_tensor_accessor_args_cta_offset<11, 25>(),\n            0>()",
        f"get_tensor_accessor_args_cta_offset<{arena_index}, 25>(),\n            0>()",
    )
    # Place K/V adapter gradients after the reserved Q rows.
    kv_writer = kv_writer.replace(
        "const uint32_t row =\n            m_tiles + 2 + member * n_tiles + n_tile;",
        """const uint32_t row =
            m_tiles + 3 + @Q_N_TILES@ + member * n_tiles + n_tile;""",
    ).replace(
        "const uint32_t row = m_tiles + member;",
        "const uint32_t row = m_tiles + 1 + member;",
    ).replace("@Q_N_TILES@", str(hidden_tiles))
    # The grouped tail normally emits its exact K/V input-gradient sum in
    # CB12.  Q also needs CB12 as dB transpose scratch, so retain the cross-
    # phase value in dedicated CB25 instead.  It has one producer and one
    # consumer and is never visible to the writer or DRAM.
    kv_compute = kv_compute.replace(
        "constexpr uint32_t cb_final_dx = get_compile_time_arg_val(12);",
        "constexpr uint32_t cb_final_dx = get_compile_time_arg_val(25);",
        1,
    )
    dx_start = kv_writer.index("    if (member == 0) {")
    dx_end = kv_writer.rindex("\n}")
    kv_writer = kv_writer[:dx_start] + kv_writer[dx_end:]
    kv_writer = kv_writer.replace(
        "constexpr uint32_t cb_final_dx = get_compile_time_arg_val(12);",
        "constexpr uint32_t cb_final_dx = get_compile_time_arg_val(25);",
        1,
    )

    # All 110 grouped writers finish and clear their semaphore protocol before
    # the first 50 linear workers enter Q. Semaphores 11/12 are phase-local.
    barrier = """
    // Use the real retained K/V result as member-zero's compute-completion
    // fence. It stays at CB25's front because Q consumes it after the global
    // device-side barrier. Member one is already fenced by its peer copy.
    if (member == 0 && valid_jobs != 0) {
        final_dx.wait_front(valid_jobs);
    }
    for (uint32_t semaphore = 0; semaphore < 11; ++semaphore) {
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore)), 0);
    }
    noc_semaphore_inc(get_noc_addr(1, 2, get_semaphore(11)), 1);
    noc_async_atomic_barrier();
    if (global_node == 0) {
        volatile tt_l1_ptr uint32_t* count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(11));
        noc_semaphore_wait_min(count, 110);
        noc_semaphore_set(count, 0);
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
        noc_semaphore_set(go, 1);
        for (uint32_t target = 1; target < 110; ++target) {
            noc_semaphore_inc(
                get_noc_addr(
                    kPhysicalX[target % 11], physical_y(target / 11),
                    get_semaphore(12)),
                1);
        }
        noc_async_atomic_barrier();
    }
"""
    kv_writer = kv_writer[:kv_writer.rindex("\n}")] + barrier + "\n}\n"

    q_replacements = _phase_replacements(
        m_tiles=m_tiles, k_tiles=hidden_tiles, n_tiles=hidden_tiles,
        scale_bits=scale_bits, base_phase=base_phase,
    )
    q_reader = _phase_reader_source(
        name="q", replacements=q_replacements,
        tensor_map=(1, 2, 3, 4, 5, 0),
    )
    q_compute = _phase_compute_source(
        name="q", replacements=q_replacements,
        accumulate=True, include_helpers=False,
    )
    q_writer = _phase_writer_source(
        name="q", replacements=q_replacements, arena_index=arena_index,
        da_row=m_tiles, db_row=m_tiles + 3, write_dx=True,
    )
    # Q is the terminal phase of this program.  The reusable projection
    # fragments normally hold their reader behind a writer barrier so another
    # phase can safely reuse semaphores.  There is no successor here: program
    # completion is the synchronization boundary, so remove that otherwise
    # redundant (and racy) terminal handshake.
    terminal_reader_wait = """    volatile tt_l1_ptr uint32_t* phase_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
    noc_semaphore_wait(phase_done, 1);
    noc_semaphore_set(phase_done, 0);
"""
    if q_reader.count(terminal_reader_wait) != 1:
        raise RuntimeError("Q terminal reader wait marker changed")
    q_reader = q_reader.replace(terminal_reader_wait, "", 1)
    terminal_writer_barrier = """    // Clear this core's complete resident-down semaphore protocol while all
    // readers are still held behind the previous phase.  The global writer
    // barrier below proves every core has completed this reset before any
    // reader is released into the next phase.
    constexpr uint32_t resident_m_tiles = @M_TILES@;
    for (uint32_t semaphore = 0;
         semaphore < 2 * resident_m_tiles + 1; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }
    // A complete device-side barrier is required here. A fast core must not
    // enter the next reader and clear semaphores while a slow core is still
    // finishing this phase's cross-core dH multicast.
    constexpr uint32_t phase_count_semaphore = 9;
    constexpr uint32_t phase_go_semaphore = 10;
    noc_semaphore_inc(
        get_noc_addr(1, 2, get_semaphore(phase_count_semaphore)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(phase_count_semaphore));
        noc_semaphore_wait_min(count, 50);
        noc_semaphore_set(count, 0);
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(phase_go_semaphore));
        noc_semaphore_set(go, 1);
        for (uint32_t target = 1; target < 50; ++target) {
            noc_semaphore_inc(
                get_noc_addr(
                    1 + target % 5, 2 + target / 5,
                    get_semaphore(phase_go_semaphore)),
                1);
        }
        noc_async_atomic_barrier();
    }
""".replace("@M_TILES@", str(m_tiles))
    if q_writer.count(terminal_writer_barrier) != 1:
        raise RuntimeError("Q terminal writer barrier marker changed")
    q_writer = q_writer.replace(terminal_writer_barrier, "", 1)
    # Q uses the first 50 row-major workers, matching the active member-zero
    # owners of both qualified shapes.
    q_reader = q_reader.replace("q_kGridCols = 5", "q_kGridCols = 11").replace(
        "q_kPhysicalX[q_kGridCols] = {1, 2, 3, 4, 5}",
        "q_kPhysicalX[q_kGridCols] = {1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13}",
    )
    q_compute = q_compute.replace(
        "constexpr uint32_t grid_cols = 5;",
        "constexpr uint32_t grid_cols = 11;",
    )
    q_writer = q_writer.replace(
        "get_absolute_logical_y() * 5 + get_absolute_logical_x()",
        "get_absolute_logical_y() * 11 + get_absolute_logical_x()",
    )
    q_node = "    const uint32_t node = logical_y * q_kGridCols + logical_x;\n"
    q_reader = q_reader.replace(
        q_node,
        q_node + """    if (node >= 50) return;
    volatile tt_l1_ptr uint32_t* phase_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
    noc_semaphore_wait(phase_go, 1);
    noc_semaphore_set(phase_go, 0);
""",
        1,
    )
    q_compute = q_compute.replace(
        "static __attribute__((noinline, noclone)) void q_compute_phase() {",
        """static __attribute__((optimize("O1"), noinline, noclone)) void q_compute_phase() {
    const uint32_t q_global_node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (q_global_node >= 50) return;""",
        1,
    )
    q_writer = q_writer.replace(
        "    const uint32_t arena_addr =",
        "    if (node >= 50) return;\n\n    const uint32_t arena_addr =",
        1,
    )
    # The exact K/V sum is already at CB12's front. Q adds its LoRA term and
    # frozen-base term with the accepted two BF16 pack boundaries.
    q_compute = q_compute.replace(
        "constexpr uint32_t cb_accum_scratch = get_compile_time_arg_val(8);",
        """constexpr uint32_t cb_accum_scratch = get_compile_time_arg_val(26);
    constexpr uint32_t cb_prior_dx = get_compile_time_arg_val(25);""",
        1,
    ).replace(
        "cb_final_dx, cb_lora_dx, cb_accum_scratch,",
        "cb_prior_dx, cb_lora_dx, cb_accum_scratch,",
        1,
    )

    # Q and K/V differ only in projection width.  Keeping two copies of the
    # complete exact phase-one body makes the SFPI compiler spend minutes in
    # rvtt_rmext (and O0 crashes that pass).  Parameterize the accepted K/V
    # body by width and reuse it for Q.  All qualified shapes have identical
    # dHidden job strides, and both dB schedules assign one output tile per
    # active worker, so this changes loop bounds—not traversal or rounding.
    kv_compute = kv_compute.replace(
        "FORCE_INLINE void lora_backward_phase() {",
        "FORCE_INLINE void lora_backward_phase(\n"
        "    uint32_t projection_n_tiles, uint32_t projection_workers) {",
        1,
    )
    kv_compute = kv_compute.replace(
        f"    constexpr uint32_t num_n_blocks = {kv_tiles};\n".replace(
            str(kv_tiles), str(kv_tiles // 2)
        )
        + "    constexpr uint32_t n_tiles =\n"
        "        num_n_blocks * reduction_block_tiles;",
        "    const uint32_t num_n_blocks =\n"
        "        projection_n_tiles / reduction_block_tiles;\n"
        "    const uint32_t n_tiles = projection_n_tiles;",
        1,
    )
    kv_compute = kv_compute.replace(
        "    constexpr uint32_t n_tiles_per_node = 1;\n"
        f"    constexpr uint32_t total_n_tiles = {kv_tiles};",
        "    const uint32_t n_tiles_per_node =\n"
        "        (projection_n_tiles + projection_workers - 1) /\n"
        "        projection_workers;\n"
        "    const uint32_t total_n_tiles = projection_n_tiles;",
        1,
    )
    kv_compute = kv_compute.replace(
        "FORCE_INLINE void resident_phase_one() {",
        "FORCE_INLINE void resident_phase_one(\n"
        "    uint32_t projection_n_tiles, uint32_t projection_workers) {",
        1,
    ).replace(
        f"    constexpr uint32_t n_tiles = {kv_tiles};",
        "    const uint32_t n_tiles = projection_n_tiles;",
        1,
    ).replace(
        "    constexpr uint32_t reduction_blocks =\n"
        "        n_tiles / reduction_tiles_per_phase;",
        "    const uint32_t reduction_blocks =\n"
        "        n_tiles / reduction_tiles_per_phase;",
        1,
    ).replace(
        "    lora_backward_phase();",
        "    lora_backward_phase(projection_n_tiles, projection_workers);",
        1,
    ).replace(
        "        resident_phase_one();",
        f"        resident_phase_one({kv_tiles}, 55);",
        1,
    )
    kv_compute = kv_compute.replace(
        "    resident_tail();\n}",
        f"""    resident_tail();
    // K/V dH is dead after its tail.  Release the window before Q's reader
    // reuses CB16 for the hidden-width dH broadcast.
    CircularBuffer kv_resident_dh(get_compile_time_arg_val(16));
    kv_resident_dh.pop_front({m_tiles});
}}""",
        1,
    )
    # Retain only Q's width-independent tail.  It consumes the shared phase-one
    # dH from CB16 and preserves the accepted two final BF16 accumulation edges.
    q_tail_marker = "// Compute tail appended after the exact phase-one helpers."
    if q_tail_marker not in q_compute:
        raise RuntimeError("Q projection tail marker changed")
    q_compute = q_compute[q_compute.index(q_tail_marker):]
    q_compute = q_compute.replace(
        "        q_resident_down_phase_one();",
        f"        resident_phase_one({hidden_tiles}, 50);",
        1,
    )
    q_compute = q_compute.replace(
        "    q_resident_down_tail();",
        "    q_resident_down_tail();",
        1,
    )

    kv_reader = kv_reader.replace(
        "void kernel_main()", "FORCE_INLINE void kv_reader_phase()", 1
    )
    kv_compute = kv_compute.replace(
        "void kernel_main()",
        "static __attribute__((optimize(\"O1\"), noinline, noclone)) void kv_compute_phase()",
        1,
    )
    # Keep the low-level Tensix/SFPI mathematical helpers FORCE_INLINE: the
    # rvtt_rmext pass cannot compile those helpers as standalone functions.
    # O1 is the lowest accepted optimization level for the two complete outer
    # phases; O0 triggers an SFPI compiler ICE and unconstrained O3 expands the
    # combined graph pathologically.
    kv_writer = kv_writer.replace(
        "void kernel_main()", "FORCE_INLINE void kv_writer_phase()", 1
    )
    reader = kv_reader + "\n" + q_reader + """
void kernel_main() { kv_reader_phase(); q_reader_phase(); }
"""
    compute = kv_compute + "\n" + q_compute + """
void kernel_main() { kv_compute_phase(); q_compute_phase(); }
"""
    writer = kv_writer + "\n" + q_writer + """
void kernel_main() { kv_writer_phase(); q_writer_phase(); }
"""
    # TensorAccessor metadata starts immediately after all CB descriptors.
    # Adding the dedicated handoff CB shifts every reader/writer CTA offset.
    reader = reader.replace(", 25>", f", {_CBS}>")
    writer = writer.replace(", 25>", f", {_CBS}>")
    return reader, compute, writer


@lru_cache(maxsize=8)
def build_tt_lang_qkv_resident_backward(
    *, tokens: int, hidden_features: int, kv_features: int,
    physical_rank: int, scale: float, diagnostic_phase_count: int = 3,
    schedule: str = "serial_diagnostic",
) -> Callable:
    """Build the isolated one-program exact QKV projection owner."""

    error = qkv_resident_backward_support_error(
        tokens=tokens, hidden_features=hidden_features,
        kv_features=kv_features, physical_rank=physical_rank,
    )
    if error is not None:
        raise ValueError(error)
    if diagnostic_phase_count not in (1, 2, 3):
        raise ValueError("diagnostic_phase_count must be 1, 2, or 3")
    if schedule not in ("serial_diagnostic", "parallel_kv_q"):
        raise ValueError("schedule must be serial_diagnostic or parallel_kv_q")
    if schedule == "parallel_kv_q" and diagnostic_phase_count != 3:
        raise ValueError("parallel_kv_q always owns all three projections")
    m_tiles = tokens // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    kv_tiles = kv_features // TILE_SIZE
    base_phase = exact_base_dx_k_phase(hidden_features, hidden_features)
    assert base_phase is not None
    jobs_per_node = -(-(m_tiles * hidden_tiles) // _WORKERS)
    da_k_tiles_per_node = -(-hidden_tiles // _WORKERS)
    n_tiles_per_node = -(-hidden_tiles // _WORKERS)
    scale_bf16 = float(torch.tensor(scale, dtype=torch.bfloat16).float().item())
    scale_bits = struct.unpack("<I", struct.pack("<f", scale_bf16))[0]

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("resident QKV backward requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    arena_shape = qkv_resident_backward_arena_shape(
        tokens=tokens, hidden_features=hidden_features,
        kv_features=kv_features,
    )

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _program(
        hidden,
        q_dy, q_weight, q_b, q_h, q_a,
        k_dy, k_weight, k_b, k_h, k_a,
        v_dy, v_weight, v_b, v_h, v_a,
        arena,
    ):
        shapes = _projection_cb_shapes(
            m_tiles=m_tiles, n_tiles=hidden_tiles,
            base_phase=base_phase, jobs_per_node=jobs_per_node,
            da_k_tiles_per_node=da_k_tiles_per_node,
            n_tiles_per_node=n_tiles_per_node,
        )
        # TT-Lang captures individual DFB objects, not Python containers.
        b0 = ttl.make_dataflow_buffer_like(arena, shape=shapes[0][0], block_count=shapes[0][1])
        b1 = ttl.make_dataflow_buffer_like(arena, shape=shapes[1][0], block_count=shapes[1][1])
        b2 = ttl.make_dataflow_buffer_like(arena, shape=shapes[2][0], block_count=shapes[2][1])
        b3 = ttl.make_dataflow_buffer_like(arena, shape=shapes[3][0], block_count=shapes[3][1])
        b4 = ttl.make_dataflow_buffer_like(arena, shape=shapes[4][0], block_count=shapes[4][1])
        b5 = ttl.make_dataflow_buffer_like(arena, shape=shapes[5][0], block_count=shapes[5][1])
        b6 = ttl.make_dataflow_buffer_like(arena, shape=shapes[6][0], block_count=shapes[6][1])
        b7 = ttl.make_dataflow_buffer_like(arena, shape=shapes[7][0], block_count=shapes[7][1])
        b8 = ttl.make_dataflow_buffer_like(arena, shape=shapes[8][0], block_count=shapes[8][1])
        b9 = ttl.make_dataflow_buffer_like(arena, shape=shapes[9][0], block_count=shapes[9][1])
        b10 = ttl.make_dataflow_buffer_like(arena, shape=shapes[10][0], block_count=shapes[10][1])
        b11 = ttl.make_dataflow_buffer_like(arena, shape=shapes[11][0], block_count=shapes[11][1])
        b12 = ttl.make_dataflow_buffer_like(arena, shape=shapes[12][0], block_count=shapes[12][1])
        b13 = ttl.make_dataflow_buffer_like(arena, shape=shapes[13][0], block_count=shapes[13][1])
        b14 = ttl.make_dataflow_buffer_like(arena, shape=shapes[14][0], block_count=shapes[14][1])
        b15 = ttl.make_dataflow_buffer_like(arena, shape=shapes[15][0], block_count=shapes[15][1])
        b16 = ttl.make_dataflow_buffer_like(arena, shape=shapes[16][0], block_count=shapes[16][1])
        b17 = ttl.make_dataflow_buffer_like(arena, shape=shapes[17][0], block_count=shapes[17][1])
        b18 = ttl.make_dataflow_buffer_like(arena, shape=shapes[18][0], block_count=shapes[18][1])
        b19 = ttl.make_dataflow_buffer_like(arena, shape=shapes[19][0], block_count=shapes[19][1])
        b20 = ttl.make_dataflow_buffer_like(arena, shape=shapes[20][0], block_count=shapes[20][1])
        b21 = ttl.make_dataflow_buffer_like(arena, shape=shapes[21][0], block_count=shapes[21][1])
        b22 = ttl.make_dataflow_buffer_like(arena, shape=shapes[22][0], block_count=shapes[22][1])
        b23 = ttl.make_dataflow_buffer_like(arena, shape=shapes[23][0], block_count=shapes[23][1])
        # The probe pushes all ABI tokens through b24; its larger depth is
        # replaced by the exact per-core capacity before launch.
        b24 = ttl.make_dataflow_buffer_like(arena, shape=shapes[24][0], block_count=16)
        # Dedicated K/V -> Q resident dHidden handoff.  It cannot alias CB12,
        # which Q's dB path reuses as transpose scratch.
        b25 = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=jobs_per_node
        )
        # Q's two-edge exact accumulation scratch has compute-only ownership.
        # It must not alias CB8: cores 0..M-1 previously use CB8 as the dH
        # output, and Blackhole's destination/packer state is observable when
        # that CB is repurposed later in the same compute kernel.
        b26 = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=jobs_per_node
        )

        @ttl.datamovement()
        def read():
            with b0.reserve() as block: tx = ttl.copy(hidden[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(q_dy[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(q_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(q_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(q_h[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(q_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(k_dy[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(k_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(k_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(k_h[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(k_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(v_dy[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(v_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(v_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(v_h[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block: tx = ttl.copy(v_a[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(16):
                with b0.wait() as item:
                    with b24.reserve() as output:
                        output.store(item)
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
                with b25.wait() as _: pass
                with b26.wait() as _: pass

        @ttl.datamovement()
        def write():
            for _ in range(16):
                with b24.wait() as item:
                    tx = ttl.copy(item, arena[0:1, 0:1]); tx.wait()

    variant = hashlib.sha256(
        (
            f"qkv-resident-v1:{tokens}:{hidden_features}:{kv_features}:"
            f"{physical_rank}:{scale_bf16:.17g}:phases={diagnostic_phase_count}:"
            f"schedule={schedule}"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.qkv_resident_backward.{variant}",
        arg_roles=("in",) * 16 + ("out",),
        version_tag=f"qkv-resident-backward-{variant}",
    )
    def operation(
        hidden,
        q_dy, q_weight, q_b, q_h, q_a,
        k_dy, k_weight, k_b, k_h, k_a,
        v_dy, v_weight, v_b, v_h, v_a,
        arena,
    ):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(
            hidden,
            q_dy, q_weight, q_b, q_h, q_a,
            k_dy, k_weight, k_b, k_h, k_a,
            v_dy, v_weight, v_b, v_h, v_a,
            arena,
        )
        closure = _program.__closure__ or ()
        for freevar, cell in zip(_program.__code__.co_freevars, closure):
            if freevar != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != _CBS:
                    raise RuntimeError(
                        f"QKV probe expected {_CBS} CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                exact_shapes = _projection_cb_shapes(
                    m_tiles=m_tiles,
                    n_tiles=hidden_tiles,
                    base_phase=base_phase,
                    jobs_per_node=jobs_per_node,
                    da_k_tiles_per_node=da_k_tiles_per_node,
                    n_tiles_per_node=n_tiles_per_node,
                )
                for index, (shape, blocks) in exact_shapes.items():
                    compiled.cb_configs[index].shape = shape
                    compiled.cb_configs[index].block_count = blocks
                compiled.cb_configs[25].shape = (1, 1)
                compiled.cb_configs[25].block_count = jobs_per_node
                compiled.cb_configs[26].shape = (1, 1)
                compiled.cb_configs[26].block_count = jobs_per_node
                if schedule == "parallel_kv_q":
                    # Grouped K/V uses CB17 as a complete transposed-X window.
                    # Sequential producers cannot wrap one large publication
                    # across a non-zero FIFO write pointer.  Size aliased CBs
                    # for their complete K/V + Q publication history, rather
                    # than only the largest individual phase.
                    compiled.cb_configs[17].block_count = max(
                        compiled.cb_configs[17].block_count, m_tiles
                    )
                    sequential_capacities = (
                        _parallel_qkv_sequential_cb_capacities(
                            m_tiles=m_tiles,
                            hidden_tiles=hidden_tiles,
                            kv_tiles=kv_tiles,
                            jobs_per_node=jobs_per_node,
                        )
                    )
                    for index, tiles in sequential_capacities.items():
                        compiled.cb_configs[index].shape = (1, 1)
                        compiled.cb_configs[index].block_count = tiles
                for index in (2, 7, 13, 20):
                    compiled.cb_configs[index].tensor = _FP32_DTYPE_ONLY_TENSOR
                if schedule == "parallel_kv_q":
                    sources = _compose_parallel_kv_q_sources(
                        m_tiles=m_tiles, hidden_tiles=hidden_tiles,
                        kv_tiles=kv_tiles, scale_bits=scale_bits,
                        base_phase=base_phase, arena_index=16,
                    )
                    core_range = _ALL_GRID
                    semaphore_count = 13
                else:
                    sources = _compose_qkv_sources(
                        m_tiles=m_tiles, hidden_tiles=hidden_tiles,
                        kv_tiles=kv_tiles, scale_bits=scale_bits,
                        base_phase=base_phase, arena_index=16,
                        phase_count=diagnostic_phase_count,
                    )
                    core_range = _GRID
                    semaphore_count = 11
                generated = Path(compiled.kernel_paths[0][0]).parent
                specs = (
                    ("qkv_resident_reader.cpp", "noc", sources[0]),
                    ("qkv_resident_compute.cpp", "compute", sources[1]),
                    ("qkv_resident_writer.cpp", "noc", sources[2]),
                )
                paths = []
                for filename, kind, source in specs:
                    path = generated / filename
                    path.write_text(source, encoding="utf-8")
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
                ]
                compiled.kernel_tensor_indices = [
                    list(range(16)), [], [16]
                ]
                compiled.kernel_core_ranges = [core_range] * 3
                compiled.kernel_noc_roles = ["reader", None, "writer"]
                compiled.cb_core_ranges = [core_range] * _CBS
                compiled.semaphore_configs = [
                    {
                        "id": semaphore,
                        "core_type": "worker",
                        # Every kernel core touches the phase semaphores.  The
                        # parallel K/V schedule spans the full 11x10 grid;
                        # allocating only the legacy 5x10 range makes member
                        # one access unallocated semaphore L1 and trips the
                        # device watcher before useful work begins.
                        "core_range": core_range,
                        "initial_value": 0,
                    }
                    for semaphore in range(semaphore_count)
                ]
                compiled.num_pipe_sync_semaphores = semaphore_count
                compiled.kernel_arg_specs = [[], [], []]
            break
        return result

    operation.arena_shape = arena_shape
    operation.projection_order = ("v", "k", "q")
    operation.intermediate_dram_outputs = 0
    operation.experimental_backend = f"qkv_resident_exact_v1:{schedule}"
    return operation
