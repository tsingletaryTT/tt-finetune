"""One multi-core exact norm program for all packed logical gradients."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .optimizer_gradient_arena_planner import (
    plan_multi_layer_optimizer_gradient_arena,
)
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache


ttl = None
tt_lang_operation = None


def _load_tt_lang() -> None:
    global ttl, tt_lang_operation
    if ttl is not None and tt_lang_operation is not None:
        return
    try:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
    except ImportError as exc:  # pragma: no cover - runtime-only path
        raise RuntimeError("grouped gradient norm requires TT-XLA") from exc
    ttl, tt_lang_operation = ttl_module, decorator


def _install_native_kernels(
    program,
    *,
    page_offsets: tuple[int, ...],
    heights: tuple[int, ...],
    widths: tuple[int, ...],
    layer_count: int,
    layer_page_stride: int,
    maximum_pages: int,
    grid_width: int,
    grid_height: int,
) -> None:
    if layer_count <= 0 or len(page_offsets) % layer_count:
        raise ValueError("grouped norm jobs must divide evenly across layers")
    jobs_per_layer = len(page_offsets) // layer_count
    pattern_offsets = page_offsets[:jobs_per_layer]
    pattern_heights = heights[:jobs_per_layer]
    pattern_widths = widths[:jobs_per_layer]
    for layer in range(layer_count):
        begin = layer * jobs_per_layer
        end = begin + jobs_per_layer
        expected_offsets = tuple(
            offset + layer * layer_page_stride for offset in pattern_offsets
        )
        if page_offsets[begin:end] != expected_offsets:
            raise ValueError("grouped norm page offsets are not layer-periodic")
        if heights[begin:end] != pattern_heights:
            raise ValueError("grouped norm heights are not layer-periodic")
        if widths[begin:end] != pattern_widths:
            raise ValueError("grouped norm widths are not layer-periodic")

    base = Path(__file__).with_name("cpp")
    replacements = {
        "@JOBS@": str(len(page_offsets)),
        "@JOBS_PER_LAYER@": str(jobs_per_layer),
        "@LAYER_PAGE_STRIDE@": str(layer_page_stride),
        "@PAGE_OFFSET_PATTERN@": ", ".join(map(str, pattern_offsets)),
        "@PAGE_COUNT_PATTERN@": ", ".join(
            str(height * width)
            for height, width in zip(pattern_heights, pattern_widths)
        ),
        "@HEIGHT_TILE_PATTERN@": ", ".join(map(str, pattern_heights)),
        "@WIDTH_TILE_PATTERN@": ", ".join(map(str, pattern_widths)),
        "@GRID_WIDTH@": str(grid_width),
        "@CORE_COUNT@": str(grid_width * grid_height),
    }
    sources = {}
    for role in ("read", "compute", "write"):
        source = base.joinpath(
            f"grouped_gradient_l2_norm_{role}.cpp"
        ).read_text(encoding="utf-8")
        for marker, value in replacements.items():
            source = source.replace(marker, value)
        if "@" in source:
            raise RuntimeError(f"unresolved grouped norm marker in {role}")
        sources[role] = source

    closure = program.__closure__ or ()
    for name, cell in zip(program.__code__.co_freevars, closure):
        if name != "cache" or not isinstance(cell.cell_contents, dict):
            continue
        for compiled in cell.cell_contents.values():
            if len(compiled.cb_configs) != 6:
                raise RuntimeError(
                    "grouped gradient norm CB descriptor contract changed: "
                    f"expected 6, got {len(compiled.cb_configs)}"
                )
            compiled.cb_configs[2].block_count = maximum_pages
            noc_paths = []
            for kernel_path, thread_type in compiled.kernel_paths:
                path = Path(kernel_path)
                if thread_type == "compute":
                    path.write_text(sources["compute"], encoding="utf-8")
                elif thread_type == "noc":
                    noc_paths.append(path)
            if len(noc_paths) != 2:
                raise RuntimeError(
                    "grouped gradient norm requires one reader and writer"
                )
            reader = next(
                (path for path in noc_paths if "read" in path.name), None
            )
            writer = next(
                (path for path in noc_paths if "write" in path.name), None
            )
            if reader is None or writer is None:
                raise RuntimeError("grouped norm NOC kernel roles changed")
            reader.write_text(sources["read"], encoding="utf-8")
            writer.write_text(sources["write"], encoding="utf-8")
        return
    raise RuntimeError("grouped norm compile cache was not materialized")


@lru_cache(maxsize=16)
def build_tt_lang_grouped_gradient_l2_norm(
    *,
    layer_count: int,
    sequence_length: int,
    hidden_features: int,
    intermediate_features: int,
    kv_features: int,
    physical_rank: int = 32,
) -> Callable:
    """Compute all exact per-logical-tensor norms in one physical program.

    Each TT-Lang worker owns region ``node + grid_size*k``. Regions retain their
    original page order and independent BF16 square/reduce/sqrt boundary, so
    grouping changes scheduling but not the accepted numerical contract.
    """

    plan = plan_multi_layer_optimizer_gradient_arena(
        layer_count=layer_count,
        tokens=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        physical_rank=physical_rank,
    )
    page_offsets = tuple(region.page_offset for region in plan.slices)
    heights = tuple(
        (region.logical_shape[0] + 31) // 32 for region in plan.slices
    )
    widths = tuple(
        (region.logical_shape[1] + 31) // 32 for region in plan.slices
    )
    maximum_pages = max(
        height * width for height, width in zip(heights, widths)
    )
    # TT-Lang currently serializes `grid="full"` as an 8x8 core range on
    # Blackhole.  Keep this explicit so native logical-core indexing exactly
    # matches the program descriptor.  A future 11x10 TT-Metal owner can update
    # both the descriptor and these planner values together.
    grid_width = 8
    grid_height = 8
    _load_tt_lang()

    @ttl.operation(grid="full", fp32_dest_acc_en=True)
    def grouped_norm_program(gradient_in, scaler_in, norm_out):
        gradient = ttl.make_dataflow_buffer_like(
            gradient_in, shape=(1, 1), block_count=2
        )
        scaler = ttl.make_dataflow_buffer_like(
            scaler_in, shape=(1, 1), block_count=1
        )
        square = ttl.make_dataflow_buffer_like(
            gradient_in,
            shape=(1, 1),
            block_count=min(maximum_pages, 32),
        )
        reduced = ttl.make_dataflow_buffer_like(
            norm_out, shape=(1, 1), block_count=1
        )
        norm = ttl.make_dataflow_buffer_like(
            norm_out, shape=(1, 1), block_count=1
        )
        exponent = ttl.make_dataflow_buffer_like(
            scaler_in, shape=(1, 1), block_count=1
        )

        @ttl.compute()
        def compute():
            for _ in range(maximum_pages):
                with gradient.wait() as value:
                    with square.reserve() as output:
                        output.store(value * value)
            with reduced.reserve() as output:
                with scaler.wait() as value:
                    output.store(value)
            with reduced.wait() as value:
                exponent.wait().pop()
                with norm.reserve() as output:
                    output.store(ttl.sqrt(value))

        @ttl.datamovement()
        def read():
            with scaler.reserve() as value:
                tx = ttl.copy(scaler_in[0:1, 0:1], value)
                tx.wait()
            with exponent.reserve() as value:
                tx = ttl.copy(scaler_in[1:2, 0:1], value)
                tx.wait()
            for page in range(maximum_pages):
                with gradient.reserve() as value:
                    tx = ttl.copy(
                        gradient_in[page : page + 1, 0:1], value
                    )
                    tx.wait()

        @ttl.datamovement()
        def write():
            with norm.wait() as value:
                tx = ttl.copy(value, norm_out[0:1, 0:1])
                tx.wait()

    digest = hashlib.sha256(
        (
            f"l{layer_count}:{sequence_length}:{hidden_features}:"
            f"{intermediate_features}:{kv_features}:grouped-v5-periodic"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.grouped_gradient_l2_norm.{digest}",
        arg_roles=("in", "in", "out"),
        version_tag=f"grouped-gradient-l2-norm-{digest}",
    )
    def operation(gradient_in, scaler_in, norm_out):
        _clear_ttl_compile_wrapper_cache(grouped_norm_program)
        result = grouped_norm_program(gradient_in, scaler_in, norm_out)
        _install_native_kernels(
            grouped_norm_program,
            page_offsets=page_offsets,
            heights=heights,
            widths=widths,
            layer_count=layer_count,
            layer_page_stride=plan.layer_gradient_pages,
            maximum_pages=maximum_pages,
            grid_width=grid_width,
            grid_height=grid_height,
        )
        return result

    operation.physical_programs = 1
    operation.logical_gradient_tensors = len(plan.slices)
    operation.maximum_pages_per_core = maximum_pages
    operation.norm_arena_shape = (len(plan.slices) * 32, 32)
    operation.safe_default = False
    operation.experimental_backend = "grouped_exact_gradient_l2_norm_v1"
    operation._ttl_program = grouped_norm_program
    return operation
