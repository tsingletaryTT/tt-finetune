"""Native exact inverse-RoPE phase for the full attention owner."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


def inverse_rope_arena_shape(
    *, sequence_length: int, query_heads: int, kv_heads: int, head_dim: int,
) -> tuple[int, int]:
    return (query_heads + kv_heads) * sequence_length, head_dim


@lru_cache(maxsize=8)
def build_tt_lang_inverse_rope_backward(
    *, sequence_length: int, query_heads: int, kv_heads: int, head_dim: int,
) -> Callable:
    if sequence_length != 128 or head_dim != 64:
        raise ValueError("first exact inverse-RoPE phase requires S=128, D=64")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must divide into positive KV heads")
    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    jobs = (query_heads + kv_heads) * seq_tiles * dim_tiles
    jobs_per_node = -(-jobs // 110)

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("inverse-RoPE phase requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
    )
    def _program(q_gradient, k_gradient, cosine, sine, arena):
        b0 = ttl.make_dataflow_buffer_like(
            q_gradient, shape=(1, 1), block_count=max(2, jobs_per_node)
        )
        b1 = ttl.make_dataflow_buffer_like(
            cosine, shape=(1, 1), block_count=max(2, jobs_per_node)
        )
        b2 = ttl.make_dataflow_buffer_like(
            q_gradient, shape=(1, 1), block_count=max(2, jobs_per_node)
        )
        b3 = ttl.make_dataflow_buffer_like(
            sine, shape=(1, 1), block_count=max(2, jobs_per_node)
        )
        b4 = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, jobs_per_node)
        )
        b5 = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, jobs_per_node)
        )
        b6 = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=max(2, jobs_per_node)
        )

        @ttl.datamovement()
        def read():
            with b0.reserve() as block:
                tx = ttl.copy(q_gradient[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_gradient[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(cosine[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(sine[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(4):
                with b0.wait() as item:
                    with b6.reserve() as result:
                        result.store(item)
            x, y = ttl.node(dims=2)
            if x + y < 0:
                with b1.wait() as _item: pass
                with b2.wait() as _item: pass
                with b3.wait() as _item: pass
                with b4.wait() as _item: pass
                with b5.wait() as _item: pass

        @ttl.datamovement()
        def write():
            for _ in range(4):
                with b6.wait() as item:
                    tx = ttl.copy(item, arena[0:1, 0:1]); tx.wait()

    variant = hashlib.sha256(
        f"inverse-rope-v1:{sequence_length}:{query_heads}:{kv_heads}:{head_dim}".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.inverse_rope_backward.{variant}",
        arg_roles=("in", "in", "in", "in", "out"),
        version_tag=f"inverse-rope-backward-{variant}",
    )
    def operation(q_gradient, k_gradient, cosine, sine, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(q_gradient, k_gradient, cosine, sine, arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 7:
                    raise RuntimeError(
                        f"inverse-RoPE expected 7 CBs, got {len(compiled.cb_configs)}"
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
                if (
                    len(reader_indices) != 1
                    or sorted(reader_indices[0]) != [0, 1, 2, 3]
                    or writer_indices != [[4]]
                ):
                    raise RuntimeError(
                        "inverse-RoPE ABI changed: "
                        f"reader={reader_indices}, writer={writer_indices}"
                    )
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles),
                    "@DIM_TILES@": str(dim_tiles),
                    "@QUERY_HEADS@": str(query_heads),
                    "@KV_HEADS@": str(kv_heads),
                    "@JOBS_PER_NODE@": str(jobs_per_node),
                }
                cpp = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp / "inverse_rope_backward_reader.cpp").read_text(),
                    "compute": (cpp / "inverse_rope_backward_compute.cpp").read_text(),
                    "write": (cpp / "inverse_rope_backward_writer.cpp").read_text(),
                }
                reader_common_position = {
                    tensor_index: common_position
                    for common_position, tensor_index in enumerate(
                        reader_indices[0]
                    )
                }
                sources["read"] = re.sub(
                    r"get_common_arg_val<uint32_t>\((\d+)\)",
                    lambda match: (
                        "get_common_arg_val<uint32_t>("
                        f"{reader_common_position[int(match.group(1))]})"
                    ),
                    sources["read"],
                )
                sources["read"] = re.sub(
                    (
                        r"get_tensor_accessor_args_cta_offset<"
                        r"(\d+), 7>\(\),\s*(\d+)"
                    ),
                    lambda match: (
                        "get_tensor_accessor_args_cta_offset<"
                        f"{match.group(1)}, 7>(), "
                        f"{reader_common_position[int(match.group(1))]}"
                    ),
                    sources["read"],
                )
                for kind, source in tuple(sources.items()):
                    for marker, value in replacements.items():
                        source = source.replace(marker, value)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(
                            f"unexpanded inverse-RoPE {kind} placeholder"
                        )
                    sources[kind] = source
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
            break
        return result

    operation.arena_shape = inverse_rope_arena_shape(
        sequence_length=sequence_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    operation.intermediate_dram_outputs = 0
    return operation
