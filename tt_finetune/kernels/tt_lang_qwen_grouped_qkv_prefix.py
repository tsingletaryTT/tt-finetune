"""Exact Qwen grouped QKV producer used after a resident RMSNorm."""

from __future__ import annotations

from functools import lru_cache

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import compiled_ttl_program

ttl = None
tt_lang_operation = None


def grouped_qwen_compute_config() -> dict[str, object]:
    return {"math_fidelity": "HiFi4", "fp32_dest_acc_en": True}


def grouped_qwen_rank_config() -> dict[str, object]:
    return {
        "family": "2d", "grid": [11, 10], "in0_block_w": 1,
        "out_subblock_h": 1, "out_subblock_w": 1,
        "out_block_h": 1, "out_block_w": 1,
        "per_core_m": 1, "per_core_n": 1,
        "transpose_mcast": False, "fuse_batch": False,
    }


def grouped_qwen_base_config() -> dict[str, object]:
    return {
        "family": "1d", "grid": [11, 10], "in0_block_w": 2,
        "out_subblock_h": 4, "out_subblock_w": 1,
        "out_block_h": 4, "out_block_w": 1,
        "per_core_m": 4, "per_core_n": 1,
        "mcast_in0": True, "gather_in0": False,
        "untilize_out": False, "fuse_batch": False,
    }


@lru_cache(maxsize=4)
def build_tt_lang_qwen_grouped_qkv_prefix(
    *, tokens: int, hidden_features: int, grouped_features: int, logical_rank: int
):
    if (tokens, hidden_features, grouped_features, logical_rank) != (
        128, 896, 1152, 16
    ):
        raise ValueError("Qwen grouped prefix has only been qualified at 128x896")
    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(hidden, weight_t, grouped_a, bias, h_out, base_out):
        source = ttl.make_dataflow_buffer_like(hidden, shape=(1, 1), block_count=1)
        result = ttl.make_dataflow_buffer_like(h_out, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(hidden[0:1, 0:1], tile); tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with result.reserve() as target: target.store(tile)

        @ttl.datamovement()
        def write():
            with result.wait() as tile:
                tx = ttl.copy(tile, h_out[0:1, 0:1]); tx.wait()

    variant = "qwen2p5_0p5b_s128.v9-linear-l1-input"

    @tt_lang_operation(
        operation_id=f"tt_finetune.qwen_grouped_qkv_prefix.{variant}",
        arg_roles=("in", "in", "in", "in", "out", "out"),
        version_tag=f"qwen-grouped-qkv-prefix-{variant}",
    )
    def operation(hidden, weight_t, grouped_a, bias, h_out, base_out):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(hidden, weight_t, grouped_a, bias, h_out, base_out)
        compiled = compiled_ttl_program(_probe)
        compiled.program_sequence = [
            {"operation": "ttnn.full", "operand_indices": [6],
             "fill_value": 0.0},
            {"operation": "ttnn.add", "operand_indices": [0, 6, 7]},
            {
                "operation": "ttnn.matmul", "operand_indices": [7, 2, 4],
                "transpose_a": False, "transpose_b": True,
                "compute_config": grouped_qwen_compute_config(),
                "program_config": grouped_qwen_rank_config(),
            },
            {
                "operation": "ttnn.linear",
                "operand_indices": [7, 1, 3, 5],
                "transpose_a": False, "transpose_b": True,
                "bias_rank1": True,
                "compute_config": grouped_qwen_compute_config(),
            },
        ]
        compiled.internal_tensors = [
            {"source_operand": 0, "memory": "l1",
             "shape": [tokens, hidden_features],
             "data_format": "BFloat16", "layout": "tile"},
            {"source_operand": 0, "memory": "l1",
             "shape": [tokens, hidden_features],
             "data_format": "BFloat16", "layout": "tile"},
        ]
        return result

    operation.output_shapes = (
        (tokens, 3 * logical_rank), (tokens, grouped_features)
    )
    operation.artifact_format = 4
    operation.physical_programs = 4
    operation.compiler_owned_l1_tensors = 2
    operation._ttl_program = _probe
    return operation


__all__ = [
    "build_tt_lang_qwen_grouped_qkv_prefix",
    "grouped_qwen_compute_config",
    "grouped_qwen_rank_config",
    "grouped_qwen_base_config",
]
