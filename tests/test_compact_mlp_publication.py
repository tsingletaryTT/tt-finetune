from __future__ import annotations

import pytest

from tt_finetune.kernels.tt_lang_program_composition import (
    serialized_child_tensor_ids,
)
from tt_finetune.kernels.tt_lang_attention_forward_qkv_resident import (
    _layer_swiglu_down_compute_source,
)
from tt_finetune.kernels.tt_lang_resident_layer_forward_compact_owner import (
    _fuse_grouped2_checkpoint_publication,
    _fuse_residual_add_publication,
    _fuse_rmsnorm_checkpoint_publication,
)


def _grouped2_writer_artifact() -> dict:
    return {
        "format_version": 3,
        "num_tensors": 7,
        "cb_configs": [],
        "kernels": [
            {
                "thread_type": "noc",
                "tensor_indices": [6],
                "cpp_source": """
void kernel_main() {
  CircularBuffer output(get_compile_time_arg_val(12));
  CircularBuffer saved_h(get_compile_time_arg_val(4));
  auto args = TensorAccessorArgs<
      tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 16>(), 0>();
  Noc noc1(1);
  noc1.async_write(source, destination, 2048, {}, {.page_id = 0});
}
""",
            }
        ],
    }


def test_grouped2_writer_publishes_arena_and_checkpoint_in_one_program() -> None:
    fused = _fuse_grouped2_checkpoint_publication(
        _grouped2_writer_artifact(),
        sequence_tiles=4,
        intermediate_tiles=48,
        rank_tiles=1,
        gate_page_offset=100,
        up_page_offset=292,
        gate_h_page_offset=484,
        up_h_page_offset=488,
    )

    assert serialized_child_tensor_ids(fused) == {6, 7}
    assert fused["num_tensors"] == 8
    writer = fused["kernels"][0]
    assert writer["tensor_indices"] == [6, 7]
    source = writer["cpp_source"]
    assert "kJobsPerNode = 4" in source
    assert "kArenaRowTiles = 98" in source
    assert "kGatePageOffset = 100" in source
    assert "kUpPageOffset = 292" in source
    assert "kGateHPageOffset = 484" in source
    assert "kUpHPageOffset = 488" in source
    assert "get_tensor_accessor_args_cta_offset<\n                7, 16>()" in source
    assert source.count("noc1.async_write(") == 4
    assert "output.pop_front(1);" in source
    assert "saved_h.pop_front(kRankTiles);" in source


def test_grouped2_writer_rewrite_fails_closed_on_unknown_abi() -> None:
    artifact = _grouped2_writer_artifact()
    artifact["kernels"][0]["cpp_source"] = "void kernel_main() {}"

    with pytest.raises(ValueError, match="one qualified writer"):
        _fuse_grouped2_checkpoint_publication(
            artifact,
            sequence_tiles=4,
            intermediate_tiles=48,
            rank_tiles=1,
            gate_page_offset=0,
            up_page_offset=1,
            gate_h_page_offset=2,
            up_h_page_offset=3,
        )


def _residual_add_artifact() -> dict:
    return {
        "format_version": 3,
        "num_tensors": 3,
        "cb_configs": [],
        "kernels": [
            {
                "thread_type": "noc",
                "tensor_indices": [2],
                "cpp_source": """
void kernel_main() {
  CircularBuffer output(get_compile_time_arg_val(2));
  auto args = TensorAccessorArgs<
      tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 3>(), 0>();
  Noc noc1(1);
  noc1.async_write(source, destination, 2048, {}, {.page_id = 0});
}
""",
            }
        ],
    }


def test_residual_writer_publishes_l1_and_all_checkpoint_views() -> None:
    fused = _fuse_residual_add_publication(
        _residual_add_artifact(),
        row_tiles=4,
        column_tiles=18,
        destination_page_offsets=(1112, 0, 1112),
    )

    assert serialized_child_tensor_ids(fused) == {2, 3}
    assert fused["num_tensors"] == 4
    writer = fused["kernels"][0]
    assert writer["tensor_indices"] == [2, 3]
    source = writer["cpp_source"]
    assert "kColumnTiles = 18" in source
    assert "kTotalTiles = 72" in source
    assert ".page_id = 1112 + tile_id" in source
    assert ".page_id = 0 + tile_id" in source
    assert source.count("noc1.async_write(") == 3
    assert source.count("noc1.async_write_barrier();") == 3
    assert "result.pop_front(1);" in source


def test_residual_writer_rewrite_fails_closed_on_unknown_abi() -> None:
    artifact = _residual_add_artifact()
    artifact["kernels"][0]["tensor_indices"] = [1]

    with pytest.raises(ValueError, match="tensor ABI changed"):
        _fuse_residual_add_publication(
            artifact,
            row_tiles=1,
            column_tiles=1,
            destination_page_offsets=(0,),
        )


def _rmsnorm_writer_artifact() -> dict:
    return {
        "format_version": 3,
        "num_tensors": 5,
        "cb_configs": [],
        "kernels": [
            {
                "thread_type": "noc",
                "tensor_indices": [31],
                "cpp_source": """
// Final BF16 writer for the exact resident RMSNorm forward qualifier.
void kernel_main() {
  constexpr uint32_t cb_output = 16;
  constexpr uint32_t published_block_tiles = 4;
  constexpr bool replicate_scalar = false;
  constexpr uint32_t total_cbs = 23;
  auto args = TensorAccessorArgs<
      tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
          31, total_cbs>(), 0>();
}
""",
            }
        ],
    }


def test_rmsnorm_writer_publishes_exact_output_and_checkpoint() -> None:
    fused = _fuse_rmsnorm_checkpoint_publication(
        _rmsnorm_writer_artifact(),
        row_tiles=4,
        column_tiles=18,
        destination_page_offset=648,
        output_tensor_index=31,
        checkpoint_tensor_index=30,
    )

    assert serialized_child_tensor_ids(fused) == {30, 31}
    assert fused["num_tensors"] == 6
    writer = fused["kernels"][0]
    assert writer["tensor_indices"] == [31, 30]
    source = writer["cpp_source"]
    assert "cb_output = 16" in source
    assert "published_block_tiles = 4" in source
    assert "output_tiles = 18" in source
    assert "token_tiles = 4" in source
    assert "checkpoint_page_offset = 648" in source
    assert "31, total_cbs" in source
    assert "30, total_cbs" in source
    assert source.count("noc1.async_write(") == 2
    assert "output.pop_front(published_block_tiles);" in source


def test_rmsnorm_writer_rewrite_fails_closed_for_scalar_schedule() -> None:
    artifact = _rmsnorm_writer_artifact()
    artifact["kernels"][0]["cpp_source"] = artifact["kernels"][0][
        "cpp_source"
    ].replace("replicate_scalar = false", "replicate_scalar = true")

    with pytest.raises(ValueError, match="full-width output"):
        _fuse_rmsnorm_checkpoint_publication(
            artifact,
            row_tiles=1,
            column_tiles=1,
            destination_page_offset=0,
            output_tensor_index=31,
            checkpoint_tensor_index=30,
        )


def test_experimental_decomposed_mlp_uses_a_valid_60_core_schedule() -> None:
    source = _layer_swiglu_down_compute_source()

    assert "constexpr uint32_t kOutputCoreCount = kCoreCount;" in source
    assert "constexpr uint32_t kProductCoreStart = 0;" in source
    assert "constexpr uint32_t kProductCoreCount = kCoreCount;" in source
    assert "kCoreCount - kOutputCoreCount" not in source


def test_experimental_decomposed_mlp_reads_the_distributed_product_window() -> None:
    source = _layer_swiglu_down_compute_source()

    assert "? cb_gate : cb_product_resident" in source
    assert "? cb_up : cb_product_resident" in source
    assert source.count("local * i_tiles") >= 2
