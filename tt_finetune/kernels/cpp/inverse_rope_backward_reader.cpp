// Read one output tile and its half-rotated dependency for inverse RoPE.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t cb_gradient = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cosine = get_compile_time_arg_val(1);
    constexpr uint32_t cb_peer_gradient = get_compile_time_arg_val(2);
    constexpr uint32_t cb_peer_sine = get_compile_time_arg_val(3);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t kv_heads = @KV_HEADS@;
    constexpr uint32_t jobs = (query_heads + kv_heads) * seq_tiles * dim_tiles;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t tile_bytes = 2048;
    static_assert(dim_tiles == 2, "first exact inverse-RoPE phase requires D=64");

    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t cos_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t sin_addr = get_common_arg_val<uint32_t>(3);
    const auto q = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 7>(), 0>(),
        q_addr, tile_bytes);
    const auto k = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 7>(), 1>(),
        k_addr, tile_bytes);
    const auto cosine = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 7>(), 2>(),
        cos_addr, tile_bytes);
    const auto sine = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 7>(), 3>(),
        sin_addr, tile_bytes);
    CircularBuffer gradient(cb_gradient), cos_cb(cb_cosine);
    CircularBuffer peer_gradient(cb_peer_gradient), peer_sine(cb_peer_sine);

    const uint32_t q_jobs = query_heads * seq_tiles * dim_tiles;
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= jobs) continue;
        const bool is_key = job >= q_jobs;
        const uint32_t local_job = is_key ? job - q_jobs : job;
        const uint32_t dim = local_job % dim_tiles;
        const uint32_t row = (local_job / dim_tiles) % seq_tiles;
        const uint32_t peer_dim = dim ^ 1u;
        const uint32_t tensor_page = local_job;
        const uint32_t peer_page = local_job - dim + peer_dim;
        const uint32_t trig_page = row * dim_tiles + dim;
        const uint32_t peer_trig_page = row * dim_tiles + peer_dim;
        const auto& source = is_key ? k : q;

        gradient.reserve_back(1);
        cos_cb.reserve_back(1);
        peer_gradient.reserve_back(1);
        peer_sine.reserve_back(1);
        noc0.async_read(
            source, CoreLocalMem<uint32_t>(gradient.get_write_ptr()),
            tile_bytes, {.page_id = tensor_page}, {});
        noc0.async_read(
            cosine, CoreLocalMem<uint32_t>(cos_cb.get_write_ptr()),
            tile_bytes, {.page_id = trig_page}, {});
        noc0.async_read(
            source, CoreLocalMem<uint32_t>(peer_gradient.get_write_ptr()),
            tile_bytes, {.page_id = peer_page}, {});
        noc0.async_read(
            sine, CoreLocalMem<uint32_t>(peer_sine.get_write_ptr()),
            tile_bytes, {.page_id = peer_trig_page}, {});
        noc0.async_read_barrier();
        gradient.push_back(1);
        cos_cb.push_back(1);
        peer_gradient.push_back(1);
        peer_sine.push_back(1);
    }
}

