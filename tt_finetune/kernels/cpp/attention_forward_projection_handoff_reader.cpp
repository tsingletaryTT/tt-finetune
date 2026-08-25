// Qualification producer: load token-major Q/K/V on the 50-core projection island.
// The final owner replaces these reads with the exact projection compute outputs.

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_q = 0, cb_k = 3, cb_v = 29;
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@, kv_tiles = @KV_TILES@;
    constexpr uint32_t workers = 50, columns = 5, tile_bytes = 2048;
    const uint32_t x = get_absolute_logical_x(), y = get_absolute_logical_y();
    const uint32_t node = y * columns + x;
    if (x >= columns) return;
    const auto query = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 32>(), 0>(),
        get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto key = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 32>(), 1>(),
        get_common_arg_val<uint32_t>(1), tile_bytes);
    const auto value = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 32>(), 2>(),
        get_common_arg_val<uint32_t>(2), tile_bytes);
    Noc noc0(0);
    CircularBuffer q(cb_q), k(cb_k), v(cb_v);
    constexpr uint32_t q_jobs = seq_tiles * hidden_tiles;
    constexpr uint32_t q_per_node = (q_jobs + workers - 1) / workers;
    for (uint32_t local = 0; local < q_per_node; ++local) {
        const uint32_t job = node * q_per_node + local;
        if (job >= q_jobs) continue;
        q.reserve_back(1);
        noc0.async_read(query, CoreLocalMem<uint32_t>(q.get_write_ptr()),
                        tile_bytes, {.page_id = job}, {});
        noc0.async_read_barrier(); q.push_back(1);
    }
    constexpr uint32_t kv_jobs = seq_tiles * kv_tiles;
    constexpr uint32_t kv_per_node = (kv_jobs + workers - 1) / workers;
    for (uint32_t local = 0; local < kv_per_node; ++local) {
        const uint32_t job = node * kv_per_node + local;
        if (job >= kv_jobs) continue;
        k.reserve_back(1);
        noc0.async_read(key, CoreLocalMem<uint32_t>(k.get_write_ptr()),
                        tile_bytes, {.page_id = job}, {});
        noc0.async_read_barrier(); k.push_back(1);
    }
    for (uint32_t local = 0; local < kv_per_node; ++local) {
        const uint32_t job = node * kv_per_node + local;
        if (job >= kv_jobs) continue;
        v.reserve_back(1);
        noc0.async_read(value, CoreLocalMem<uint32_t>(v.get_write_ptr()),
                        tile_bytes, {.page_id = job}, {});
        noc0.async_read_barrier(); v.push_back(1);
    }
}
