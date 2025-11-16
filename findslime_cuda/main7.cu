#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <cstdlib>

#define DIM 1251
#define MASK_SIZE 17
#define INTS_PER_ROW 40  // ceil(1251 / 32) = 40

// 通过对每行增加 1 个 uint32 的填充，降低共享内存 bank 冲突
#define PADDED_INTS_PER_ROW (INTS_PER_ROW + 1)

// 外层 5 个分块（带重叠）用于全局维度覆盖
#define NUM_CHUNKS 5
#define MAX_CHUNK_ROWS 258

// 在每个大分块内部再做行方向的子分块（带重叠）以降低单块共享内存占用，提高占用率
// 144 行子瓦片，步长为 128 行，保证与 17x17 掩码的 16 行重叠
#define TILE_ROWS 144
#define TILE_STRIDE (TILE_ROWS - (MASK_SIZE - 1))

// 分块边界
__constant__ int chunk_starts[NUM_CHUNKS] = {0, 249, 498, 747, 996};
__constant__ int chunk_rows[NUM_CHUNKS] = {258, 258, 258, 258, 256};

// 直接使用预压缩的 mask（每行低 17 位有效）
__constant__ uint32_t d_mask_compressed[MASK_SIZE] = {
    1984,   // row 0: bits 6..10
    8176,   // row 1: bits 4..12
    16376,  // row 2: bits 3..13
    32764,  // row 3: bits 2..14
    65534,  // row 4: bits 1..15
    65534,  // row 5: bits 1..15
    131071, // row 6: bits 0..16
    130175, // row 7: bits 0..6 and 10..16
    130175, // row 8
    130175, // row 9
    131071, // row 10
    65534,  // row 11
    65534,  // row 12
    32764,  // row 13
    16376,  // row 14
    8176,   // row 15
    1984    // row 16
};

static __device__ __forceinline__ int32_t wrapping_mul_signed(int32_t a, int32_t b) {
    int64_t raw = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    return static_cast<int32_t>(raw);
}

static __device__ __forceinline__ uint8_t compute_mask(int64_t seed, int32_t x, int32_t z) {
    int32_t x_sq = wrapping_mul_signed(x, x);
    int64_t term1 = static_cast<int64_t>(wrapping_mul_signed(x_sq, 0x4C1906));
    int64_t term2 = static_cast<int64_t>(wrapping_mul_signed(x, 0x5AC0DB));
    int32_t z_sq = wrapping_mul_signed(z, z);
    int64_t term3 = static_cast<int64_t>(z_sq) * static_cast<int64_t>(0x4307A7);
    int64_t term4 = static_cast<int64_t>(wrapping_mul_signed(z, 0x5F24F));

    int64_t sum = seed + term1 + term2 + term3 + term4;
    int64_t mixed = sum ^ 0x3AD8025F;

    uint64_t s = (static_cast<uint64_t>(mixed) ^ 25214903917ULL) & 0x0000FFFFFFFFFFFFULL;
    s = (s * 25214903917ULL + 11ULL) & 0x0000FFFFFFFFFFFFULL;
    return static_cast<uint8_t>((s >> 17) % 10 == 0 ? 1 : 0);
}

// 生成并压缩一个子分块到共享内存（行向瓦片）
static __device__ __forceinline__ void generate_and_compress_tile(
    int64_t seed,
    int global_chunk_start_row,
    int tile_local_start_row,
    int tile_num_rows,
    uint32_t shared_data[][PADDED_INTS_PER_ROW]
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    for (int row = tid; row < tile_num_rows; row += total_threads) {
        int global_row = global_chunk_start_row + tile_local_start_row + row;
        if (global_row >= DIM) continue;

        int gz = global_row - DIM / 2;
        uint32_t* row_ptr = shared_data[row];

        for (int int_idx = 0; int_idx < INTS_PER_ROW; int_idx++) {
            uint32_t compressed = 0u;
            int base_col = int_idx * 32;

            // 展开到 32 位
            #pragma unroll
            for (int bit = 0; bit < 32; bit++) {
                int col = base_col + bit;
                if (col < DIM) {
                    int gx = col - DIM / 2;
                    uint8_t val = compute_mask(seed, gx, gz);
                    compressed |= (static_cast<uint32_t>(val) << bit);
                }
            }
            row_ptr[int_idx] = compressed;
        }

        // 行尾填充，避免相邻行同一 bank 重复映射
        row_ptr[INTS_PER_ROW] = 0u;
    }
}

// 从压缩数据中提取指定列范围的 17 位（使用 64 位窗口）
static __device__ __forceinline__ uint32_t extract_17_bits(
    const uint32_t* row_data,
    int start_col
) {
    int start_int = start_col >> 5;         // /32
    int start_bit = start_col & 31;         // %32

    uint64_t lo = static_cast<uint64_t>(row_data[start_int]);
    // 读下一段安全：最大需要 start_int+1（见维度分析），PADDED 已预留
    uint64_t hi = static_cast<uint64_t>(row_data[start_int + 1]);
    uint64_t window = lo | (hi << 32);
    return static_cast<uint32_t>((window >> start_bit) & ((1u << MASK_SIZE) - 1u));
}

// 在共享内存瓦片上应用 mask 并返回该瓦片的最大值
static __device__ __forceinline__ int compute_max_in_tile(
    uint32_t shared_data[][PADDED_INTS_PER_ROW],
    int tile_num_rows
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    int local_max = 0;

    int valid_rows = tile_num_rows - MASK_SIZE + 1;
    int valid_cols = DIM - MASK_SIZE + 1;
    if (valid_rows <= 0) return 0;

    int total_positions = valid_rows * valid_cols;
    for (int idx = tid; idx < total_positions; idx += total_threads) {
        int row = idx / valid_cols;
        int col = idx % valid_cols;

        int count = 0;
        #pragma unroll
        for (int mz = 0; mz < MASK_SIZE; mz++) {
            uint32_t mask_row = d_mask_compressed[mz];
            const uint32_t* row_ptr = shared_data[row + mz];
            uint32_t data_bits = extract_17_bits(row_ptr, col);
            count += __popc(data_bits & mask_row);
        }

        if (count > local_max) local_max = count;
    }

    return local_max;
}

// 采用共享内存行填充与行向子瓦片，提高吞吐
__global__ void process_seed_optimized_v2(
    const int64_t* __restrict__ seeds,
    int32_t* __restrict__ max_counts,
    int batch_size
) {
    // 144 行 × (40+1) 列 = 144 × 41 × 4B = 23616B < 24KB，允许更高并发
    __shared__ uint32_t shared_tile[TILE_ROWS][PADDED_INTS_PER_ROW];

    int batch_id = blockIdx.x;
    if (batch_id >= batch_size) return;

    int64_t seed = seeds[batch_id];
    int global_max = 0;

    // 遍历外层 5 个重叠大分块
    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
        int chunk_start = chunk_starts[chunk];
        int rows_in_chunk = chunk_rows[chunk];

        // 行向子瓦片，步长带 16 行重叠
        for (int tile_offset = 0; tile_offset < rows_in_chunk; tile_offset += TILE_STRIDE) {
            int tile_rows = rows_in_chunk - tile_offset;
            if (tile_rows > TILE_ROWS) tile_rows = TILE_ROWS;

            generate_and_compress_tile(seed, chunk_start, tile_offset, tile_rows, shared_tile);
            __syncthreads();

            int tile_max = compute_max_in_tile(shared_tile, tile_rows);
            if (tile_max > global_max) global_max = tile_max;
            __syncthreads();
        }
    }

    // warp 级规约 + 最少共享内存占用
    // 先 warp 内做最大值规约
    unsigned mask = 0xFFFFFFFFu;
    int val = global_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(mask, val, offset);
        if (other > val) val = other;
    }

    // 写入每个 warp 的结果到共享内存，再由 1 个 warp 完成最终规约
    __shared__ int warp_max[32];
    int lane = threadIdx.x & 31;
    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;
    if (lane == 0) warp_max[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        int final_val = (lane < (blockDim.x * blockDim.y + 31) / 32) ? warp_max[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            int other = __shfl_down_sync(mask, final_val, offset);
            if (other > final_val) final_val = other;
        }
        if (lane == 0) max_counts[batch_id] = final_val;
    }
}


static bool process_batch_async(int64_t* h_seeds, int32_t* h_results,
                                int64_t* d_seeds, int32_t* d_max_counts,
                                int batch_size, cudaStream_t stream) {
    cudaMemcpyAsync(d_seeds, h_seeds, sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_max_counts, 0, sizeof(int32_t) * batch_size, stream);

    dim3 grid(batch_size);
    dim3 block(16, 16);  // 256 线程
    process_seed_optimized_v2<<<grid, block, 0, stream>>>(d_seeds, d_max_counts, batch_size);

    cudaMemcpyAsync(h_results, d_max_counts, sizeof(int32_t) * batch_size, cudaMemcpyDeviceToHost, stream);
    return true;
}

static bool initialize_pinned_memory(int64_t*& h_seeds, int32_t*& h_results, int max_batch_size) {
    cudaError_t err1 = cudaHostAlloc(&h_seeds, sizeof(int64_t) * max_batch_size, cudaHostAllocDefault);
    cudaError_t err2 = cudaHostAlloc(&h_results, sizeof(int32_t) * max_batch_size, cudaHostAllocDefault);
    return (err1 == cudaSuccess && err2 == cudaSuccess);
}

static bool initialize_device_memory(int64_t*& d_seeds, int32_t*& d_max_counts, int max_batch_size) {
    cudaError_t err1 = cudaMalloc(&d_seeds, sizeof(int64_t) * max_batch_size);
    cudaError_t err2 = cudaMalloc(&d_max_counts, sizeof(int32_t) * max_batch_size);
    return (err1 == cudaSuccess && err2 == cudaSuccess);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <start_seed> <count>" << std::endl;
        return 1;
    }

    int64_t start_seed = atoll(argv[1]);
    int64_t seed_count = atoll(argv[2]);
    const int batch_size = 512;
    const int stream_count = 2;


    std::vector<cudaStream_t> streams(stream_count);
    for (int i = 0; i < stream_count; ++i) {
        if (cudaStreamCreate(&streams[i]) != cudaSuccess) {
            std::cerr << "Failed to create stream " << i << std::endl;
            return 1;
        }
    }

    std::vector<int64_t*> h_seeds(stream_count);
    std::vector<int32_t*> h_results(stream_count);
    std::vector<int64_t*> d_seeds(stream_count);
    std::vector<int32_t*> d_max_counts(stream_count);
    std::vector<int> batch_sizes(stream_count);

    for (int i = 0; i < stream_count; ++i) {
        if (!initialize_pinned_memory(h_seeds[i], h_results[i], batch_size) ||
            !initialize_device_memory(d_seeds[i], d_max_counts[i], batch_size)) {
            std::cerr << "Memory allocation failed for stream " << i << std::endl;
            return 1;
        }
    }

    std::ofstream file("results.csv", std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open results.csv" << std::endl;
        return 1;
    }

    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "seed,count\n";
    }

    auto total_start = std::chrono::high_resolution_clock::now();
    int64_t processed = 0;

    while (processed < seed_count) {
        for (int i = 0; i < stream_count && processed < seed_count; ++i) {
            int current_batch = std::min((int64_t)batch_size, seed_count - processed);
            batch_sizes[i] = current_batch;

            for (int j = 0; j < current_batch; ++j) {
                h_seeds[i][j] = start_seed + processed + j;
            }

            process_batch_async(h_seeds[i], h_results[i],
                                d_seeds[i], d_max_counts[i],
                                current_batch, streams[i]);

            processed += current_batch;
        }

        for (int i = 0; i < stream_count; ++i) {
            cudaStreamSynchronize(streams[i]);
            for (int j = 0; j < batch_sizes[i]; ++j) {
                file << h_seeds[i][j] << "," << h_results[i][j] << "\n";
            }
        }

        file.flush();

        if (processed % 10000 < batch_size * stream_count || processed == seed_count) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - total_start);
            double rate = elapsed.count() > 0 ? processed / (double)elapsed.count() : 0;

            std::cout << "Progress: " << processed << "/" << seed_count
                      << " (" << std::fixed << std::setprecision(1)
                      << (processed * 100.0 / seed_count) << "%), Avg: "
                      << std::fixed << std::setprecision(2) << rate << " seeds/sec" << std::endl;
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);

    std::cout << "\n=== Computation Complete ===" << std::endl;
    std::cout << "Processed: " << processed << " seeds" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl;
    std::cout << "Average: " << std::fixed << std::setprecision(2)
              << (processed / (double)total_duration.count()) << " seeds/sec" << std::endl;

    file.close();
    for (int i = 0; i < stream_count; ++i) {
        cudaFreeHost(h_seeds[i]);
        cudaFreeHost(h_results[i]);
        cudaFree(d_seeds[i]);
        cudaFree(d_max_counts[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}


