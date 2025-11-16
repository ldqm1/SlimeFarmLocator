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
#define MASK_RADIUS 8
#define INTS_PER_ROW 40  // ceil(1251 / 32) = 40

// 5个分块的定义（带重叠）
#define NUM_CHUNKS 5
#define MAX_CHUNK_ROWS 258

// 分块边界
__constant__ int chunk_starts[NUM_CHUNKS] = {0, 249, 498, 747, 996};
__constant__ int chunk_rows[NUM_CHUNKS] = {258, 258, 258, 258, 256};

// 原始 mask pattern
__constant__ uint8_t d_mask_pattern[MASK_SIZE * MASK_SIZE] = {
    0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
    0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
    0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
    0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0
};

// 压缩后的 mask：每行用一个 uint32_t 存储（低17位有效）
__constant__ uint32_t d_mask_compressed[MASK_SIZE];

__device__ int32_t wrapping_mul_signed(int32_t a, int32_t b) {
    int64_t raw = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    return static_cast<int32_t>(raw);
}

__device__ uint8_t compute_mask(int64_t seed, int32_t x, int32_t z) {
    int32_t x_sq = wrapping_mul_signed(x, x);
    int32_t term1_32 = wrapping_mul_signed(x_sq, 0x4C1906);
    int64_t term1 = static_cast<int64_t>(term1_32);

    int32_t term2_32 = wrapping_mul_signed(x, 0x5AC0DB);
    int64_t term2 = static_cast<int64_t>(term2_32);

    int32_t z_sq = wrapping_mul_signed(z, z);
    int64_t term3 = static_cast<int64_t>(z_sq) * static_cast<int64_t>(0x4307A7);

    int32_t term4_32 = wrapping_mul_signed(z, 0x5F24F);
    int64_t term4 = static_cast<int64_t>(term4_32);

    int64_t sum = seed + term1 + term2 + term3 + term4;
    int64_t mixed = sum ^ 0x3AD8025F;

    uint64_t s = (static_cast<uint64_t>(mixed) ^ 25214903917ULL) & 0x0000FFFFFFFFFFFFULL;
    s = (s * 25214903917ULL + 11ULL) & 0x0000FFFFFFFFFFFFULL;
    return static_cast<uint8_t>((s >> 17) % 10 == 0 ? 1 : 0);
}

// 生成并压缩一个分块到共享内存
__device__ void generate_and_compress_chunk(
    int64_t seed,
    int chunk_start_row,
    int chunk_num_rows,
    uint32_t shared_data[][INTS_PER_ROW]
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    // 每个线程生成多行
    for (int row = tid; row < chunk_num_rows; row += total_threads) {
        int global_row = chunk_start_row + row;
        if (global_row >= DIM) continue;

        int gz = global_row - DIM / 2;

        // 生成并压缩这一行
        for (int int_idx = 0; int_idx < INTS_PER_ROW; int_idx++) {
            uint32_t compressed = 0;
            for (int bit = 0; bit < 32; bit++) {
                int col = int_idx * 32 + bit;
                if (col < DIM) {
                    int gx = col - DIM / 2;
                    uint8_t val = compute_mask(seed, gx, gz);
                    compressed |= (static_cast<uint32_t>(val) << bit);
                }
            }
            shared_data[row][int_idx] = compressed;
        }
    }
}

// 从压缩数据中提取指定列范围的位
__device__ inline uint32_t extract_bits(
    const uint32_t* row_data,
    int start_col,
    int num_bits
) {
    int start_int = start_col / 32;
    int start_bit = start_col % 32;

    if (start_bit + num_bits <= 32) {
        // 所有位都在一个 uint32_t 中
        uint32_t data = row_data[start_int];
        uint32_t mask = (num_bits == 32) ? 0xFFFFFFFF : ((1u << num_bits) - 1);
        return (data >> start_bit) & mask;
    } else {
        // 跨越两个 uint32_t
        uint32_t data1 = row_data[start_int];
        uint32_t data2 = row_data[start_int + 1];
        int bits_from_first = 32 - start_bit;
        int bits_from_second = num_bits - bits_from_first;
        uint32_t part1 = data1 >> start_bit;
        uint32_t part2 = data2 & ((1u << bits_from_second) - 1);
        return part1 | (part2 << bits_from_first);
    }
}

// 在压缩数据上应用 mask 并计算最大值
__device__ int compute_max_in_chunk(
    uint32_t shared_data[][INTS_PER_ROW],
    int chunk_start_row,
    int chunk_num_rows
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    int local_max = 0;

    // 可以应用 mask 的有效区域
    int valid_rows = chunk_num_rows - MASK_SIZE + 1;
    int valid_cols = DIM - MASK_SIZE + 1;

    if (valid_rows <= 0) return 0;

    // 每个线程处理多个位置
    for (int idx = tid; idx < valid_rows * valid_cols; idx += total_threads) {
        int row = idx / valid_cols;
        int col = idx % valid_cols;

        // 应用 mask 计数
        int count = 0;
        for (int mz = 0; mz < MASK_SIZE; mz++) {
            uint32_t mask_row = d_mask_compressed[mz];
            int data_row = row + mz;

            // 提取数据行中从 col 开始的 17 位
            uint32_t data_bits = extract_bits(shared_data[data_row], col, MASK_SIZE);

            // 位与并计数
            count += __popc(data_bits & mask_row);
        }

        if (count > local_max) {
            local_max = count;
        }
    }

    return local_max;
}

__global__ void process_seed_optimized(
    const int64_t* seeds,
    int32_t* max_counts,
    int batch_size
) {
    // 共享内存：最大 258 行 × 40 列 = 10320 个 uint32_t = 41280 字节 < 48KB
    __shared__ uint32_t shared_chunk[MAX_CHUNK_ROWS][INTS_PER_ROW];
    __shared__ int s_max[256];

    int batch_id = blockIdx.x;
    if (batch_id >= batch_size) return;

    int64_t seed = seeds[batch_id];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int global_max = 0;

    // 处理 5 个分块
    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
        int start_row = chunk_starts[chunk];
        int num_rows = chunk_rows[chunk];

        // 生成并压缩分块到共享内存
        generate_and_compress_chunk(seed, start_row, num_rows, shared_chunk);
        __syncthreads();

        // 在该分块上应用 mask 并计算最大值
        int chunk_max = compute_max_in_chunk(shared_chunk, start_row, num_rows);
        if (chunk_max > global_max) {
            global_max = chunk_max;
        }
        __syncthreads();
    }

    // 规约：收集所有线程的 local max
    s_max[tid] = global_max;
    __syncthreads();

    // 树形规约找到全局最大值
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_counts[batch_id] = s_max[0];
    }
}

void initialize_compressed_mask() {
    // 从 GPU 读取原始 mask
    uint8_t h_mask[MASK_SIZE * MASK_SIZE];
    cudaMemcpyFromSymbol(h_mask, d_mask_pattern, MASK_SIZE * MASK_SIZE);

    // 压缩每一行
    uint32_t h_mask_compressed[MASK_SIZE];
    for (int row = 0; row < MASK_SIZE; row++) {
        uint32_t compressed = 0;
        for (int col = 0; col < MASK_SIZE; col++) {
            if (h_mask[row * MASK_SIZE + col]) {
                compressed |= (1u << col);
            }
        }
        h_mask_compressed[row] = compressed;
    }

    // 拷贝到 GPU 常量内存
    cudaMemcpyToSymbol(d_mask_compressed, h_mask_compressed, MASK_SIZE * sizeof(uint32_t));
}

bool process_batch_async(int64_t* h_seeds, int32_t* h_results,
                         int64_t* d_seeds, int32_t* d_max_counts,
                         int batch_size, cudaStream_t stream) {
    cudaMemcpyAsync(d_seeds, h_seeds, sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_max_counts, 0, sizeof(int32_t) * batch_size, stream);

    // 每个 block 处理一个 seed
    dim3 grid(batch_size);
    dim3 block(16, 16);  // 256 个线程
    process_seed_optimized<<<grid, block, 0, stream>>>(d_seeds, d_max_counts, batch_size);

    cudaMemcpyAsync(h_results, d_max_counts, sizeof(int32_t) * batch_size, cudaMemcpyDeviceToHost, stream);
    return true;
}

bool initialize_pinned_memory(int64_t*& h_seeds, int32_t*& h_results, int max_batch_size) {
    cudaError_t err1 = cudaHostAlloc(&h_seeds, sizeof(int64_t) * max_batch_size, cudaHostAllocDefault);
    cudaError_t err2 = cudaHostAlloc(&h_results, sizeof(int32_t) * max_batch_size, cudaHostAllocDefault);
    return (err1 == cudaSuccess && err2 == cudaSuccess);
}

bool initialize_device_memory(int64_t*& d_seeds, int32_t*& d_max_counts, int max_batch_size) {
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

    // 初始化压缩的 mask
    initialize_compressed_mask();

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
