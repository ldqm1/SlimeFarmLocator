#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <cstdlib>

#define DIM 1251
#define TILE_SIZE 121
#define OVERLAP 8
#define EFFECTIVE_TILE 113
#define MASK_SIZE 17
#define MASK_RADIUS 8
#define OUTPUT_TILE (TILE_SIZE - MASK_SIZE + 1)
#define BLOCKS_PER_DIM ((DIM - MASK_RADIUS) / EFFECTIVE_TILE)

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

__global__ void process_tile(int64_t seed, int32_t* best_x, int32_t* best_z, int32_t* max_count) {
    __shared__ uint8_t tile[TILE_SIZE][128]; // 避免 bank conflict
    __shared__ int s_max[256];               // 每线程的最大值
    __shared__ int s_x[256];                 // 每线程的坐标
    __shared__ int s_z[256];

    int tile_x = blockIdx.x;
    int tile_z = blockIdx.y;
    int start_x = tile_x * EFFECTIVE_TILE;
    int start_z = tile_z * EFFECTIVE_TILE;
    int global_x = start_x - MASK_RADIUS;
    int global_z = start_z - MASK_RADIUS;

    int tx = threadIdx.x;
    int tz = threadIdx.y;
    int tid = tz * blockDim.x + tx;

    // 加载共享内存
    for (int dz = tz; dz < TILE_SIZE; dz += blockDim.y) {
        for (int dx = tx; dx < TILE_SIZE; dx += blockDim.x) {
            int gx = global_x + dx;
            int gz = global_z + dz;
            tile[dz][dx] = (gx >= 0 && gx < DIM && gz >= 0 && gz < DIM)
                ? compute_mask(seed, gx - DIM / 2, gz - DIM / 2)
                : 0;
        }
    }

    __syncthreads();

    int local_max = 0;
    int local_best_x = global_x;
    int local_best_z = global_z;

    for (int out_z = tz; out_z < OUTPUT_TILE; out_z += blockDim.y) {
        for (int out_x = tx; out_x < OUTPUT_TILE; out_x += blockDim.x) {
            int count = 0;
            for (int mz = 0; mz < MASK_SIZE; mz++) {
                for (int mx = 0; mx < MASK_SIZE; mx++) {
                    if (d_mask_pattern[mz * MASK_SIZE + mx]) {
                        count += tile[out_z + mz][out_x + mx];
                    }
                }
            }
            if (count > local_max) {
                local_max = count;
                local_best_x = global_x + out_x;
                local_best_z = global_z + out_z;
            }
        }
    }

    // 写入共享内存
    s_max[tid] = local_max;
    s_x[tid] = local_best_x - DIM / 2;
    s_z[tid] = local_best_z - DIM / 2;

    __syncthreads();

    // block 规约：线程0找出最大值
    if (tid == 0) {
        int block_max = s_max[0];
        int bx = s_x[0];
        int bz = s_z[0];
        for (int i = 1; i < blockDim.x * blockDim.y; i++) {
            if (s_max[i] > block_max) {
                block_max = s_max[i];
                bx = s_x[i];
                bz = s_z[i];
            }
        }

        // 原子更新全局最大值
        int old_max = atomicMax(max_count, block_max);
        if (block_max > old_max) {
            atomicExch(best_x, bx);
            atomicExch(best_z, bz);
        }
    }
}

bool process_single_seed(int64_t seed, int32_t& best_x, int32_t& best_z, int32_t& max_count) {
    int32_t *d_best_x, *d_best_z, *d_max_count;
    cudaMalloc(&d_best_x, sizeof(int32_t));
    cudaMalloc(&d_best_z, sizeof(int32_t));
    cudaMalloc(&d_max_count, sizeof(int32_t));
    cudaMemset(d_best_x, 0, sizeof(int32_t));
    cudaMemset(d_best_z, 0, sizeof(int32_t));
    cudaMemset(d_max_count, 0, sizeof(int32_t));

    dim3 grid(BLOCKS_PER_DIM, BLOCKS_PER_DIM);
    dim3 block(16, 16);

    process_tile<<<grid, block>>>(seed, d_best_x, d_best_z, d_max_count);
    cudaDeviceSynchronize();

    cudaMemcpy(&best_x, d_best_x, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&best_z, d_best_z, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_count, d_max_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_best_x);
    cudaFree(d_best_z);
    cudaFree(d_max_count);

    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <start_seed> <count>" << std::endl;
        return 1;
    }

    int64_t start_seed = atoll(argv[1]);
    int64_t seed_count = atoll(argv[2]);

    if (seed_count <= 0) {
        std::cerr << "Count must be positive" << std::endl;
        return 1;
    }

    std::cout << "Starting computation from seed " << start_seed 
              << " for " << seed_count << " seeds" << std::endl;
    std::cout << "Results will be appended to results.csv" << std::endl;

    std::ofstream file("results.csv", std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open results.csv for writing" << std::endl;
        return 1;
    }

    // 写入表头（如果文件为空）
    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "seed,x,z,count\n";
    }

    auto total_start = std::chrono::high_resolution_clock::now();
    int64_t processed_count = 0;

    for (int64_t i = 0; i < seed_count; i++) {
        int64_t current_seed = start_seed + i;
        int32_t best_x = 0, best_z = 0, max_count = 0;

        auto seed_start = std::chrono::high_resolution_clock::now();
        bool success = process_single_seed(current_seed, best_x, best_z, max_count);
        auto seed_end = std::chrono::high_resolution_clock::now();
        auto seed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(seed_end - seed_start);

        if (success) {
            file << current_seed << "," << best_x << "," << best_z << "," << max_count << "\n";
            file.flush();
            processed_count++;
        } else {
            std::cerr << "Failed to process seed " << current_seed << std::endl;
        }

        if ((i + 1) % 10000 == 0 || i == seed_count - 1) {
            auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - total_start);

            double avg_time = total_duration.count() * 1000.0 / (i + 1);
            double seeds_per_second = total_duration.count() > 0 ? (i + 1) / (double)total_duration.count() : 0;

            std::cout << "Progress: " << (i + 1) << "/" << seed_count 
                      << " seeds (" << std::fixed << std::setprecision(1) 
                      << ((i + 1) * 100.0 / seed_count) << "%)"
                      << ", Avg: " << std::fixed << std::setprecision(2) 
                      << seeds_per_second << " seeds/sec" << std::endl;
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);

    std::cout << "\n=== Computation Complete ===" << std::endl;
    std::cout << "Successfully processed " << processed_count << " seeds" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl;
    if (total_duration.count() > 0) {
        std::cout << "Average: " << (processed_count / (double)total_duration.count()) << " seeds/second" << std::endl;
    }

    file.close();
    return 0;
}