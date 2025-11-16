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

__global__ void process_tile_batch(const int64_t* seeds, int32_t* max_counts, int batch_size) {
    __shared__ uint8_t tile[TILE_SIZE][128];
    __shared__ int s_max[256];

    int tile_x = blockIdx.x;
    int tile_z = blockIdx.y;
    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int64_t seed = seeds[batch_id];
    int start_x = tile_x * EFFECTIVE_TILE;
    int start_z = tile_z * EFFECTIVE_TILE;
    int global_x = start_x - MASK_RADIUS;
    int global_z = start_z - MASK_RADIUS;

    int tx = threadIdx.x;
    int tz = threadIdx.y;
    int tid = tz * blockDim.x + tx;

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
            }
        }
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(&max_counts[batch_id], s_max[0]);
    }
}

bool initialize_streams(std::vector<cudaStream_t>& streams, int stream_count) {
    streams.resize(stream_count);
    for (int i = 0; i < stream_count; ++i) {
        if (cudaStreamCreate(&streams[i]) != cudaSuccess) return false;
    }
    return true;
}

bool process_batch_async(int64_t* h_seeds, int32_t* h_results,
                         int64_t* d_seeds, int32_t* d_max_counts,
                         int batch_size, cudaStream_t stream) {
    cudaMemcpyAsync(d_seeds, h_seeds, sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_max_counts, 0, sizeof(int32_t) * batch_size, stream);

    dim3 grid(BLOCKS_PER_DIM, BLOCKS_PER_DIM, batch_size);
    dim3 block(16, 16);
    process_tile_batch<<<grid, block, 0, stream>>>(d_seeds, d_max_counts, batch_size);

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