// #define ENABLE_NVTX
// #define ENABLE_CPU_BASELINE
// #define DEBUG_PRINTS
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#define ENABLE_CORRECTNESS_CHECK

#define EXIT_INCORRECT_DISTANCES 10

#include <cuda_runtime.h>
#include <stdio.h>

#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include "../distributed_mmio/include/mmio.h"
#include "../distributed_mmio/include/mmio_utils.h"

#include "../include/bfs_baseline.cuh"
#include "../include/cli.hpp"
#include "../include/colors.h"
#include "../include/mt19937-64.hpp"
#include "../include/utils.cuh"

#define BLOCK_SIZE 512
#define T 0.1
#define A 3
#define B 24

void printInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads Per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max Grid Size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Max Blocks Per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory Per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers Per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;

        // Runtime limits for dynamic parallelism
        size_t stackSize, syncDepth;
        cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
        cudaDeviceGetLimit(&syncDepth, cudaLimitDevRuntimeSyncDepth);

        std::cout << "  Device Runtime Stack Size: " << stackSize << " bytes" << std::endl;
        std::cout << "  Device Runtime Sync Depth: " << syncDepth << std::endl;

        std::cout << std::endl;
    }
}

__global__ void bfs_topdown_kernel(const uint32_t *row_offsets,  // CSR row offsets
                                   const uint32_t *col_indices,  // CSR column indices (neighbors)
                                   int *distances,               // Output distances array
                                   uint32_t *frontier,           // Current frontier
                                   uint32_t *next_frontier,      // Next frontier to populate
                                   uint32_t frontier_size,       // Size of current frontier
                                   int current_level,            // BFS level (depth)
                                   uint32_t *next_frontier_size, // Counter for next frontier
                                   uint32_t num_vertices,        // Total number of vertices
                                   uint32_t num_edges,           // Total number of edges
                                   uint32_t *global_barrier_counter);

__device__ void gpuSleep(clock_t sleep_cycles) {
    clock_t start = clock();
    clock_t now;
    do {
        now = clock();
    } while (now - start < sleep_cycles);
}

/**
 * @brief Process neighbors with unrolling optimization up to 8 neighbors
 *
 * This function uses loop unrolling to process up to 8 neighbors simultaneously,
 * improving instruction-level parallelism and reducing loop overhead. The unrolling
 * strategy uses a cascading approach: 8-way -> 4-way -> 2-way -> 1-way processing
 * to handle all remaining neighbors efficiently.
 *
 * @param col_indices CSR column indices (neighbor list)
 * @param distances Output distances array
 * @param next_frontier Next frontier to populate
 * @param next_frontier_size Counter for next frontier size
 * @param current_level Current BFS level
 * @param row_start Start index of neighbor list
 * @param row_end End index of neighbor list
 */
__device__ __forceinline__ void process_neighbors_unrolled(
    const uint32_t *col_indices,
    int *distances,
    uint32_t *next_frontier,
    uint32_t *next_frontier_size,
    int current_level,
    uint32_t row_start,
    uint32_t row_end) {

    uint32_t i = row_start;

    // Primary 8-way unrolling loop - processes neighbors in chunks of 8
    while (i + 8 <= row_end) {
        uint32_t neighbor0 = col_indices[i];
        uint32_t neighbor1 = col_indices[i + 1];
        uint32_t neighbor2 = col_indices[i + 2];
        uint32_t neighbor3 = col_indices[i + 3];
        uint32_t neighbor4 = col_indices[i + 4];
        uint32_t neighbor5 = col_indices[i + 5];
        uint32_t neighbor6 = col_indices[i + 6];
        uint32_t neighbor7 = col_indices[i + 7];

        if (atomicCAS(&distances[neighbor0], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor0;
        }
        if (atomicCAS(&distances[neighbor1], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor1;
        }
        if (atomicCAS(&distances[neighbor2], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor2;
        }
        if (atomicCAS(&distances[neighbor3], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor3;
        }
        if (atomicCAS(&distances[neighbor4], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor4;
        }
        if (atomicCAS(&distances[neighbor5], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor5;
        }
        if (atomicCAS(&distances[neighbor6], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor6;
        }
        if (atomicCAS(&distances[neighbor7], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor7;
        }
        i += 8;
    }

    // Handle remaining 4-7 neighbors with 4-way unrolling
    if (i + 4 <= row_end) {
        uint32_t neighbor0 = col_indices[i];
        uint32_t neighbor1 = col_indices[i + 1];
        uint32_t neighbor2 = col_indices[i + 2];
        uint32_t neighbor3 = col_indices[i + 3];

        if (atomicCAS(&distances[neighbor0], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor0;
        }
        if (atomicCAS(&distances[neighbor1], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor1;
        }
        if (atomicCAS(&distances[neighbor2], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor2;
        }
        if (atomicCAS(&distances[neighbor3], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor3;
        }
        i += 4;
    }

    // Handle remaining 2-3 neighbors with 2-way unrolling
    if (i + 2 <= row_end) {
        uint32_t neighbor0 = col_indices[i];
        uint32_t neighbor1 = col_indices[i + 1];

        if (atomicCAS(&distances[neighbor0], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor0;
        }
        if (atomicCAS(&distances[neighbor1], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor1;
        }
        i += 2;
    }

    // Handle remaining single neighbor
    if (i < row_end) {
        uint32_t neighbor = col_indices[i];
        if (atomicCAS(&distances[neighbor], -1, current_level + 1) == -1) {
            uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[index] = neighbor;
        }
    }
}



__device__ __inline__ void hybrid_work(const uint32_t *row_offsets,  // CSR row offsets
                                       const uint32_t *col_indices,  // CSR column indices (neighbors)
                                       int *distances,               // Output distances array
                                       uint32_t *frontier,           // Current frontier
                                       uint32_t *next_frontier,      // Next frontier to populate
                                       uint32_t frontier_size,       // Size of current frontier
                                       int current_level,            // BFS level (depth)
                                       uint32_t *next_frontier_size, // Counter for next frontier
                                       uint32_t num_vertices,        // Total number of vertices
                                       uint32_t num_edges,           // Total number of edges
                                       bool top_down, uint32_t *n_frontier_edges, uint32_t *global_barrier_counter) {

    // Heuristic to choose between top-down and bottom-up
    // Use top-down when frontier is small, bottom-up when frontier is large
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (top_down) {
        // Top-down approach: iterate over frontier vertices
        if (tid >= frontier_size)
            return;

        uint32_t node = frontier[tid];
        uint32_t row_start = row_offsets[node];
        uint32_t row_end = row_offsets[node + 1];

        atomicAdd(n_frontier_edges, row_end - row_start);

        // Use optimized unrolled neighbor processing for maximum throughput
        process_neighbors_unrolled(col_indices, distances, next_frontier, next_frontier_size,
                                 current_level, row_start, row_end);
    } else {
        // Bottom-up approach: iterate over all unvisited vertices
        // Only process unvisited vertices
        if (tid >= num_vertices || atomicAdd(&distances[tid], 0) != -1) {
            return;
        }

        uint32_t row_start = row_offsets[tid];
        uint32_t row_end = row_offsets[tid + 1];

        // Check if any neighbor is in the current frontier (at current_level)
        for (uint32_t i = row_start; i < row_end; i++) {
            uint32_t neighbor = col_indices[i];

            if (atomicAdd(&distances[neighbor], 0) == current_level) {
                // Found a neighbor in current frontier, add this vertex to next frontier
                if (atomicCAS(&distances[tid], -1, current_level + 1) == -1) {
                    uint32_t index = atomicAdd(next_frontier_size, 1);
                    next_frontier[index] = tid;

                    uint32_t row_start = row_offsets[tid];
                    uint32_t row_end = row_offsets[tid + 1];
                    atomicAdd(n_frontier_edges, row_end - row_start);
                    break; // No need to check other neighbors
                }
            }
        }
    }
}

__device__ __inline__ void topdown_work(const uint32_t *row_offsets,  // CSR row offsets
                                        const uint32_t *col_indices,  // CSR column indices (neighbors)
                                        int *distances,               // Output distances array
                                        uint32_t *frontier,           // Current frontier
                                        uint32_t *next_frontier,      // Next frontier to populate
                                        uint32_t frontier_size,       // Size of current frontier
                                        int current_level,            // BFS level (depth)
                                        uint32_t *next_frontier_size, // Counter for next frontier
                                        uint32_t num_vertices,        // Total number of vertices
                                        uint32_t num_edges,           // Total number of edges
                                        uint32_t *global_barrier_counter) {
    // Heuristic to choose between top-down and bottom-up
    // Use top-down when frontier is small, bottom-up when frontier is large

    // Top-down approach: iterate over frontier vertices
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size)
        return;

    // uint32_t next_level = current_level + 1;
    uint32_t node = frontier[tid];
    uint32_t row_start = row_offsets[node];
    uint32_t row_end = row_offsets[node + 1];

    // Use optimized unrolled neighbor processing for top-down traversal
    process_neighbors_unrolled(col_indices, distances, next_frontier, next_frontier_size,
                             current_level, row_start, row_end);
}

__device__ __inline__ void die(const uint32_t *row_offsets,  // CSR row offsets
                               const uint32_t *col_indices,  // CSR column indices (neighbors)
                               int *distances,               // Output distances array
                               uint32_t *frontier,           // Current frontier
                               uint32_t *next_frontier,      // Next frontier to populate
                               uint32_t frontier_size,       // Size of current frontier
                               int current_level,            // BFS level (depth)
                               uint32_t *next_frontier_size, // Counter for next frontier
                               uint32_t num_vertices,        // Total number of vertices
                               uint32_t num_edges,           // Total number of edges
                               uint32_t *global_barrier_counter) {
    if (threadIdx.x == 0) {
        atomicAdd(global_barrier_counter, 1);
    }
}

__global__ void bfs_topdown_kernel(const uint32_t *row_offsets,  // CSR row offsets
                                   const uint32_t *col_indices,  // CSR column indices (neighbors)
                                   int *distances,               // Output distances array
                                   uint32_t *frontier,           // Current frontier
                                   uint32_t *next_frontier,      // Next frontier to populate
                                   uint32_t frontier_size,       // Size of current frontier
                                   int current_level,            // BFS level (depth)
                                   uint32_t *next_frontier_size, // Counter for next frontier
                                   uint32_t num_vertices,        // Total number of vertices
                                   uint32_t num_edges,           // Total number of edges
                                   uint32_t *global_barrier_counter) {
    topdown_work(row_offsets, col_indices, distances, frontier, next_frontier, frontier_size, current_level,
                 next_frontier_size, num_vertices, num_edges, global_barrier_counter);
    __syncthreads();
    die(row_offsets, col_indices, distances, frontier, next_frontier, frontier_size, current_level, next_frontier_size,
        num_vertices, num_edges, global_barrier_counter);
}

__global__ void bfs_hybrid_kernel(const uint32_t *row_offsets,  // CSR row offsets
                                  const uint32_t *col_indices,  // CSR column indices (neighbors)
                                  int *distances,               // Output distances array
                                  uint32_t *frontier,           // Current frontier
                                  uint32_t *next_frontier,      // Next frontier to populate
                                  uint32_t frontier_size,       // Size of current frontier
                                  int current_level,            // BFS level (depth)
                                  uint32_t *next_frontier_size, // Counter for next frontier
                                  uint32_t num_vertices,        // Total number of vertices
                                  uint32_t num_edges,           // Total number of edges
                                  bool top_down,                // Top-down or bottom-up traversal
                                  uint32_t *n_frontier_edges,   // Number of edges in the frontier
                                  uint32_t *global_barrier_counter) {
    hybrid_work(row_offsets, col_indices, distances, frontier, next_frontier, frontier_size, current_level,
                next_frontier_size, num_vertices, num_edges, top_down, n_frontier_edges, global_barrier_counter);
    __syncthreads();
    die(row_offsets, col_indices, distances, frontier, next_frontier, frontier_size, current_level, next_frontier_size,
        num_vertices, num_edges, global_barrier_counter);
}

__global__ void bfs_hybrid_launcher(const uint32_t *row_offsets,  // CSR row offsets
                                    const uint32_t *col_indices,  // CSR column indices (neighbors)
                                    int *distances,               // Output distances array
                                    uint32_t *frontier,           // Current frontier
                                    uint32_t *next_frontier,      // Next frontier to populate
                                    uint32_t frontier_size,       // Size of current frontier
                                    int current_level,            // BFS level (depth)
                                    uint32_t *next_frontier_size, // Counter for next frontier
                                    uint32_t num_vertices,        // Total number of vertices
                                    uint32_t num_edges,           // Total number of edges
                                    uint32_t *n_frontier_edges,   // Number of edges in the frontier
                                    uint32_t *global_barrier_counter) {
    // bool top_down = true;
    uint32_t node = frontier[0];
    uint32_t row_start = row_offsets[node];
    uint32_t row_end = row_offsets[node + 1];
    *n_frontier_edges = row_end - row_start;
    // uint32_t edges_from_unexplored_vertices = num_edges - *n_frontier_edges;
    uint32_t edges_from_unexplored_vertices = num_edges;
    uint32_t top_down = true;
    do {
        if (top_down) {
            if (*n_frontier_edges > edges_from_unexplored_vertices / A) {
                top_down = false;
                // printf("Switching to bottom-up on level %d\n", current_level);
            }
        } else {
            if (frontier_size < num_vertices / B) {
                top_down = true;
                // printf("Switching to top-down on level %d\n", current_level);
            }
        }
        // printf("level %d: _frontier_edges: %d, edges_from_unexplored_vertices: %d\n", current_level, *n_frontier_edges, edges_from_unexplored_vertices);
        uint32_t num_blocks = CEILING(top_down ? frontier_size : num_vertices, BLOCK_SIZE);
        *next_frontier_size = 0;

        // uint32_t prev_n_frontier_edges = *n_frontier_edges;
        *n_frontier_edges = 0;

        // printf("Lauching kernel: top_down %d, frontier_size %d, current_level: %d, num_blocks: %d, "
        //        "num_vertices: %d\n",
        //        top_down, frontier_size, current_level, num_blocks, num_vertices);
        bfs_hybrid_kernel<<<num_blocks, BLOCK_SIZE, 0, 0>>>(
            row_offsets, col_indices, distances, frontier, next_frontier, frontier_size, current_level,
            next_frontier_size, num_vertices, num_edges, top_down, n_frontier_edges, global_barrier_counter);

        while (atomicAdd(global_barrier_counter, 0) < num_blocks)
            ;
        *global_barrier_counter = 0;

        edges_from_unexplored_vertices -= *n_frontier_edges;

        // Swap frontiers
        uint32_t *temp = frontier;
        frontier = next_frontier;
        next_frontier = temp;

        frontier_size = *next_frontier_size;
        // printf("Frontier size: %d\n", frontier_size);
        // printf("Current frontier at level %d (size %d): ", current_level, frontier_size);
        // for (uint32_t i = 0; i < frontier_size; i++) {
        //     printf("%d ", frontier[i]);
        // }
        // printf("\n");
        current_level++;
    } while (frontier_size != 0);
}

__global__ void bfs_topdown_launcher(const uint32_t *row_offsets,  // CSR row offsets
                                     const uint32_t *col_indices,  // CSR column indices (neighbors)
                                     int *distances,               // Output distances array
                                     uint32_t *frontier,           // Current frontier
                                     uint32_t *next_frontier,      // Next frontier to populate
                                     uint32_t frontier_size,       // Size of current frontier
                                     int current_level,            // BFS level (depth)
                                     uint32_t *next_frontier_size, // Counter for next frontier
                                     uint32_t num_vertices,        // Total number of vertices
                                     uint32_t num_edges,           // Total number of edges
                                     uint32_t *global_barrier_counter) {
    do {
        uint32_t num_blocks = CEILING(frontier_size, BLOCK_SIZE);
        *next_frontier_size = 0;

        // printf("Lauching kernel: frontier_size %d, current_level: %d, num_blocks: %d, BLOCK_SIZE: %d, "
        //        "num_vertices: %d\n",
        //        frontier_size, current_level, num_blocks, BLOCK_SIZE, num_vertices);
        bfs_topdown_kernel<<<num_blocks, BLOCK_SIZE, 0, 0>>>(
            row_offsets, col_indices, distances, frontier, next_frontier, frontier_size, current_level,
            next_frontier_size, num_vertices, num_edges, global_barrier_counter);

        // printf("Thread %d: Imma wait for everyone else\n", tid);
        while (atomicAdd(global_barrier_counter, 0) < num_blocks)
            ;
        *global_barrier_counter = 0;

        // Swap frontiers
        uint32_t *temp = frontier;
        frontier = next_frontier;
        next_frontier = temp;

        frontier_size = *next_frontier_size;
        // printf("Frontier size: %d\n", frontier_size);
        // printf("Current frontier at level %d (size %d): ", current_level, frontier_size);
        // for (uint32_t i = 0; i < frontier_size; i++) {
        //     printf("%d ", frontier[i]);
        // }
        // printf("\n");
        current_level++;
    } while (frontier_size != 0);
}

void gpu_bfs(const uint32_t N,         // Number of veritices
             const uint32_t M,         // Number of edges
             const uint32_t *h_rowptr, // Graph CSR rowptr
             const uint32_t *h_colidx, // Graph CSR colidx
             const uint32_t source,    // Source veritex
             int *h_distances,         // Write here your distances
             Matrix_Metadata *meta) {
    /***********************
     * IMPLEMENT HERE YOUR CUDA BFS
     * Feel free to structure you code (i.e. create other files, macros etc.)
     * *********************/

    // !! This is just a placeholder !!
    // gpu_bfs_baseline(N, M, h_rowptr, h_colidx, source, h_distances, true);

    // printInfo();

    /* Preprocessing */
    CPU_TIMER_INIT(BFS_preprocess)

    // Allocate and copy graph to device
    uint32_t *d_row_offsets, *d_col_indices;

    CHECK_CUDA(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_col_indices, M * sizeof(uint32_t)));

    CHECK_CUDA(cudaMemcpyAsync(d_row_offsets, h_rowptr, (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(d_col_indices, h_colidx, M * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Allocate memory for distances and frontier queues
    int *d_distances;
    uint32_t *d_frontier, *d_next_frontier, *d_next_frontier_size;
    uint32_t *d_frontier_edges, *d_global_barrier_counter;

    CHECK_CUDA(cudaMalloc(&d_distances, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_frontier, N * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_next_frontier, N * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_frontier_edges, sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_global_barrier_counter, sizeof(uint32_t)));

    // Initialize distances to -1
    CHECK_CUDA(cudaMemsetAsync(d_distances, 0xFF, N * sizeof(int))); // -1 is 0xFFFFFFFF

    // Initialize frontier with source
    CHECK_CUDA(cudaMemcpyAsync(d_frontier, &source, sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Set source distance to 0 using kernel (since cudaMemset won't work for 0-sized int)
    int zero = 0;
    CHECK_CUDA(cudaMemcpyAsync(&d_distances[source], &zero, sizeof(int), cudaMemcpyHostToDevice));

    float tot_time = 0.0f;
    uint32_t current_frontier_size = 1;
    int32_t level = 0;

    CHECK_CUDA(cudaDeviceSynchronize());
    CPU_TIMER_STOP(BFS_preprocess)
    tot_time += CPU_TIMER_ELAPSED(BFS_preprocess);
    CPU_TIMER_PRINT(BFS_preprocess)


    /* Kernel */
    CPU_TIMER_INIT(BFS)

    // printf("Launching kernel with source %d\n", source);
    if (meta->is_symmetric) {
        bfs_hybrid_launcher<<<1, 1, 0, 0>>>(d_row_offsets, d_col_indices, d_distances, d_frontier, d_next_frontier,
                                            current_frontier_size, level, d_next_frontier_size, N, M, d_frontier_edges,
                                            d_global_barrier_counter);
    } else {
        bfs_topdown_launcher<<<1, 1, 0, 0>>>(d_row_offsets, d_col_indices, d_distances, d_frontier, d_next_frontier,
                                             current_frontier_size, level, d_next_frontier_size, N, M,
                                             d_global_barrier_counter);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    CPU_TIMER_STOP(BFS)
    tot_time += CPU_TIMER_ELAPSED(BFS);
    CPU_TIMER_PRINT(BFS)

    /* Postprocessing */
    CPU_TIMER_INIT(BFS_postprocess)

    CHECK_CUDA(cudaMemcpy(h_distances, d_distances, N * sizeof(int), cudaMemcpyDeviceToHost));

    CPU_TIMER_STOP(BFS_postprocess)
    tot_time += CPU_TIMER_ELAPSED(BFS_postprocess);
    CPU_TIMER_PRINT(BFS_postprocess)

    CHECK_CUDA(cudaFree(d_row_offsets));
    CHECK_CUDA(cudaFree(d_col_indices));
    CHECK_CUDA(cudaFree(d_distances));
    CHECK_CUDA(cudaFree(d_frontier));
    CHECK_CUDA(cudaFree(d_next_frontier));
    CHECK_CUDA(cudaFree(d_next_frontier_size));
    CHECK_CUDA(cudaFree(d_global_barrier_counter));

    // This output format is MANDATORY, DO NOT CHANGE IT
    printf("\n[OUT] Total BFS time: %f ms\n" RESET, tot_time);
}

void print_graph_csr(GraphCSR *graph, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file");
        return;
    }

    fprintf(file, "Graph in MTX format (source, target):\n");
    for (uint32_t i = 0; i < graph->num_vertices; i++) {
        uint32_t start = graph->row_ptr[i];
        uint32_t end = graph->row_ptr[i + 1];
        for (uint32_t j = start; j < end; j++) {
            uint32_t neighbor = graph->col_idx[j];
            fprintf(file, "%u, %u\n", i, neighbor);
        }
    }

    fclose(file);
}

int main(int argc, char **argv) {
    int return_code = EXIT_SUCCESS;

    Cli_Args args;
    init_cli();
    if (parse_args(argc, argv, &args) != 0) {
        return -1;
    }

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0) {
        fprintf(stderr, "No GPU available: device_count=%d\n", device_count);
        return EXIT_FAILURE;
    }
    cudaSetDevice(0);

    Matrix_Metadata meta;
    CPU_TIMER_INIT(MTX_read)
    CSR_local<uint32_t, float> *csr = Distr_MMIO_CSR_local_read<uint32_t, float>(args.filename, false, &meta);
    printf("Is symmetric: %d\n", meta.is_symmetric);

    if (csr == NULL) {
        printf("Failed to import graph from file [%s]\n", args.filename);
        return -1;
    }
    CPU_TIMER_STOP(MTX_read)
    printf("\n[OUT] MTX file read time: %f ms\n", CPU_TIMER_ELAPSED(MTX_read));
    printf("Graph size: %.3fM vertices, %.3fM edges\n", csr->nrows / 1e6, csr->nnz / 1e6);

    GraphCSR graph;
    graph.row_ptr = csr->row_ptr;
    graph.col_idx = csr->col_idx;
    graph.num_vertices = csr->nrows;
    graph.num_edges = csr->nnz;
    // print_graph_csr(&graph, "bosta");

    uint32_t *sources = generate_sources(&graph, args.runs, graph.num_vertices, args.source);
    int *distances_gpu_baseline = (int *)malloc(graph.num_vertices * sizeof(int));
    int *distances = (int *)malloc(graph.num_vertices * sizeof(int));
    bool correct = true;

    for (int source_i = 0; source_i < args.runs; source_i++) {
        uint32_t source = sources[source_i];
        printf("\n[OUT] -- BFS iteration #%u, source=%u --\n", source_i, source);

        // Run the BFS baseline
        gpu_bfs_baseline(graph.num_vertices, graph.num_edges, graph.row_ptr, graph.col_idx, source,
                         distances_gpu_baseline, false);

#ifdef ENABLE_NVTX
        nvtxRangePushA("Complete BFS");
#endif
        gpu_bfs(graph.num_vertices, graph.num_edges, graph.row_ptr, graph.col_idx, source, distances, &meta);
#ifdef ENABLE_NVTX
        nvtxRangePop();
#endif

        bool match = true;
#ifdef ENABLE_CORRECTNESS_CHECK
        for (uint32_t i = 0; i < graph.num_vertices; ++i) {
            if (distances_gpu_baseline[i] != distances[i]) {
                printf("Mismatch at node %u: Baseline distance = %d, Your distance = %d\n", i,
                       distances_gpu_baseline[i], distances[i]);
                match = false;
                // break;
            }
        }
        if (match) {
            printf(BRIGHT_GREEN "Correctness OK\n" RESET);
        } else {
            printf(BRIGHT_RED "GPU and CPU BFS results do not match for source node %u.\n" RESET, source);
            return_code = EXIT_INCORRECT_DISTANCES;
            correct = false;
        }
#endif

#ifdef ENABLE_CPU_BASELINE
        int cpu_distances[graph.num_vertices];

        CPU_TIMER_INIT(CPU_BFS)
        cpu_bfs_baseline(graph.num_vertices, graph.row_ptr, graph.col_idx, source, cpu_distances);
        CPU_TIMER_CLOSE(CPU_BFS)

        match = true;
        for (uint32_t i = 0; i < graph.num_vertices; ++i) {
            if (distances_gpu_baseline[i] != cpu_distances[i]) {
                printf("Mismatch at node %u: GPU distance = %d, CPU distance = %d\n", i, distances_gpu_baseline[i],
                       cpu_distances[i]);
                match = false;
                break;
            }
        }
        if (match) {
            printf(BRIGHT_GREEN "[CPU] Correctness OK\n" RESET);
        } else {
            printf(BRIGHT_RED "GPU and CPU BFS results do not match for source node %u.\n" RESET, source);
            return_code = EXIT_INCORRECT_DISTANCES;
        }
#endif
    }

    if (correct)
        printf("\n[OUT] ALL RESULTS ARE CORRECT\n");
    else
        printf(BRIGHT_RED "\nSOME RESULTS ARE WRONG\n" RESET);

    Distr_MMIO_CSR_local_destroy(&csr);
    free(sources);
    free(distances_gpu_baseline);
    free(distances);

    return return_code;
}
