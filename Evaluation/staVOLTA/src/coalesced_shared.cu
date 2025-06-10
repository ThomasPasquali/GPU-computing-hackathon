#include "../include/utils.cuh"
#include <vector>
// Kernel: Process each node in the frontier and add unvisited neighbors to
// next_frontier
#define SHARED_BUFFER_SIZE 10000 // Adjusted for shared memory size
__global__ void bfs_kernel_coalesced_shared(
    const uint32_t *row_offsets, const uint32_t *col_indices, int *distances,
    const uint32_t *frontier, uint32_t *next_frontier, uint32_t frontier_size,
    uint32_t current_level, uint32_t *next_frontier_size) {
  __shared__ uint32_t shared_buffer[SHARED_BUFFER_SIZE];
  __shared__ uint32_t shared_buffer_size;
  __shared__ uint32_t start_index;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    shared_buffer_size =
        0; // Reset shared buffer size at the start of each block
  }
  __syncthreads();
  if (tid >= frontier_size)
    return;

  uint32_t node = frontier[tid];
  uint32_t row_start = row_offsets[node];
  uint32_t row_end = row_offsets[node + 1];

  for (uint32_t i = row_start + threadIdx.y; i < row_end; i += blockDim.y) {
    uint32_t neighbor = col_indices[i];
    if (atomicCAS(&distances[neighbor], -1, current_level + 1) == -1) {
      uint32_t index_shared = atomicAdd(&shared_buffer_size, 1);
      if (index_shared < SHARED_BUFFER_SIZE) {
        shared_buffer[index_shared] = neighbor;
      } else {
        // If shared buffer is full, flush it to next_frontier
        uint32_t index = atomicAdd(next_frontier_size, 1);
        next_frontier[index] = neighbor;
      }
    }
  }
  __syncthreads();
    uint32_t flush_size = min(shared_buffer_size, SHARED_BUFFER_SIZE);
    if (threadIdx.y == 0 && threadIdx.x == 0) {
      start_index = atomicAdd(next_frontier_size, flush_size);
    }
    __syncthreads();

    if(threadIdx.x == 0) {
      // Copy the shared buffer to next_frontier
      for (uint32_t i = threadIdx.y; i < flush_size; i+= blockDim.y) {
        next_frontier[start_index + i] = shared_buffer[i];
      }
    }

    // Reset the shared buffer for the next batch
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      shared_buffer_size = 0;
    }
  }


void gpu_bfs_coalesced_shared(const uint32_t N, const uint32_t M,
                              const uint32_t *h_rowptr,
                              const uint32_t *h_colidx, const uint32_t source,
                              int *h_distances) {
  float tot_time = 0.0;
  CUDA_TIMER_INIT(H2D_copy)

  // Allocate and copy graph to device
  uint32_t *d_row_offsets;
  uint32_t *d_col_indices;
  CHECK_CUDA(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_col_indices, M * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemcpy(d_row_offsets, h_rowptr, (N + 1) * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_col_indices, h_colidx, M * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  // Allocate memory for distances and frontier queues
  int *d_distances;
  uint32_t *d_frontier;
  uint32_t *d_next_frontier;
  uint32_t *d_next_frontier_size;
  CHECK_CUDA(cudaMalloc(&d_distances, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));

  std::vector<uint32_t> h_frontier(N);
  h_frontier[0] = source;

  CHECK_CUDA(cudaMemcpy(d_frontier, h_frontier.data(), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  // Initialize all distances to -1 (unvisited), and source distance to 0
  CHECK_CUDA(cudaMemset(d_distances, -1, N * sizeof(int)));
  CHECK_CUDA(cudaMemset(d_distances + source, 0, sizeof(int))); // set to 0

  CUDA_TIMER_STOP(H2D_copy)
#ifdef DEBUG_PRINTS
  CUDA_TIMER_PRINT(H2D_copy)
#endif
  tot_time += CUDA_TIMER_ELAPSED(H2D_copy);
  printf("after init %f\n", tot_time);
  CUDA_TIMER_DESTROY(H2D_copy)

  uint32_t current_frontier_size = 1;
  uint32_t level = 0;

  // Main BFS loop
  CPU_TIMER_INIT(BASELINE_BFS)
  while (current_frontier_size > 0) {

#ifdef DEBUG_PRINTS
    printf("[GPU BFS%s] level=%u, current_frontier_size=%u\n",
           is_placeholder ? "" : " BASELINE", level, current_frontier_size);
#endif
#ifdef ENABLE_NVTX
    // Mark start of level in NVTX
    nvtxRangePushA(("BFS Level " + std::to_string(level)).c_str());
#endif

    // Reset counter for next frontier
    CHECK_CUDA(cudaMemset(d_next_frontier_size, 0, sizeof(uint32_t)));

    // CUDA_TIMER_INIT(BFS_kernel)
    dim3 block_dim(24, 32);
    dim3 grid_dim(CEILING(current_frontier_size, block_dim.x));
    bfs_kernel_coalesced_shared<<<grid_dim, block_dim>>>(
        d_row_offsets, d_col_indices, d_distances, d_frontier, d_next_frontier,
        current_frontier_size, level, d_next_frontier_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    // CUDA_TIMER_STOP(BFS_kernel)
    // #ifdef DEBUG_PRINTS
    //   CUDA_TIMER_PRINT(BFS_kernel)
    // #endif
    // CUDA_TIMER_DESTROY(BFS_kernel)

    // Swap frontier pointers
    std::swap(d_frontier, d_next_frontier);

    // Copy size of next frontier to host
    CHECK_CUDA(cudaMemcpy(&current_frontier_size, d_next_frontier_size,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    level++;

#ifdef ENABLE_NVTX
    // End NVTX range for level
    nvtxRangePop();
#endif
  }
  CPU_TIMER_STOP(BASELINE_BFS)
#ifdef DEBUG_PRINTS
  CPU_TIMER_PRINT(BASELINE_BFS)
#endif
  tot_time += CPU_TIMER_ELAPSED(BASELINE_BFS);
  printf("after coalesced %f\n", tot_time);
  CUDA_TIMER_INIT(D2H_copy)
  CHECK_CUDA(cudaMemcpy(h_distances, d_distances, N * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_TIMER_STOP(D2H_copy)
#ifdef DEBUG_PRINTS
  CUDA_TIMER_PRINT(D2H_copy)
#endif
  tot_time += CUDA_TIMER_ELAPSED(D2H_copy);
  CUDA_TIMER_DESTROY(D2H_copy)

  printf("\n[OUT] Total BFS time: %f ms\n" RESET, tot_time);

  // Free device memory
  cudaFree(d_row_offsets);
  cudaFree(d_col_indices);
  cudaFree(d_distances);
  cudaFree(d_frontier);
  cudaFree(d_next_frontier);
  cudaFree(d_next_frontier_size);
}