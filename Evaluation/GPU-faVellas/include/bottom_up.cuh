#include <vector>
#include <queue>
#include <string>
#include "utils.cuh"

// Kernel: Process each node. If it's unvisited and close to the frontier, visit it
__global__ void bfs_kernel_bottom_up(
    const uint32_t *row_offsets,  // CSR row offsets
    const uint32_t *col_indices,  // CSR column indices (neighbors)
    int *distances,               // Output distances array
    const uint32_t *d_level,      // BFS level (depth)
    uint32_t *next_frontier_size, // Counter for next frontier
    int *visited,
    int N
) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;

  uint32_t node = tid;
  uint32_t row_start = row_offsets[node];
  uint32_t row_end = row_offsets[node + 1];

  uint32_t current_level = *d_level;

  // if the node has already been visited
  if (visited[node] & 0b100)
    return;

  for (uint32_t i = row_start; i < row_end; i++)
  {
    uint32_t neighbor = col_indices[i];

    // Check if any of the adjacent nodes are frontier
    if ((visited[neighbor] & 0b1) == 1)
    {
      // add node to next frontier
      visited[node] |= 0b10;
      // set the node as visited
      visited[node] |= 0b100;
      // set the distance of the node
      distances[node] = current_level + 1;
      // track the amount of added nodes to know when to stop
      atomicAdd(next_frontier_size, 1);
      return;
    }
  }
}

__global__ void swap_kernel(int *visited, int N)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;

  int is_next_frontier = visited[tid] & 0b10; // keeping the is_next_frontier info
  visited[tid] &= 0b11111100;                 // deleting the last two bits
  visited[tid] |= is_next_frontier / 2;       // moving the is_next_frontier info into the is_frontier bit
}

// function called to perform the BFS
void gpu_bfs_bottom_up(
    const uint32_t N,
    const uint32_t M,
    const uint32_t *h_rowptr,
    const uint32_t *h_colidx,
    const uint32_t source,
    int *h_distances,
    bool is_placeholder)
{

  /*
    ALLOCATIONS AND INITIALIZATIONS
  */

  float tot_time = 0.0;
  CUDA_TIMER_INIT(H2D_copy)

  // Allocate and copy graph to device
  uint32_t *d_row_offsets;
  uint32_t *d_col_indices;
  CHECK_CUDA(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_col_indices, M * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemcpy(d_row_offsets, h_rowptr, (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_col_indices, h_colidx, M * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // Allocate memory for distances and frontier queues
  int *d_distances;
  uint32_t *d_next_frontier_size;
  CHECK_CUDA(cudaMalloc(&d_distances, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));

  // Allocate depth on the GPU to avoid a ping-pong between CPU and GPU at every update
  uint32_t *d_level;
  CHECK_CUDA(cudaMalloc(&d_level, sizeof(uint32_t)));
  CHECK_CUDA(cudaMemset(d_level, 0, sizeof(uint32_t))); // level = 0

  // Allocate memory for unvisited nodes

  // from the least-significant, visited[i] bits:
  // - bit 0 = is_frontier
  // - bit 1 = is_next_frontier
  // - bit 2 = visited

  int *visited;
  CHECK_CUDA(cudaMalloc(&visited, N * sizeof(int)));

  /*
    ACTUAL IMPLEMENTATION
  */

  int *h_visited = new int[N]();
  h_visited[source] = 0b101;
  CHECK_CUDA(cudaMemcpy(visited, h_visited, N * sizeof(int), cudaMemcpyHostToDevice));

  // Initialize all distances to -1 (unvisited), and source distance to 0
  CHECK_CUDA(cudaMemset(d_distances, -1, N * sizeof(int)));
  CHECK_CUDA(cudaMemset(d_distances + source, 0, sizeof(int))); // set to 0

  CUDA_TIMER_STOP(H2D_copy)
#ifdef DEBUG_PRINTS
  CUDA_TIMER_PRINT(H2D_copy)
#endif
  tot_time += CUDA_TIMER_ELAPSED(H2D_copy);
  CUDA_TIMER_DESTROY(H2D_copy)

  uint32_t next_frontier_size = 1;
  uint32_t level = 0;

  // Main BFS loop
  CPU_TIMER_INIT(BFS)
  while (next_frontier_size > 0)
  {

#ifdef DEBUG_PRINTS
    printf("[GPU BFS%s] level=%u, unvisited nodes=%u\n", is_placeholder ? "" : " BOTTOM-UP", level, next_frontier_size);
#endif
#ifdef ENABLE_NVTX
    // Mark start of level in NVTX
    nvtxRangePushA(("BFS Level " + std::to_string(level)).c_str());
#endif

    // Reset counter for next frontier
    CHECK_CUDA(cudaMemset(d_next_frontier_size, 0, sizeof(uint32_t)));

    uint32_t block_size = 512;
    uint32_t num_blocks = CEILING(N, block_size);

    // CUDA_TIMER_INIT(BFS_kernel)
    bfs_kernel_bottom_up<<<num_blocks, block_size>>>(
        d_row_offsets,
        d_col_indices,
        d_distances,
        d_level,
        d_next_frontier_size,
        visited,
        N);
    CHECK_CUDA(cudaDeviceSynchronize());
    // CUDA_TIMER_STOP(BFS_kernel)
    // #ifdef DEBUG_PRINTS
    //   CUDA_TIMER_PRINT(BFS_kernel)
    // #endif
    // CUDA_TIMER_DESTROY(BFS_kernel)

    // Swap frontier pointers
    // std::swap(d_frontier, d_next_frontier);

    // Adding newly-discovered nodes to the new frontier
    swap_kernel<<<num_blocks, block_size>>>(visited, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy size of next frontier to host
    CHECK_CUDA(cudaMemcpy(&next_frontier_size, d_next_frontier_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    level++;
    CHECK_CUDA(cudaMemcpy(d_level, &level, sizeof(uint32_t), cudaMemcpyHostToDevice));

#ifdef ENABLE_NVTX
    // End NVTX range for level
    nvtxRangePop();
#endif
  }

  /*
    DEALLOCATIONS AND TIMER
  */

  // Handling the timing

  CPU_TIMER_STOP(BFS)
#ifdef DEBUG_PRINTS
  CPU_TIMER_PRINT(BFS)
#endif
  tot_time += CPU_TIMER_ELAPSED(BFS);

  CUDA_TIMER_INIT(D2H_copy)
  CHECK_CUDA(cudaMemcpy(h_distances, d_distances, N * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_TIMER_STOP(D2H_copy)
#ifdef DEBUG_PRINTS
  CUDA_TIMER_PRINT(D2H_copy)
#endif
  tot_time += CUDA_TIMER_ELAPSED(D2H_copy);
  CUDA_TIMER_DESTROY(D2H_copy)

  // Printing the output information
  printf("\n[OUT] Total BFS time: %f ms\n", tot_time);
  uint32_t final_level = 0;
  CHECK_CUDA(cudaMemcpy(&final_level, d_level, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  // Freeing the dynamically allocated array
  delete[] h_visited;

  // Free device memory
  cudaFree(d_row_offsets);
  cudaFree(d_col_indices);
  cudaFree(d_distances);
  cudaFree(visited);
  cudaFree(d_next_frontier_size);
  cudaFree(d_level);
}