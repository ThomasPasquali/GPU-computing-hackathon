#include "../include/utils.cuh"
#include <sys/types.h>
// how big should the shared buffer be? it depends on the available storage
#define SHARED_BUFFER_SIZE 10000 // Adjusted for shared memory size
__global__ void bfs_kernel_coalesced_shared_faster(
    const uint32_t *__restrict__ row_offsets,
    const uint32_t *__restrict__ col_indices, int *__restrict__ distances,
    const uint32_t *__restrict__ frontier, uint32_t *__restrict__ next_frontier,
    uint32_t *frontier_size, uint32_t *current_level,
    uint32_t *__restrict__ next_frontier_size) {
  __shared__ uint32_t shared_buffer[SHARED_BUFFER_SIZE];
  __shared__ uint32_t shared_buffer_size;
  __shared__ uint32_t start_index;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    shared_buffer_size =
        0; // Reset shared buffer size at the start of each block
  }
  while (tid < *frontier_size) {
    __syncthreads();
    uint32_t node = frontier[tid];
    uint32_t row_start = row_offsets[node];
    uint32_t row_end = row_offsets[node + 1];

    for (uint32_t i = row_start + threadIdx.y; i < row_end; i += blockDim.y) {
      uint32_t neighbor = col_indices[i];
      if (atomicCAS(&distances[neighbor], -1, *current_level + 1) == -1) {
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

    tid += blockDim.x * gridDim.x;
  }
}

// dummy thread to reset the pointer, and increment the level, it is done in another launch to be sure the memory get's correctly synchronized
__global__ void reset_pointer(uint32_t *d_frontier_size,
                              uint32_t *current_level) {
  *current_level = *current_level + 1;
  *d_frontier_size = 0; // Reset the frontier size
}

void gpu_bfs_coalesced_shared_faster(const uint32_t N, const uint32_t M,
                                     const uint32_t *h_rowptr,
                                     const uint32_t *h_colidx,
                                     const uint32_t source, int *h_distances) {
  cudaMemPool_t pool;
  cudaDeviceGetDefaultMemPool(&pool, 0);
  cudaDeviceSetMemPool(0, pool);
  // Warm up the async allocator (I don't think it should be included in the
  // timing)
  void *dummy_ptr = nullptr;
  cudaStream_t dummy_stream;
  cudaStreamCreate(&dummy_stream);
  cudaMallocAsync(&dummy_ptr, 1, dummy_stream); // 1 byte is enough
  cudaFreeAsync(dummy_ptr, dummy_stream);
  cudaStreamSynchronize(dummy_stream);
  cudaStreamDestroy(dummy_stream);

  float tot_time = 0.0;
  CUDA_TIMER_INIT(H2D_copy)
  cudaStream_t stream_row, stream_col, stream_frontier, stream_next_frontier,
      stream_distances;
  CHECK_CUDA(cudaStreamCreate(&stream_row));
  CHECK_CUDA(cudaStreamCreate(&stream_col));
  CHECK_CUDA(cudaStreamCreate(&stream_frontier));
  CHECK_CUDA(cudaStreamCreate(&stream_next_frontier));
  CHECK_CUDA(cudaStreamCreate(&stream_distances));

  // Allocate and copy graph to device
  uint32_t *d_row_offsets;
  uint32_t *d_col_indices;

  CHECK_CUDA(
      cudaMallocAsync(&d_row_offsets, (N + 1) * sizeof(uint32_t), stream_row));
  CHECK_CUDA(cudaMemcpyAsync(d_row_offsets, h_rowptr,
                             (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice,
                             stream_row));
  CHECK_CUDA(cudaMallocAsync(&d_col_indices, M * sizeof(uint32_t), stream_col));

  CHECK_CUDA(cudaMemcpyAsync(d_col_indices, h_colidx, M * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_col));

  // Allocate memory for distances and frontier queues
  int *d_distances;
  uint32_t *d_frontier;
  uint32_t *d_frontier_size;
  uint32_t *d_next_frontier;
  uint32_t *d_next_frontier_size, *level;

  CHECK_CUDA(cudaMallocAsync(&d_distances, N * sizeof(int), stream_distances));
  CHECK_CUDA(
      cudaMallocAsync(&d_frontier, N * sizeof(uint32_t), stream_frontier));
  CHECK_CUDA(
      cudaMallocAsync(&d_frontier_size, sizeof(uint32_t), stream_frontier));
  CHECK_CUDA(cudaMallocAsync(&d_next_frontier, N * sizeof(uint32_t),
                             stream_next_frontier));
  CHECK_CUDA(cudaMallocAsync(&d_next_frontier_size, sizeof(uint32_t),
                             stream_next_frontier));
  CHECK_CUDA(cudaMallocAsync(&level, sizeof(uint32_t), stream_next_frontier));
  CHECK_CUDA(cudaMemsetAsync(level, 0, sizeof(uint32_t), stream_next_frontier));

  CHECK_CUDA(cudaMemset(d_distances + source, 0, sizeof(int))); // set to 0*/
  CHECK_CUDA(cudaMemcpyAsync(d_frontier, &source, sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_frontier));
  uint32_t dummy = 1;
  CHECK_CUDA(cudaMemcpyAsync(d_frontier_size, &dummy, sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_frontier));
  CHECK_CUDA(cudaMemsetAsync(d_next_frontier_size, 0, sizeof(uint32_t),
                             stream_next_frontier));
  CHECK_CUDA(
      cudaMemsetAsync(d_distances, -1, N * sizeof(int), stream_distances));
  CHECK_CUDA(
      cudaMemsetAsync(d_distances + source, 0, sizeof(int), stream_distances));

  CHECK_CUDA(cudaStreamSynchronize(stream_frontier));
  CHECK_CUDA(cudaStreamSynchronize(stream_distances));
  CHECK_CUDA(cudaStreamSynchronize(stream_next_frontier));

  CHECK_CUDA(cudaStreamSynchronize(stream_row));
  CHECK_CUDA(cudaStreamSynchronize(stream_col));

  cudaStream_t stream_kernel, stream_memcpy;
  cudaEvent_t kernel_done;
  cudaStreamCreate(&stream_kernel);
  cudaStreamCreate(&stream_memcpy);
  cudaEventCreate(&kernel_done);

  CUDA_TIMER_STOP(H2D_copy)
#ifdef DEBUG_PRINTS
  CUDA_TIMER_PRINT(H2D_copy)
#endif
  tot_time += CUDA_TIMER_ELAPSED(H2D_copy);
  // printf for debugging purposes, out of timing
  printf("after init %f\n", tot_time);
  CUDA_TIMER_DESTROY(H2D_copy)

  uint32_t host_frontier_size = 1;
  // Main BFS loop
  CPU_TIMER_INIT(BASELINE_BFS)
  dim3 block_dim(24, 32);
  dim3 grid_dim(568);
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  bool graphCreated = false;
  // main BFS loop
  while (host_frontier_size > 0) {

#ifdef DEBUG_PRINTS
    printf("[GPU BFS%s] level=%u, current_frontier_size=%u\n",
           is_placeholder ? "" : " BASELINE", level, current_frontier_size);
#endif
#ifdef ENABLE_NVTX
    // Mark start of level in NVTX
    nvtxRangePushA(("BFS Level " + std::to_string(level)).c_str());
#endif
    {
      if (!graphCreated) {
        cudaStreamBeginCapture(stream_kernel, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < 12; ++i) {
          bfs_kernel_coalesced_shared_faster<<<grid_dim, block_dim, 0,
                                               stream_kernel>>>(
              d_row_offsets, d_col_indices, d_distances, d_frontier,
              d_next_frontier, d_frontier_size, level, d_next_frontier_size);
          reset_pointer<<<1, 1, 0, stream_kernel>>>(d_frontier_size, level);
          bfs_kernel_coalesced_shared_faster<<<grid_dim, block_dim, 0,
                                               stream_kernel>>>(
              d_row_offsets, d_col_indices, d_distances, d_next_frontier,
              d_frontier, d_next_frontier_size, level, d_frontier_size);
          reset_pointer<<<1, 1, 0, stream_kernel>>>(d_next_frontier_size,
                                                    level);
        }
        cudaMemcpyAsync(&host_frontier_size, d_frontier_size, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream_kernel);

        cudaStreamEndCapture(stream_kernel, &graph);
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
        graphCreated = true;
      }

      cudaGraphLaunch(graphExec, stream_kernel);
      cudaStreamSynchronize(stream_kernel);
    }
  }
  cudaDeviceSynchronize();
  CPU_TIMER_STOP(BASELINE_BFS)
#ifdef DEBUG_PRINTS
  CPU_TIMER_PRINT(BASELINE_BFS)
#endif
  tot_time += CPU_TIMER_ELAPSED(BASELINE_BFS);
  printf("after execution %f\n", tot_time);
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
  //should memory be included in the timing don't think so
  cudaFree(d_row_offsets);
  cudaFree(d_col_indices);
  cudaFree(d_distances);
  cudaFree(d_frontier);
  cudaFree(d_next_frontier);
  cudaFree(d_next_frontier_size);
  cudaFree(d_frontier_size);
  cudaFree(level);
}