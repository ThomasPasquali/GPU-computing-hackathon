// #define ENABLE_NVTX
// #define ENABLE_CPU_BASELINE
// #define DEBUG_PRINTS
#define ENABLE_CORRECTNESS_CHECK

#define EXIT_INCORRECT_DISTANCES 10

#include <stdio.h>
#include <cuda_runtime.h>

#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include "../distributed_mmio/include/mmio.h"
#include "../distributed_mmio/include/mmio_utils.h"

#include "../include/colors.h"
#include "../include/utils.cuh"
#include "../include/cli.hpp"
#include "../include/mt19937-64.hpp"
#include "../include/bfs_baseline.cuh"



#define WARP_SIZE 32
#define SMALL_FRONTIER_THRESHOLD 1024
#define LARGE_FRONTIER_THRESHOLD 32768

// Kernel for small frontiers - one thread per frontier node
__global__ void bfs_small_kernel(
  const uint32_t* row_offsets,
  const uint32_t* col_indices,
  int* distances,
  const uint32_t* frontier,
  uint32_t* next_frontier,
  uint32_t frontier_size,
  uint32_t current_level,
  uint32_t* next_frontier_size
) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid < frontier_size) {
    uint32_t node = frontier[tid];
    uint32_t start = row_offsets[node];
    uint32_t end = row_offsets[node + 1];
    
    for (uint32_t i = start; i < end; i++) {
      uint32_t neighbor = col_indices[i];
      if (atomicCAS(&distances[neighbor], -1, current_level + 1) == -1) {
        uint32_t pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = neighbor;
      }
    }
  }
}

// Kernel for medium frontiers - warp-centric approach
__global__ void bfs_medium_kernel(
  const uint32_t* row_offsets,
  const uint32_t* col_indices,
  int* distances,
  const uint32_t* frontier,
  uint32_t* next_frontier,
  uint32_t frontier_size,
  uint32_t current_level,
  uint32_t* next_frontier_size
) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id = tid / WARP_SIZE;
  uint32_t lane_id = tid % WARP_SIZE;
  
  if (warp_id < frontier_size) {
    uint32_t node = frontier[warp_id];
    uint32_t start = row_offsets[node];
    uint32_t end = row_offsets[node + 1];
    
    // Warp processes neighbors collaboratively
    for (uint32_t i = start + lane_id; i < end; i += WARP_SIZE) {
      uint32_t neighbor = col_indices[i];
      if (atomicCAS(&distances[neighbor], -1, current_level + 1) == -1) {
        uint32_t pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = neighbor;
      }
    }
  }
}

// Kernel for large frontiers - work-stealing approach
__global__ void bfs_large_kernel(
  const uint32_t* row_offsets,
  const uint32_t* col_indices,
  int* distances,
  const uint32_t* frontier,
  uint32_t* next_frontier,
  uint32_t frontier_size,
  uint32_t current_level,
  uint32_t* next_frontier_size,
  uint32_t* work_counter
) {
  __shared__ uint32_t shared_frontier[256];
  __shared__ uint32_t shared_size;
  
  if (threadIdx.x == 0) {
    shared_size = 0;
  }
  __syncthreads();
  
  // Work-stealing loop
  while (true) {
    uint32_t work_item = atomicAdd(work_counter, 1);
    if (work_item >= frontier_size) break;
    
    uint32_t node = frontier[work_item];
    uint32_t start = row_offsets[node];
    uint32_t end = row_offsets[node + 1];
    
    for (uint32_t i = start; i < end; i++) {
      uint32_t neighbor = col_indices[i];
      if (atomicCAS(&distances[neighbor], -1, current_level + 1) == -1) {
        // Use shared memory to reduce atomic operations
        uint32_t local_pos = atomicAdd(&shared_size, 1);
        if (local_pos < 256) {
          shared_frontier[local_pos] = neighbor;
        } else {
          // Fallback to global memory
          uint32_t pos = atomicAdd(next_frontier_size, 1);
          next_frontier[pos] = neighbor;
        }
      }
    }
  }
  
  __syncthreads();
  
  // Flush shared memory to global memory
  for (uint32_t i = threadIdx.x; i < shared_size && i < 256; i += blockDim.x) {
    uint32_t pos = atomicAdd(next_frontier_size, 1);
    next_frontier[pos] = shared_frontier[i];
  }
}

void gpu_bfs(
  const uint32_t N,           // Number of vertices
  const uint32_t M,           // Number of edges
  const uint32_t *h_rowptr,   // Graph CSR rowptr
  const uint32_t *h_colidx,   // Graph CSR colidx
  const uint32_t source,      // Source vertex
  int *h_distances            // Write here your distances
) {
  float tot_time = 0.0f;
  CPU_TIMER_INIT(BFS_preprocess)

  // Allocate and copy graph to device
  uint32_t* d_row_offsets; uint32_t* d_col_indices; int* d_distances;
  CHECK_CUDA(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_col_indices, M * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_distances, N * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_row_offsets, h_rowptr, (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_col_indices, h_colidx, M * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // Initialize distances
  CHECK_CUDA(cudaMemset(d_distances, -1, N * sizeof(int)));
  int zero = 0;
  CHECK_CUDA(cudaMemcpy(d_distances + source, &zero, sizeof(int), cudaMemcpyHostToDevice));

  uint32_t* d_frontier; 
  uint32_t* d_next_frontier; 
  uint32_t* d_next_frontier_size;
  uint32_t* d_work_counter;
  CHECK_CUDA(cudaMalloc(&d_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_work_counter, sizeof(uint32_t)));

  // Initialize frontier
  CHECK_CUDA(cudaMemcpy(d_frontier, &source, sizeof(uint32_t), cudaMemcpyHostToDevice));
  
  uint32_t current_frontier_size = 1;
  uint32_t level = 0;

  CPU_TIMER_STOP(BFS_preprocess)
  tot_time += CPU_TIMER_ELAPSED(BFS_preprocess);
  CPU_TIMER_PRINT(BFS_preprocess)

  CPU_TIMER_INIT(BFS)

  while (current_frontier_size > 0) {
    // Reset counters
    CHECK_CUDA(cudaMemset(d_next_frontier_size, 0, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_work_counter, 0, sizeof(uint32_t)));

    // Choose kernel based on frontier size
    if (current_frontier_size <= SMALL_FRONTIER_THRESHOLD) {
      // Small frontier: one thread per frontier node
      uint32_t block_size = min(256U, max(32U, current_frontier_size));
      uint32_t num_blocks = (current_frontier_size + block_size - 1) / block_size;
      
      bfs_small_kernel<<<num_blocks, block_size>>>(
        d_row_offsets, d_col_indices, d_distances,
        d_frontier, d_next_frontier, current_frontier_size,
        level, d_next_frontier_size
      );
    } 
    else if (current_frontier_size <= LARGE_FRONTIER_THRESHOLD) {
      // Medium frontier: warp-centric approach
      uint32_t block_size = 256;
      uint32_t num_blocks = (current_frontier_size + block_size/WARP_SIZE - 1) / (block_size/WARP_SIZE);
      
      bfs_medium_kernel<<<num_blocks, block_size>>>(
        d_row_offsets, d_col_indices, d_distances,
        d_frontier, d_next_frontier, current_frontier_size,
        level, d_next_frontier_size
      );
    } 
    else {
      // Large frontier: work-stealing with shared memory
      uint32_t block_size = 256;
      uint32_t num_blocks = min(2048U, (N + block_size - 1) / block_size);
      
      bfs_large_kernel<<<num_blocks, block_size>>>(
        d_row_offsets, d_col_indices, d_distances,
        d_frontier, d_next_frontier, current_frontier_size,
        level, d_next_frontier_size, d_work_counter
      );
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy size of next frontier to host
    CHECK_CUDA(cudaMemcpy(&current_frontier_size, d_next_frontier_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // Swap frontier pointers
    std::swap(d_frontier, d_next_frontier);
    
    level++;
  }

  CPU_TIMER_STOP(BFS)
  tot_time += CPU_TIMER_ELAPSED(BFS);
  CPU_TIMER_PRINT(BFS)

  CPU_TIMER_INIT(BFS_postprocess)
  CHECK_CUDA(cudaMemcpy(h_distances, d_distances, N * sizeof(int), cudaMemcpyDeviceToHost));
  CPU_TIMER_STOP(BFS_postprocess)
  tot_time += CPU_TIMER_ELAPSED(BFS_postprocess);
  CPU_TIMER_PRINT(BFS_postprocess)

  printf("\n[OUT] Total BFS time: %f ms\n", tot_time);

  // Free device memory
  cudaFree(d_row_offsets);
  cudaFree(d_col_indices);
  cudaFree(d_distances);
  cudaFree(d_frontier);
  cudaFree(d_next_frontier);
  cudaFree(d_next_frontier_size);
  cudaFree(d_work_counter);
}


int main(int argc, char **argv) {
  int return_code = EXIT_SUCCESS;
  Cli_Args args;
  init_cli();
  if (parse_args(argc, argv, &args) != 0) {
    return -1;
  }
  CPU_TIMER_INIT(MTX_read)
  CSR_local<uint32_t, float> *csr = Distr_MMIO_CSR_local_read<uint32_t, float>(args.filename);
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
  // print_graph_csr(graph);

  uint32_t *sources = generate_sources(&graph, args.runs, graph.num_vertices, args.source);
  int *distances_gpu_baseline = (int *)malloc(graph.num_vertices * sizeof(int));
  int *distances = (int *)malloc(graph.num_vertices * sizeof(int));
  bool correct = true;

  for (int source_i = 0; source_i < args.runs; source_i++) {
    uint32_t source = sources[source_i];
    printf("\n[OUT] -- BFS iteration #%u, source=%u --\n", source_i, source);

    // Run the BFS baseline
    gpu_bfs_baseline(graph.num_vertices, graph.num_edges, graph.row_ptr, graph.col_idx, source, distances_gpu_baseline, false);
    // gpu_bfs(graph.num_vertices, graph.num_edges, graph.row_ptr, graph.col_idx, source, distances_gpu_baseline, false);

    #ifdef ENABLE_NVTX
		  nvtxRangePushA("Complete BFS");
    #endif
    gpu_bfs(graph.num_vertices, graph.num_edges, graph.row_ptr, graph.col_idx, source, distances);
    #ifdef ENABLE_NVTX
		  nvtxRangePop();
    #endif

    bool match = true;
    #ifdef ENABLE_CORRECTNESS_CHECK
      for (uint32_t i = 0; i < graph.num_vertices; ++i) {
        if (distances_gpu_baseline[i] != distances[i]) {
          printf("Mismatch at node %u: Baseline distance = %d, Your distance = %d\n", i, distances_gpu_baseline[i], distances[i]);
          match = false;
          break;
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
          printf("Mismatch at node %u: GPU distance = %d, CPU distance = %d\n", i, distances_gpu_baseline[i], cpu_distances[i]);
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

  if (correct) printf("\n[OUT] ALL RESULTS ARE CORRECT\n");
  else         printf(BRIGHT_RED "\nSOME RESULTS ARE WRONG\n" RESET);

  Distr_MMIO_CSR_local_destroy(&csr);
  free(sources);
  free(distances_gpu_baseline);
  free(distances);

  return return_code;
}