#include <cstdint>
#include <sys/types.h>
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

#include "coalesced.cu"
#include"coalesced_shared.cu"
#include"coalesced_independent.cu"
#include"coalesced_shared_copy.cu"
#include"coalesced_shared_faster.cu"
//#include"coalesced_shared_copy_rec.cu"
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
  Matrix_Metadata
      metadata; //=(Matrix_Metadata*)malloc(sizeof(Matrix_Metadata));
  CPU_TIMER_INIT(MTX_read);
  CSR_local<uint32_t, float> *csr = Distr_MMIO_CSR_local_read<uint32_t, float>(
      args.filename, false, &metadata);
  if (csr == NULL) {
    printf("Failed to import graph from file [%s]\n", args.filename);
    return -1;
  }
  CPU_TIMER_STOP(MTX_read)
  printf("\n[OUT] MTX file read time: %f ms\n", CPU_TIMER_ELAPSED(MTX_read));
  printf("Graph size: %.3fM vertices, %.3fM edges\n", csr->nrows / 1e6,
         csr->nnz / 1e6);

  GraphCSR graph;
  graph.row_ptr = csr->row_ptr;
  graph.col_idx = csr->col_idx;
  graph.num_vertices = csr->nrows;
  graph.num_edges = csr->nnz;
  // print_graph_csr(graph);

  uint32_t *sources =
      generate_sources(&graph, args.runs, graph.num_vertices, args.source);
  int *distances_gpu_baseline = (int *)malloc(graph.num_vertices * sizeof(int));
  int *distances = (int *)malloc(graph.num_vertices * sizeof(int));
  bool correct = true;
  
  uint32_t *d_row_ptr_pinned, *d_col_idx_pinned;
  int *distances_pinned;
  CHECK_CUDA(cudaMallocHost((void **)&d_row_ptr_pinned,
                            (graph.num_vertices + 1) * sizeof(uint32_t)));
  CHECK_CUDA(cudaMallocHost((void **)&d_col_idx_pinned,
                            graph.num_edges * sizeof(uint32_t)));
  CHECK_CUDA(cudaMallocHost((void **)&distances_pinned,
                            graph.num_vertices * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_row_ptr_pinned, graph.row_ptr,
                        (graph.num_vertices + 1) * sizeof(uint32_t),
                        cudaMemcpyHostToHost));
  CHECK_CUDA(cudaMemcpy(d_col_idx_pinned, graph.col_idx,
                        graph.num_edges * sizeof(uint32_t),
                        cudaMemcpyHostToHost));
  

  for (int source_i = 0; source_i < args.runs; source_i++) {
    uint32_t source = sources[source_i];
    printf("\n[OUT] -- BFS iteration #%u, source=%u --\n", source_i, source);

    // Run the BFS baseline
    gpu_bfs_baseline(graph.num_vertices, graph.num_edges, graph.row_ptr,
                     graph.col_idx, source, distances_gpu_baseline, false);

#ifdef ENABLE_NVTX
    nvtxRangePushA("Complete BFS");
#endif
    gpu_bfs_coalesced_shared_faster(graph.num_vertices, graph.num_edges, d_row_ptr_pinned,
                     d_col_idx_pinned, source, distances_pinned);
    CHECK_CUDA(cudaMemcpy(distances, distances_pinned,
                          graph.num_vertices * sizeof(int),
                          cudaMemcpyHostToHost));
        
#ifdef ENABLE_NVTX
    nvtxRangePop();
#endif

    bool match = true;
#ifdef ENABLE_CORRECTNESS_CHECK
    for (uint32_t i = 0; i < graph.num_vertices; ++i) {
      if (distances_gpu_baseline[i] != distances[i]) {
        printf(
            "Mismatch at node %u: Baseline distance = %d, Your distance = %d\n",
            i, distances_gpu_baseline[i], distances[i]);
        match = false;
        break;
      }
    }
    if (match) {
      printf(BRIGHT_GREEN "Correctness OK\n" RESET);
    } else {
      printf(BRIGHT_RED
             "GPU and CPU BFS results do not match for source node %u.\n" RESET,
             source);
      return_code = EXIT_INCORRECT_DISTANCES;
      correct = false;
    }
#endif

#ifdef ENABLE_CPU_BASELINE
    int cpu_distances[graph.num_vertices];

    CPU_TIMER_INIT(CPU_BFS)
    cpu_bfs_baseline(graph.num_vertices, graph.row_ptr, graph.col_idx, source,
                     cpu_distances);
    CPU_TIMER_CLOSE(CPU_BFS)

    match = true;
    for (uint32_t i = 0; i < graph.num_vertices; ++i) {
      if (distances_gpu_baseline[i] != cpu_distances[i]) {
        printf("Mismatch at node %u: GPU distance = %d, CPU distance = %d\n", i,
               distances_gpu_baseline[i], cpu_distances[i]);
        match = false;
        break;
      }
    }
    if (match) {
      printf(BRIGHT_GREEN "[CPU] Correctness OK\n" RESET);
    } else {
      printf(BRIGHT_RED
             "GPU and CPU BFS results do not match for source node %u.\n" RESET,
             source);
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
