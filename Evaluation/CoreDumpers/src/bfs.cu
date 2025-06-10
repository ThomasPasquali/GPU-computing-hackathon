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
#include "../include/bfs_baseline_shared.cuh"

void gpu_bfs(
  const uint32_t N,           // Number of veritices
  const uint32_t M,           // Number of edges
  const uint32_t *h_rowptr,   // Graph CSR rowptr
  const uint32_t *h_colidx,   // Graph CSR colidx
  const uint32_t source,      // Source veritex
  int *h_distances            // Write here your distances
) {
  /***********************
   * IMPLEMENT HERE YOUR CUDA BFS
   * Feel free to structure you code (i.e. create other files, macros etc.)
   * *********************/


  // !! This is an example of how to keep track of runtime. Make sure to include everything. !!
  float tot_time = 0.0f;
  CPU_TIMER_INIT(BFS_preprocess)

  // <<< preprocess >>>

  // CHECK_CUDA(cudaDeviceSynchronize());
  CPU_TIMER_STOP(BFS_preprocess)
  tot_time += CPU_TIMER_ELAPSED(BFS_preprocess);
  CPU_TIMER_PRINT(BFS_preprocess)

  CPU_TIMER_INIT(BFS)

  // <<< kernel >>>
  gpu_bfs_shared_baseline(N, M, h_rowptr, h_colidx, source, h_distances, false);

  CHECK_CUDA(cudaDeviceSynchronize());
  CPU_TIMER_STOP(BFS)
  tot_time += CPU_TIMER_ELAPSED(BFS);
  CPU_TIMER_PRINT(BFS)

  CPU_TIMER_INIT(BFS_postprocess)

  // <<< postprocess >>>

  // CHECK_CUDA(cudaDeviceSynchronize());
  CPU_TIMER_STOP(BFS_postprocess)
  tot_time += CPU_TIMER_ELAPSED(BFS_postprocess);
  CPU_TIMER_PRINT(BFS_postprocess)

  // This output format is MANDATORY, DO NOT CHANGE IT
  printf("\n[OUT] Total BFS time: %f ms\n" RESET, tot_time);
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
