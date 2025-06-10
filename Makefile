CUDAC=nvcc
CUDA_FLAGS=-O3 -Iinclude -arch=sm_80
C_FLAGS=-Wall -Wextra # -std=c99
PROF_FLAGS=-lineinfo

SRC_DIR=src
BIN_DIR=bin

DIST_MMIO=distributed_mmio
DIST_MMIO_INCLUDE=$(DIST_MMIO)/include
DIST_MMIO_SRCS=$(DIST_MMIO)/src/mmio.cpp $(DIST_MMIO)/src/mmio_utils.cpp

SRCS=$(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cuh)

all: $(BIN_DIR)/bfs $(BIN_DIR)/bfs_profiling

$(BIN_DIR)/bfs: $(SRCS)
	@ mkdir -p $(BIN_DIR)
	$(CUDAC) $(CUDA_FLAGS) -I$(DIST_MMIO_INCLUDE) --compiler-options "$(C_FLAGS)" -o $@ $^ $(DIST_MMIO_SRCS)

$(BIN_DIR)/bfs_dbg: $(SRCS)
	@ mkdir -p $(BIN_DIR)
	$(CUDAC) $(CUDA_FLAGS) -I$(DIST_MMIO_INCLUDE) --compiler-options "$(C_FLAGS)" -DDEBUG_PRINTS -o $@ $^ $(DIST_MMIO_SRCS)

$(BIN_DIR)/bfs_profiling: $(SRCS)
	@ mkdir -p $(BIN_DIR)
	$(CUDAC) $(CUDA_FLAGS) -I$(DIST_MMIO_INCLUDE) --compiler-options "$(C_FLAGS)" $(PROF_FLAGS) -DENABLE_NVTX -o $@ $^ $(DIST_MMIO_SRCS)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
