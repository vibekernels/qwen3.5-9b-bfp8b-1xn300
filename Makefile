NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# RTX 5090 = SM 12.0
CUDA_ARCH = -gencode arch=compute_120,code=sm_120

NVCC_FLAGS = $(CUDA_ARCH) -O3 -std=c++17 --use_fast_math -Xcompiler -Wall
LDFLAGS = -lcudart

BUILD_DIR = build
SRC_DIR = src

# Source files
CU_SOURCES = \
    $(SRC_DIR)/main.cu \
    $(SRC_DIR)/sampling.cu \
    $(SRC_DIR)/kernels/rmsnorm.cu \
    $(SRC_DIR)/kernels/embedding.cu \
    $(SRC_DIR)/kernels/rope.cu \
    $(SRC_DIR)/kernels/attention.cu \
    $(SRC_DIR)/kernels/ffn.cu \
    $(SRC_DIR)/kernels/mamba.cu

CPP_SOURCES = \
    $(SRC_DIR)/gguf_loader.cpp \
    $(SRC_DIR)/tokenizer.cpp

# Object files
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
ALL_OBJECTS = $(CU_OBJECTS) $(CPP_OBJECTS)

TARGET = qwen-inference

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(ALL_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

# CUDA source compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# C++ source compilation (compiled through nvcc for cuda_bf16.h)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -x cu -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
