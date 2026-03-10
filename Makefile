# Qwen 3.5-9B inference on Tenstorrent N300
# Usage:
#   make                  — build everything
#   make quicktest        — smoke test: "The capital of France is"
#   make test             — run full integration tests
#   make clean            — remove build artifacts
#
# Environment:
#   TT_METAL_HOME         — tt-metal source tree    (default: /home/ubuntu/tt-metal)
#   TT_METAL_BUILD        — tt-metal build dir      (default: $(TT_METAL_HOME)/build_Release)
#   MODEL_PATH            — path to .gguf model     (default: auto-resolve from HuggingFace)

TT_METAL_HOME  ?= /home/ubuntu/tt-metal
TT_METAL_BUILD ?= $(TT_METAL_HOME)/build_Release
CXX            := clang++-20
BUILD          := build

# tt-metal SDK paths (extracted from CMake find_package(TT-Metalium))
TT_INCLUDES := \
	-isystem $(TT_METAL_BUILD)/include \
	-isystem $(TT_METAL_BUILD)/libexec/tt-metalium/tt_metal/hostdevcommon/api \
	-isystem $(TT_METAL_BUILD)/libexec/tt-metalium \
	-isystem $(TT_METAL_BUILD)/include/metalium-thirdparty

TT_LIBS := \
	$(TT_METAL_BUILD)/lib/libtt_metal.so \
	$(TT_METAL_BUILD)/lib/libtracy.so.0.10.0 \
	-ldl \
	$(TT_METAL_BUILD)/lib/libdevice.so \
	$(TT_METAL_BUILD)/lib/libtt_stl.so \
	$(TT_METAL_BUILD)/lib/libspdlog.a

TT_DEFINES := \
	-DENCHANTUM_ENABLE_MSVC_SPEEDUP=1 \
	-DFMT_HEADER_ONLY=1 \
	-DNTEST \
	-DSPDLOG_COMPILED_LIB \
	-DSPDLOG_FMT_EXTERNAL \
	-DTRACY_ENABLE \
	-DTRACY_IMPORTS \
	-DTRACY_TIMER_FALLBACK

CXXFLAGS := -std=gnu++20 -Wno-int-to-pointer-cast -fno-omit-frame-pointer \
	$(TT_DEFINES) $(TT_INCLUDES) -I src -I src/third_party \
	-DKERNEL_DIR=\"$(CURDIR)/src/kernels\"

ENGINE_CXXFLAGS := $(CXXFLAGS) -march=native -ffast-math
LDFLAGS := -rdynamic -Wl,-rpath,$(TT_METAL_BUILD)/lib

# Source files
ENGINE_SRCS := src/engine.cpp src/gguf_loader.cpp src/tokenizer.cpp
ENGINE_OBJS := $(ENGINE_SRCS:%.cpp=$(BUILD)/%.o)

# Targets
.PHONY: all clean test quicktest

all: $(BUILD)/test_forward $(BUILD)/test_inference

# Engine static library
$(BUILD)/libqwen_engine.a: $(ENGINE_OBJS)
	@mkdir -p $(@D)
	ar rcs $@ $^

$(BUILD)/src/engine.o: src/engine.cpp
	@mkdir -p $(@D)
	$(CXX) $(ENGINE_CXXFLAGS) -c $< -o $@

$(BUILD)/src/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Main test targets (link against engine)
$(BUILD)/test_forward: src/tests/test_forward.cpp $(BUILD)/libqwen_engine.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(BUILD)/libqwen_engine.a $(TT_LIBS)

$(BUILD)/test_inference: src/tests/test_inference.cpp src/download.cpp $(BUILD)/libqwen_engine.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) src/tests/test_inference.cpp src/download.cpp -o $@ $(LDFLAGS) $(BUILD)/libqwen_engine.a $(TT_LIBS)

# Standalone test targets (link directly against tt-metal)
$(BUILD)/test_matmul: src/tests/test_matmul.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(TT_LIBS)

$(BUILD)/test_dram_bw: src/tests/test_dram_bw.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(TT_LIBS)

$(BUILD)/test_mesh_overhead: src/tests/test_mesh_overhead.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(TT_LIBS)

# Quick smoke test: "The capital of France is" → should output "Paris"
quicktest: $(BUILD)/test_forward
	@env TT_METAL_RUNTIME_ROOT=$(TT_METAL_HOME) QUIET=1 \
		$(BUILD)/test_forward \
		$${MODEL_PATH:-/home/ubuntu/qwen3.5-9b-bf16-1x5090/models/Qwen3.5-9B-BF16.gguf} \
		"The capital of France is" 16 --raw 2>/dev/null

# Run integration test suite
test: $(BUILD)/test_inference
	@env TT_METAL_RUNTIME_ROOT=$(TT_METAL_HOME) \
		MODEL_PATH=$${MODEL_PATH:-/home/ubuntu/qwen3.5-9b-bf16-1x5090/models/Qwen3.5-9B-BF16.gguf} \
		QUIET=1 \
		$(BUILD)/test_inference 2>/dev/null

clean:
	rm -rf $(BUILD)
