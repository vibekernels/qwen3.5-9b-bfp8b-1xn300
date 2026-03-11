# Qwen 3.5-9B inference on Tenstorrent N300
# Usage:
#   make setup            — init submodule + build tt-metal (first time only)
#   make                  — build everything
#   make quicktest        — smoke test: "The capital of France is"
#   make test             — run full integration tests
#   make clean            — remove build artifacts
#
# Environment:
#   TT_METAL_HOME         — tt-metal source tree    (default: third_party/tt-metal)
#   TT_METAL_BUILD        — tt-metal build dir      (default: $(TT_METAL_HOME)/build_Release)
#   MODEL_PATH            — path to .gguf model     (default: auto-resolve from HuggingFace)

TT_METAL_HOME  ?= $(CURDIR)/third_party/tt-metal
TT_METAL_BUILD ?= $(TT_METAL_HOME)/build_Release
CXX            := clang++-20
BUILD          := build

# tt-metal SDK paths (extracted from CMake find_package(TT-Metalium))
TT_INCLUDES := \
	-isystem $(TT_METAL_BUILD)/include \
	-isystem $(TT_METAL_BUILD)/libexec/tt-metalium/tt_metal/hostdevcommon/api \
	-isystem $(TT_METAL_BUILD)/libexec/tt-metalium \
	-isystem $(TT_METAL_BUILD)/include/metalium-thirdparty \
	-isystem $(TT_METAL_HOME)/tt_metal/impl/data_format

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
	$(TT_DEFINES) $(TT_INCLUDES) -I src -I third_party \
	-DKERNEL_DIR=\"$(CURDIR)/src/kernels\"

MARCH ?= native
ENGINE_CXXFLAGS := $(CXXFLAGS) -march=$(MARCH) -ffast-math
LDFLAGS := -rdynamic -Wl,-rpath,$(TT_METAL_BUILD)/lib

# Source files
ENGINE_SRCS := src/engine.cpp src/gguf_loader.cpp src/tokenizer.cpp
ENGINE_OBJS := $(ENGINE_SRCS:%.cpp=$(BUILD)/%.o)

# Targets
.PHONY: all clean test quicktest setup chat serve

all: $(BUILD)/test_forward $(BUILD)/test_inference $(BUILD)/qwen-chat $(BUILD)/qwen-server $(BUILD)/make_bfp8_gguf

# Auto-setup: init submodule + build tt-metal if SDK not found
# Only the three required submodules (tracy, tt_llk, umd) are needed; skipping
# the full --recursive init avoids cloning large demo submodules like llama2_70b.
# Build flags:
#   --without-python-bindings  skip _ttnncpp.so (~5-7 min saved)
#   --without-distributed      skip OpenMPI (not needed for single-machine MeshDevice)
$(TT_METAL_BUILD)/lib/libtt_metal.so:
	git submodule update --init --depth 1 third_party/tt-metal
	cd $(TT_METAL_HOME) && git submodule update --init --depth 1 \
		tt_metal/third_party/tracy \
		tt_metal/third_party/tt_llk \
		tt_metal/third_party/umd
	cd $(TT_METAL_HOME) && ./build_metal.sh \
		--without-python-bindings \
		--without-distributed

setup: $(TT_METAL_BUILD)/lib/libtt_metal.so

# Engine static library
$(BUILD)/libqwen_engine.a: $(ENGINE_OBJS)
	@mkdir -p $(@D)
	ar rcs $@ $^

$(BUILD)/src/engine.o: src/engine.cpp | $(TT_METAL_BUILD)/lib/libtt_metal.so
	@mkdir -p $(@D)
	$(CXX) $(ENGINE_CXXFLAGS) -c $< -o $@

$(BUILD)/src/%.o: src/%.cpp | $(TT_METAL_BUILD)/lib/libtt_metal.so
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Main test targets (link against engine)
$(BUILD)/test_forward: src/tests/test_forward.cpp src/download.cpp $(BUILD)/libqwen_engine.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) src/tests/test_forward.cpp src/download.cpp -o $@ $(LDFLAGS) $(BUILD)/libqwen_engine.a $(TT_LIBS)

$(BUILD)/test_inference: src/tests/test_inference.cpp src/download.cpp $(BUILD)/libqwen_engine.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) src/tests/test_inference.cpp src/download.cpp -o $@ $(LDFLAGS) $(BUILD)/libqwen_engine.a $(TT_LIBS)

# Interactive chat CLI
$(BUILD)/qwen-chat: src/chat.cpp src/download.cpp $(BUILD)/libqwen_engine.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) src/chat.cpp src/download.cpp -o $@ $(LDFLAGS) $(BUILD)/libqwen_engine.a $(TT_LIBS)

chat: $(BUILD)/qwen-chat
	@env TT_METAL_RUNTIME_ROOT=$(abspath $(TT_METAL_HOME)) QUIET=1 \
		TT_METAL_OPERATION_TIMEOUT_SECONDS=$${TT_METAL_OPERATION_TIMEOUT_SECONDS:-5} \
		$(BUILD)/qwen-chat \
		$${MODEL_PATH:-unsloth/Qwen3.5-9B-GGUF:BF16} 2>/dev/null

# HTTP server with chat UI
$(BUILD)/qwen-server: src/server.cpp src/download.cpp $(BUILD)/libqwen_engine.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) src/server.cpp src/download.cpp -o $@ $(LDFLAGS) $(BUILD)/libqwen_engine.a $(TT_LIBS) -lpthread

serve: $(BUILD)/qwen-server
	@env TT_METAL_RUNTIME_ROOT=$(abspath $(TT_METAL_HOME)) QUIET=1 \
		TT_METAL_OPERATION_TIMEOUT_SECONDS=$${TT_METAL_OPERATION_TIMEOUT_SECONDS:-5} \
		$(BUILD)/qwen-server \
		-m $${MODEL_PATH:-unsloth/Qwen3.5-9B-GGUF:BF16} \
		--port $${PORT:-8888} 2>/dev/null

# BFP8_B GGUF converter — CPU-only tool (no device needed)
$(BUILD)/make_bfp8_gguf: tools/make_bfp8_gguf.cpp | $(TT_METAL_BUILD)/lib/libtt_metal.so
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -march=$(MARCH) $< -o $@ $(LDFLAGS) $(TT_LIBS)

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
	@env TT_METAL_RUNTIME_ROOT=$(abspath $(TT_METAL_HOME)) QUIET=1 \
		TT_METAL_OPERATION_TIMEOUT_SECONDS=$${TT_METAL_OPERATION_TIMEOUT_SECONDS:-5} \
		$(BUILD)/test_forward \
		$${MODEL_PATH:-unsloth/Qwen3.5-9B-GGUF:BF16} \
		"The capital of France is" 16 --raw 2>/dev/null

# Run integration test suite
test: $(BUILD)/test_inference
	@env TT_METAL_RUNTIME_ROOT=$(abspath $(TT_METAL_HOME)) \
		MODEL_PATH=$${MODEL_PATH:-unsloth/Qwen3.5-9B-GGUF:BF16} \
		QUIET=1 \
		TT_METAL_OPERATION_TIMEOUT_SECONDS=$${TT_METAL_OPERATION_TIMEOUT_SECONDS:-5} \
		$(BUILD)/test_inference 2>/dev/null

clean:
	rm -rf $(BUILD)
