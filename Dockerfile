ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies (base image already has clang-20)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl make && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The tt-metalium SDK is installed as part of the ttnn Python package.
# Discover paths and create a symlink tree matching our Makefile's expected layout:
#   TT_METAL_BUILD/{include,lib,libexec}
RUN set -e && TT_SDK=/app/tt-sdk && \
    mkdir -p $TT_SDK/build_Release/lib $TT_SDK/build_Release/libexec \
             $TT_SDK/build_Release/include $TT_SDK/tt_metal/impl/data_format && \
    # Find the ttnn package root \
    TTNN_PKG=$(python3 -c "import ttnn; import os; print(os.path.dirname(ttnn.__file__))") && \
    echo "TTNN package: $TTNN_PKG" && \
    # Include: tt-metalium headers are under ttnn/tt_metal/api/ \
    API_DIR="$TTNN_PKG/tt_metal/api" && \
    if [ -d "$API_DIR/tt-metalium" ]; then \
        ln -sf "$API_DIR/tt-metalium" $TT_SDK/build_Release/include/tt-metalium; \
    fi && \
    # Include: metalium-thirdparty headers \
    for thirdparty_candidate in "$API_DIR/../../../metalium-thirdparty" \
                                "$TTNN_PKG/metalium-thirdparty" \
                                "$(dirname $TTNN_PKG)/metalium-thirdparty"; do \
        if [ -d "$thirdparty_candidate" ]; then \
            ln -sf "$(realpath $thirdparty_candidate)" $TT_SDK/build_Release/include/metalium-thirdparty; \
            break; \
        fi; \
    done && \
    # Libexec: symlink tt-metalium dir (for hostdevcommon and internal headers) \
    if [ -d "$API_DIR/tt-metalium" ]; then \
        ln -sf "$API_DIR/tt-metalium" $TT_SDK/build_Release/libexec/tt-metalium; \
    fi && \
    # Libraries from ttnn build dir \
    TTNN_LIB="$TTNN_PKG/build/lib" && \
    for lib in libtt_metal.so libdevice.so libtt_stl.so; do \
        if [ -f "$TTNN_LIB/$lib" ] || [ -L "$TTNN_LIB/$lib" ]; then \
            ln -sf "$TTNN_LIB/$lib" $TT_SDK/build_Release/lib/$lib; \
        else \
            found=$(find / -name "$lib" \( -type f -o -type l \) 2>/dev/null | head -1); \
            if [ -n "$found" ]; then ln -sf "$found" $TT_SDK/build_Release/lib/$lib; fi; \
        fi; \
    done && \
    # Tracy (versioned soname in ttnn.libs/) \
    TTNN_LIBS_DIR="$(dirname $TTNN_PKG)/ttnn.libs" && \
    tracy=$(find "$TTNN_LIBS_DIR" / -name "libtracy*.so*" \( -type f -o -type l \) 2>/dev/null | head -1); \
    if [ -n "$tracy" ]; then \
        ln -sf "$tracy" $TT_SDK/build_Release/lib/libtracy.so.0.10.0; \
    fi && \
    # spdlog \
    spdlog=$(find "$TTNN_LIB" "$TTNN_LIBS_DIR" / -name "libspdlog*" \( -type f -o -type l \) 2>/dev/null | head -1); \
    if [ -n "$spdlog" ]; then \
        ln -sf "$spdlog" $TT_SDK/build_Release/lib/$(basename "$spdlog"); \
    fi && \
    # Debug: show what we found \
    echo "=== SDK shim libs ===" && ls -la $TT_SDK/build_Release/lib/ && \
    echo "=== SDK shim includes ===" && ls -la $TT_SDK/build_Release/include/ && \
    echo "=== SDK shim libexec ===" && ls -la $TT_SDK/build_Release/libexec/

# Copy project source
COPY src/ src/
COPY third_party/json.hpp third_party/json.hpp
COPY Makefile .

# Build using the pre-installed SDK.
# Touch libtt_metal.so so Make skips the auto-setup rule.
RUN touch /app/tt-sdk/build_Release/lib/libtt_metal.so && \
    make -j$(nproc) \
    TT_METAL_HOME=/app/tt-sdk \
    TT_METAL_BUILD=/app/tt-sdk/build_Release

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Model is downloaded at runtime via the binary's built-in HuggingFace resolver.
# To bake the model into the image for faster cold starts, uncomment:
# RUN mkdir -p /models && \
#     curl -L -o /models/Qwen3.5-9B-BF16.gguf \
#       "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-BF16.gguf"
# ENV MODEL_PATH=/models/Qwen3.5-9B-BF16.gguf

ENV MODEL_PATH=unsloth/Qwen3.5-9B-GGUF:BF16
ENV TT_METAL_RUNTIME_ROOT=/usr/libexec/tt-metalium
ENV QUIET=1
ENV PORT=8888

EXPOSE 8888

CMD ["/entrypoint.sh"]
