ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install clang-20 and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget gnupg ca-certificates curl make && \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | \
        tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc > /dev/null && \
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" \
        > /etc/apt/sources.list.d/llvm-20.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends clang-20 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set up SDK shim from pre-installed tt-metalium.
# The base image has the SDK installed system-wide. We create a symlink tree
# that matches the directory layout our Makefile expects:
#   TT_METAL_BUILD/{include,lib,libexec}
#   TT_METAL_HOME/tt_metal/impl/data_format
RUN TT_SDK=/app/tt-sdk && \
    mkdir -p $TT_SDK/build_Release/lib $TT_SDK/tt_metal/impl/data_format && \
    ln -sf /usr/include $TT_SDK/build_Release/include && \
    mkdir -p $TT_SDK/build_Release/libexec && \
    ln -sf /usr/libexec/tt-metalium $TT_SDK/build_Release/libexec/tt-metalium && \
    for lib in libtt_metal.so libtracy.so libdevice.so libtt_stl.so libspdlog.a; do \
        src=$(find /usr/lib -name "${lib}*" \( -type f -o -type l \) 2>/dev/null | head -1); \
        if [ -n "$src" ]; then \
            ln -sf "$src" $TT_SDK/build_Release/lib/$lib; \
            case "$lib" in libtracy.so) \
                versioned=$(find /usr/lib -name "libtracy.so.0*" \( -type f -o -type l \) 2>/dev/null | head -1); \
                if [ -n "$versioned" ]; then \
                    ln -sf "$versioned" $TT_SDK/build_Release/lib/$(basename "$versioned"); \
                fi ;; \
            esac; \
        fi; \
    done && \
    echo "=== SDK shim ===" && ls -la $TT_SDK/build_Release/lib/

# Copy project source
COPY src/ src/
COPY third_party/json.hpp third_party/json.hpp
COPY Makefile .

# Build using the pre-installed SDK
RUN make -j$(nproc) \
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
