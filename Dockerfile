FROM --platform=linux/amd64 nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

ARG REPO_URL=https://github.com/vibekernels/qwen3.5-9b-1x5090.git
ARG COMMIT_SHA=main

# Install essentials + SSH
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server sudo git curl wget vim tmux build-essential ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# SSH setup
RUN mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh \
    && rm -f /etc/ssh/ssh_host_* \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# Clone repo at the exact commit that triggered the build
RUN git clone "$REPO_URL" /root/qwen3.5-9b-1x5090 \
    && cd /root/qwen3.5-9b-1x5090 \
    && git checkout "$COMMIT_SHA"

# Build the inference binary
RUN cd /root/qwen3.5-9b-1x5090 && make -j$(nproc)

WORKDIR /root/qwen3.5-9b-1x5090

# Startup script: configure SSH, download model, start server
RUN printf '#!/bin/bash\n\
set -e\n\
ssh-keygen -A\n\
if [ -n "$PUBLIC_KEY" ]; then\n\
  mkdir -p /root/.ssh\n\
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys\n\
  chmod 600 /root/.ssh/authorized_keys\n\
fi\n\
# Snapshot container env vars so SSH sessions inherit them.\n\
printenv | grep -E "^[A-Z_][A-Z0-9_]*=" | grep -v "^PUBLIC_KEY=" | \\\n\
  awk -F = '"'"'{ val=$0; sub(/^[^=]*=/, "", val); print "export " $1 "=\\\"" val "\\\"" }'"'"' > /etc/rp_environment\n\
grep -q "source /etc/rp_environment" /root/.bashrc 2>/dev/null || \\\n\
  echo "source /etc/rp_environment" >> /root/.bashrc\n\
/usr/sbin/sshd\n\
\n\
# Parse HF_MODEL (format: org/repo-GGUF:quant)\n\
HF_MODEL="${HF_MODEL:-unsloth/Qwen3.5-9B-GGUF:BF16}"\n\
REPO="${HF_MODEL%%:*}"\n\
QUANT="${HF_MODEL##*:}"\n\
REPO_NAME="${REPO##*/}"\n\
FILE_STEM="${REPO_NAME%%-GGUF}"\n\
FILENAME="${FILE_STEM}-${QUANT}.gguf"\n\
MODEL_PATH="/workspace/models/${FILENAME}"\n\
\n\
# Download model weights if not already present\n\
if [ ! -f "$MODEL_PATH" ]; then\n\
  echo "Downloading ${FILENAME} from ${REPO}..."\n\
  mkdir -p /workspace/models\n\
  wget -O "$MODEL_PATH" \\\n\
    "https://huggingface.co/${REPO}/resolve/main/${FILENAME}"\n\
  echo "Download complete."\n\
fi\n\
\n\
# Start the inference server\n\
exec /root/qwen3.5-9b-1x5090/qwen-server -m "$MODEL_PATH" --host 0.0.0.0 --port 8080\n\
' > /start.sh && chmod +x /start.sh

EXPOSE 22 8080

CMD ["/start.sh"]
