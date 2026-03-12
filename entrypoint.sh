#!/bin/bash

PORT="${SERVER_PORT:-${PORT:-8888}}"
MODEL="${MODEL_PATH:-vibekernels/Qwen3.5-9B-GGUF:BFP8B-tiled}"
CTX_SIZE="${CTX_SIZE:-4096}"

# Start SSH server if a public key is provided
if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 644 /root/.ssh/authorized_keys
    /usr/sbin/sshd
    echo "SSH server started on port 22"
fi

# Run qwen-server in a restart loop so SSH remains available if it exits
while true; do
    echo "Starting qwen-server on port $PORT (model: $MODEL, ctx: $CTX_SIZE)"
    /app/build/qwen-server \
        -m "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --ctx-size "$CTX_SIZE"
    echo "qwen-server exited (status $?), restarting in 5 seconds..."
    sleep 5
done
