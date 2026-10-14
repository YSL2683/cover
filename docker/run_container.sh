#!/usr/bin/env bash
# ==============================================================================
# Helper script to launch the Cover RL Docker container with full GPU acceleration
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="${DOCKER_IMAGE_NAME:-cover:latest}"
CONTAINER_NAME="${DOCKER_CONTAINER_NAME:-cover_training_$(date +%s)}"
GPU_DEVICE="${GPU_DEVICE:-all}"

# Check for nvidia-container-toolkit / docker GPU support
if ! docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "[WARNING] Testing GPU access in Docker: 'nvidia/cuda:12.8.0-base-ubuntu22.04' test failed or not cached."
    echo "[INFO] Attempting to continue using '--gpus all'. Ensure NVIDIA Container Toolkit is installed."
fi

# Wandb credentials mount if available
WANDB_MOUNT=()
if [ -f "$HOME/.netrc" ]; then
    WANDB_MOUNT+=("-v" "$HOME/.netrc:/root/.netrc:ro")
fi

# HuggingFace cache mount if available
HF_MOUNT=()
if [ -d "$HOME/.cache/huggingface" ]; then
    HF_MOUNT+=("-v" "$HOME/.cache/huggingface:/root/.cache/huggingface")
fi

# Git config mount if available
GIT_MOUNT=()
if [ -f "$HOME/.gitconfig" ]; then
    GIT_MOUNT+=("-v" "$HOME/.gitconfig:/root/.gitconfig:ro")
fi

# Antigravity CLI (agy) binary mount if available
AGY_MOUNT=()
AGY_BIN_PATH=""
if command -v agy &>/dev/null; then
    AGY_BIN_PATH="$(command -v agy)"
elif [ -f "$HOME/.local/bin/agy" ]; then
    AGY_BIN_PATH="$HOME/.local/bin/agy"
elif [ -f "/usr/local/bin/agy" ]; then
    AGY_BIN_PATH="/usr/local/bin/agy"
fi

if [ -n "$AGY_BIN_PATH" ]; then
    AGY_MOUNT+=("-v" "$AGY_BIN_PATH:/usr/local/bin/agy:ro")
fi

# Antigravity / Gemini config, credentials, and session mount if available
if [ -d "$HOME/.gemini" ]; then
    AGY_MOUNT+=("-v" "$HOME/.gemini:/root/.gemini")
fi

echo "=================================================="
echo "Starting Cover RL Docker Container"
echo "Image           : $IMAGE_NAME"
echo "Container Name  : $CONTAINER_NAME"
echo "Workspace Mount : $WORKSPACE_ROOT -> /workspace/cover"
echo "GPU Devices     : $GPU_DEVICE"
if [ -n "$AGY_BIN_PATH" ]; then
    echo "AGY CLI Mount   : $AGY_BIN_PATH -> /usr/local/bin/agy"
fi
if [ -d "$HOME/.gemini" ]; then
    echo "AGY Session     : $HOME/.gemini -> /root/.gemini"
fi
echo "=================================================="

# Default to bash if no arguments provided
CMD_ARGS=("$@")
if [ ${#CMD_ARGS[@]} -eq 0 ]; then
    CMD_ARGS=("bash")
fi

# Run container with interactive TTY, all GPUs, 32GB shared memory, and host IPC
docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --gpus "$GPU_DEVICE" \
    --ipc host \
    --shm-size 32G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e MUJOCO_GL=egl \
    -e PYTHONUNBUFFERED=1 \
    -v "$WORKSPACE_ROOT:/workspace/cover" \
    -w /workspace/cover \
    "${WANDB_MOUNT[@]}" \
    "${HF_MOUNT[@]}" \
    "${GIT_MOUNT[@]}" \
    "${AGY_MOUNT[@]}" \
    "$IMAGE_NAME" \
    "${CMD_ARGS[@]}"
