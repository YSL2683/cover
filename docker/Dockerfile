# ==============================================================================
# Dockerfile for ResFiT & DexMG RL Training Environment
# Multi-GPU support: RTX 3090 (sm_86) & RTX 5090 (sm_120) with Headless MuJoCo EGL
# ==============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Set up system dependencies for MuJoCo, EGL rendering, OpenGL, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    bzip2 \
    patchelf \
    ffmpeg \
    rsync \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libglfw3 \
    libegl1-mesa-dev \
    libegl1 \
    libgles2 \
    libglvnd0 \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge3 (conda-forge based, no ToS prompts)
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh && \
    $CONDA_DIR/bin/conda clean -afy

# Create conda environment 'cover' with Python 3.10
RUN $CONDA_DIR/bin/conda create -n cover python=3.10 -y && \
    $CONDA_DIR/bin/conda clean -afy

# Configure Environment Variables
ENV PATH=$CONDA_DIR/envs/cover/bin:$CONDA_DIR/bin:$PATH
ENV CONDA_DEFAULT_ENV=cover

# Environment flags for MuJoCo EGL, OpenMP and NVIDIA Container Runtime
ENV MUJOCO_GL=egl
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

# Backward-compatibility symlinks for existing training shell scripts
RUN mkdir -p /home/moai/miniconda3/envs && \
    ln -s $CONDA_DIR/envs/cover /home/moai/miniconda3/envs/cover && \
    mkdir -p /home/moai/miniconda3/etc/profile.d && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /home/moai/miniconda3/etc/profile.d/conda.sh && \
    mkdir -p /home/ysl2683/anaconda3/envs && \
    ln -s $CONDA_DIR/envs/cover /home/ysl2683/anaconda3/envs/cover && \
    mkdir -p /home/ysl2683/anaconda3/etc/profile.d && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /home/ysl2683/anaconda3/etc/profile.d/conda.sh

# Install Python requirements
WORKDIR /tmp/requirements
COPY docker/requirements_docker.txt /tmp/requirements/
RUN pip install --no-cache-dir -r requirements_docker.txt

# Clone and install dependencies into /opt/deps at exact commit hashes
WORKDIR /opt/deps

# 1. robosuite
RUN git clone https://github.com/ARISE-Initiative/robosuite.git /opt/deps/robosuite && \
    git -C /opt/deps/robosuite checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa && \
    pip install --no-cache-dir --no-deps -e /opt/deps/robosuite && \
    python /opt/deps/robosuite/robosuite/scripts/setup_macros.py

# 2. lerobot
RUN git clone https://github.com/huggingface/lerobot.git /opt/deps/lerobot && \
    git -C /opt/deps/lerobot checkout 69901b9b6a2300914ca3de0ea14b6fa6e0203bd4 && \
    pip install --no-cache-dir --no-deps -e /opt/deps/lerobot

# 3. dexmimicgen
RUN git clone https://github.com/NVlabs/dexmimicgen.git /opt/deps/dexmimicgen && \
    git -C /opt/deps/dexmimicgen checkout e606f36a38b1d4ba8f56d06d6c0cd059b20ebbaf && \
    pip install --no-cache-dir --no-deps -e /opt/deps/dexmimicgen

# 4. mimicgen
RUN git clone https://github.com/NVlabs/mimicgen.git /opt/deps/mimicgen && \
    git -C /opt/deps/mimicgen checkout 72bd767c255545f462e7ccfb2731f2e5d4c1d9bb && \
    pip install --no-cache-dir --no-deps -e /opt/deps/mimicgen

# Set default working directory
WORKDIR /workspace/cover
ENV PYTHONPATH=/workspace/cover

# Default command
CMD ["/bin/bash"]
