# Stage 1: Base environment
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common tzdata build-essential g++ \
    git wget libsndfile1 ffmpeg nano sudo && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 python3.9-venv python3.9-dev python3.9-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create virtual environments
RUN python3.9 -m venv /opt/rvc_env --prompt rvc_env && \
    python3.9 -m venv /opt/bark_env --prompt bark_env

# Stage 2: RVC installation
FROM base AS rvc

# Install RVC dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/rvc_env/bin/pip install pip==23.3.1

# Clone and modify RVC
RUN git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /app/rvc && \
    sed -i '/hydra-core/d' /app/rvc/requirements.txt

RUN /opt/rvc_env/bin/pip install hydra-core==1.1.0

# Clean cache after hydra install
RUN /opt/rvc_env/bin/pip cache purge && \
    rm -rf /root/.cache/pip /tmp/*

# Install RVC requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/rvc_env/bin/pip install --no-cache-dir -r /app/rvc/requirements.txt

# Clean cache after requirements
RUN /opt/rvc_env/bin/pip cache purge && \
    rm -rf /root/.cache/pip /tmp/*

# Install Torch
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/rvc_env/bin/pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchaudio==2.0.2+cu118 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Clean cache after torch
RUN /opt/rvc_env/bin/pip cache purge && \
    rm -rf /root/.cache/pip /tmp/*

# Download models
RUN apt-get update && apt-get install -y curl && \
    mkdir -p /app/models/rvc && \
    curl -A "Mozilla/5.0" -L -o /app/models/rvc/model.pth \
    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k/G_0.pth && \
    curl -A "Mozilla/5.0" -L -o /app/models/rvc/model.index \
    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k/f0D40k.index && \
    apt-get purge -y curl && apt-get autoremove -y

# Stage 3: Bark installation
FROM base AS bark

# Install core dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/bark_env/bin/pip install --upgrade pip uvicorn && \
    /opt/bark_env/bin/pip install --no-cache-dir \
    git+https://github.com/suno-ai/bark.git \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/cu121/torch_stable.html


# Clean cache between steps
RUN /opt/bark_env/bin/pip cache purge && \
    rm -rf /tmp/* /root/.cache/pip

# Install secondary dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/bark_env/bin/pip install --no-cache-dir \
    numpy==1.23.5 \
    noisereduce \
    soundfile \
    slowapi \
    python-multipart \
    tqdm \
    encodec==0.1.1 \
    sentencepiece \
    resampy \
    webrtcvad

# Preload models
RUN /opt/bark_env/bin/python -c "from bark.generation import preload_models; preload_models(use_gpu=False)"

# Stage 4: Final image
FROM base

# Remove build tools
RUN apt-get purge -y build-essential g++ git wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy components
COPY --from=rvc /app/rvc /app/rvc
COPY --from=rvc /app/models /app/models
COPY --from=bark /opt/bark_env /opt/bark_env

# Clean Python caches
RUN find /opt -type d -name __pycache__ -exec rm -rf {} + && \
    find /app -type d -name __pycache__ -exec rm -rf {} +

# Application code
COPY app /app

# Final setup
RUN apt-get clean && \
    rm -rf /tmp/* /var/tmp/* /root/.cache && \
    useradd -m appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/output /app/logs && \
    echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER appuser
WORKDIR /app
EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

CMD ["/opt/bark_env/bin/python", "/app/main.py"]