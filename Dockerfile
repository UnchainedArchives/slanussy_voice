# Stage 1: Base environment
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.9

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
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Stage 2: RVC installation
FROM base AS rvc

# Create RVC virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/rvc_env --prompt rvc_env

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
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    mkdir -p /app/models/rvc && \
    curl -A "Mozilla/5.0" -L -o /app/models/rvc/model.pth \
    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k/G_0.pth && \
    curl -A "Mozilla/5.0" -L -o /app/models/rvc/model.index \
    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k/f0D40k.index && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Stage 3: Bark installation
FROM base AS bark

# Create Bark virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/bark_env --prompt bark_env

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

# Disable GPU during build to avoid driver errors
ENV CUDA_VISIBLE_DEVICES=-1

# Preload models incrementally with CPU
RUN /opt/bark_env/bin/python - <<EOF
import torch
from bark.generation import load_model, models

# Move all models to CPU after loading
for model_type in ["text", "coarse", "fine", "codec"]:
    print(f"Loading model: {model_type}")
    model = load_model(model_type)

    # Handle different model types
    if model_type == "codec":
        # Codec model is a dictionary
        for key in model:
            if isinstance(model[key], torch.nn.Module):
                model[key].to("cpu")
    elif isinstance(model, tuple):
        # Handle tuple models
        for m in model:
            m.to("cpu")
    elif isinstance(model, torch.nn.Module):
        # Single model
        model.to("cpu")
    else:
        print(f"Unexpected model type for {model_type}: {type(model)}")

    models[model_type] = model
    print(f"Model {model_type} loaded and moved to CPU.")
EOF

# Stage 4: Final image
FROM base

# Remove build tools
RUN apt-get purge -y build-essential g++ git wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy components
COPY --from=rvc /app/rvc /app/rvc
COPY --from=rvc /app/models /app/models
COPY --from=rvc /opt/rvc_env /opt/rvc_env
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
    chown -R appuser:appuser /app /opt/rvc_env /opt/bark_env && \
    mkdir -p /app/output /app/logs

# Enable GPU usage at runtime by default
ENV CUDA_VISIBLE_DEVICES=0

USER appuser
WORKDIR /app
EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

CMD ["/opt/bark_env/bin/python", "/app/main.py"]