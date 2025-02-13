##### -----  Stage 1: Base Environment Setup ----- #####
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.9

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata build-essential g++ git wget libsndfile1 ffmpeg && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/log/* /var/cache/* /etc/apt/sources.list.d/deadsnakes-ppa* /usr/share/doc /usr/share/man
    
##### -----  Stage 2: RVC installation ----- #####
FROM base AS rvc

# Create RVC virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/rvc_env --prompt rvc_env

# Install RVC dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/rvc_env/bin/pip install pip==23.3.1 && \
    git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /app/rvc && \
    sed -i '/hydra-core/d' /app/rvc/requirements.txt && \
    /opt/rvc_env/bin/pip install hydra-core==1.1.0 && \
    find /opt/rvc_env -type d -name __pycache__ -exec rm -rf {} + && \
    /opt/rvc_env/bin/pip cache purge && \
    rm -rf /tmp/* /root/.cache/pip && \
    rm -rf /app/rvc/.git /app/rvc/docs

# Install RVC requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/rvc_env/bin/pip install --no-cache-dir -r /app/rvc/requirements.txt && \
    find /opt/rvc_env -type d -name __pycache__ -exec rm -rf {} + && \
    /opt/rvc_env/bin/pip cache purge && \
    rm -rf /tmp/* /root/.cache/pip

# Install Torch and cleans
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/rvc_env/bin/pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchaudio==2.0.2+cu118 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html && \
    find /opt/rvc_env -type d -name __pycache__ -exec rm -rf {} + && \
    /opt/rvc_env/bin/pip cache purge && \
    rm -rf /tmp/* /root/.cache/pip

# Download models and cleans
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    mkdir -p /app/models/rvc && \
    curl -A "Mozilla/5.0" -L -o /app/models/rvc/model.pth \
    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k/G_0.pth && \
    curl -A "Mozilla/5.0" -L -o /app/models/rvc/model.index \
    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k/f0D40k.index && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

##### -----  Stage 3: Bark installation ----- #####
FROM base AS bark

# Create Bark virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/bark_env --prompt bark_env

# Install all dependencies in a single layer with CUDA 11.8 compatibility (+cleanup)
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/bark_env/bin/pip install --no-cache-dir \
    git+https://github.com/suno-ai/bark.git \
    torch==2.0.1+cu118 \
    torchaudio==2.0.2+cu118 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html \
    librosa==0.10.1 \
    numpy==1.23.5 \
    soundfile==0.12.1 \
    noisereduce==0.0.14 \
    encodec==0.1.1 \
    uvicorn==0.24.0 \
    slowapi==0.1.7 \
    python-multipart==0.0.6 \
    tqdm==4.66.1 \
    sentencepiece==0.1.99 \
    resampy==0.4.2 \
    webrtcvad==2.0.10 && \
    find /opt/bark_env -type d -name __pycache__ -exec rm -rf {} + && \
    /opt/bark_env/bin/pip cache purge && \
    rm -rf /tmp/* /root/.cache/pip

# Disable GPU during build to avoid driver errors
ENV CUDA_VISIBLE_DEVICES=-1

# Load models with CPU
RUN /opt/bark_env/bin/python - <<EOF
import os
import torch
from bark.generation import load_model, models

os.environ["XDG_CACHE_HOME"] = "/app/models/cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Redundant but safe

for model_type in ["text", "coarse", "fine", "codec"]:
    print(f"Loading {model_type} model")
    models[model_type] = load_model(model_type, device='cpu')

    # Codec model special handling
    if model_type == "codec" and isinstance(models[model_type], dict):
        for k in models[model_type]:
            component = models[model_type][k]
            if isinstance(component, torch.nn.Module):
                models[model_type][k] = component.cpu()

    # Post-load cleanup
    torch.cuda.empty_cache()
EOF

# Re-enable GPU for runtime
ENV CUDA_VISIBLE_DEVICES=0

# Additional cleanup
RUN rm -rf /app/models/cache/huggingface/transformers /tmp/* /root/.cache/pip && \
    find /opt/bark_env -name "*.so" -delete

##### -----  Stage 4: Final image  ----- #####
FROM base

# Remove build tools
RUN apt-get purge -y build-essential g++ git wget && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
RUN useradd -m appuser && \
    chown -R appuser:appuser /app /opt/rvc_env /opt/bark_env && \
    mkdir -p /app/output /app/logs

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV WORKSPACE=/workspace \
    HF_HOME=/workspace/huggingface \
    TMPDIR=/workspace/tmp \
    PYTHONUNBUFFERED=1

USER appuser
WORKDIR /app
EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

    ENTRYPOINT ["/app/entrypoint.sh"]