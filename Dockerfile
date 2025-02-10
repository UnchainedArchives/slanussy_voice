# Base image with CUDA 11.8
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# System dependencies
RUN apt update && apt install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    git \
    wget \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environments
RUN python3.9 -m venv /opt/rvc_env --prompt rvc_env && \
    python3.9 -m venv /opt/bark_env --prompt bark_env

# Upgrade pip
RUN /opt/rvc_env/bin/pip install --upgrade pip && \
    /opt/bark_env/bin/pip install --upgrade pip

# Install RVC
RUN /opt/rvc_env/bin/pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchaudio==2.0.2+cu118 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html

RUN git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /app/rvc && \
    /opt/rvc_env/bin/pip install --no-cache-dir -r /app/rvc/requirements.txt

# Download default RVC models
RUN mkdir -p /app/models/rvc && \
    wget -qO /app/models/rvc/model.pth https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth && \
    wget -qO /app/models/rvc/model.index https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.index

# Install Bark
RUN /opt/bark_env/bin/pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/cu121/torch_stable.html

RUN /opt/bark_env/bin/pip install --no-cache-dir \
    bark==0.0.0a1 \
    numpy==1.23.5 \
    noisereduce \
    soundfile \
    slowapi \
    python-multipart \
    tqdm \
    encodec \
    sentencepiece \
    resampy \
    webrtcvad

# Preload Bark models
RUN /opt/bark_env/bin/python -c "from bark.generation import preload_models; preload_models()"

# Copy app
COPY app /app

# Final setup
RUN apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    useradd -m appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/output /app/logs

USER appuser
WORKDIR /app
EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

CMD ["/opt/bark_env/bin/python", "/app/main.py"]