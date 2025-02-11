FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    tzdata \
    build-essential \
    g++ \
    git \
    wget \
    libsndfile1 \
    ffmpeg \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    && rm -rf /var/lib/apt/lists/*

# Virtual envs
RUN python3.9 -m venv /opt/rvc_env --prompt rvc_env && \
    python3.9 -m venv /opt/bark_env --prompt bark_env

# Upgrade pip
RUN /opt/rvc_env/bin/pip install pip==23.3.1 && \
    /opt/bark_env/bin/pip install --upgrade pip

# RVC setup
RUN git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /app/rvc && \
    sed -i 's/fairseq==0.12.2/fairseq @ git+https:\/\/github.com\/facebookresearch\/fairseq.git@main/' /app/rvc/requirements.txt && \
    sed -i '/hydra-core/d' /app/rvc/requirements.txt && \
    /opt/rvc_env/bin/pip install --no-cache-dir -r /app/rvc/requirements.txt

RUN /opt/rvc_env/bin/pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchaudio==2.0.2+cu118 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Download RVC models
RUN apt update && apt install -y curl && \
    mkdir -p /app/models/rvc && \
    curl -A "Mozilla/5.0" -L -o /app/models/rvc/model.pth https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k/G_0.pth && \
    curl -A "Mozilla/5.0" -L -o /app/models/rvc/model.index https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k/f0D40k.index && \
    apt purge -y curl && apt autoremove -y

# Bark setup
RUN /opt/bark_env/bin/pip install --no-cache-dir \
    git+https://github.com/suno-ai/bark.git \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    numpy==1.23.5 \
    noisereduce \
    soundfile \
    slowapi \
    python-multipart \
    tqdm \
    encodec==0.1.1 \
    sentencepiece \
    resampy \
    webrtcvad \
    -f https://download.pytorch.org/whl/cu121/torch_stable.html

RUN /opt/bark_env/bin/python -c "from bark.generation import preload_models; preload_models()"

COPY app /app

# Finalize
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