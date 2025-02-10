# Base image with CUDA 11.8
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# System dependencies, including libsndfile1 for soundfile support
RUN apt update && apt install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    git \
    wget \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environments with descriptive prompts
RUN python3.9 -m venv /opt/rvc_env --prompt rvc_env
RUN python3.9 -m venv /opt/bark_env --prompt bark_env

# Upgrade pip in both environments and purge cache
RUN /opt/rvc_env/bin/pip install --upgrade pip && /opt/rvc_env/bin/pip cache purge
RUN /opt/bark_env/bin/pip install --upgrade pip && /opt/bark_env/bin/pip cache purge

# Install RVC dependencies using CUDA 11.8 builds
RUN /opt/rvc_env/bin/pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
RUN git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /app/rvc
RUN /opt/rvc_env/bin/pip install --no-cache-dir -r /app/rvc/requirements.txt
RUN mkdir -p /app/models/rvc && \
    wget -O /app/models/rvc/model.pth https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth && \
    wget -O /app/models/rvc/model.index https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.index

# Install Bark dependencies
RUN /opt/bark_env/bin/pip install --no-cache-dir \
    slowapi \
    python-multipart \
    tqdm \
    python-dotenv \
    encodec \
    sentencepiece \
    resampy \
    webrtcvad
RUN /opt/bark_env/bin/pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
RUN /opt/bark_env/bin/pip install bark numpy==1.23.5 noisereduce soundfile

# Pre-download Bark models
RUN /opt/bark_env/bin/python -c "from bark.generation import preload_models; preload_models()"

# Copy application code
COPY app /app

# Cleanup after installations
RUN apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create a non-root user and assign ownership
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Create output directories
RUN mkdir -p /app/output /app/logs

# Set working directory
WORKDIR /app

# Expose API port (only need to declare once)
EXPOSE 5001

# Start the application using the Bark environment
CMD ["/opt/bark_env/bin/python", "/app/main.py"]