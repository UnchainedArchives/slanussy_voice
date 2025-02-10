import os
import torch
import numpy as np
import soundfile as sf
import uvicorn
import librosa
import logging
import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware import Middleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi.concurrency import run_in_threadpool
from transformers import pipeline
from bark.generation import (
    generate_text_semantic,
    generate_coarse,
    generate_fine,
    load_codec_model,
    preload_models
)
from rvc.infer import infer_vc

# Configuration with environment variables
CONFIG = {
    "workspace": os.getenv("WORKSPACE", "/app/output"),
    "max_text_length": 500,
    "rvc_sample_rate": 40000,
    "emotion_threshold": 0.5,
    "default_voice": "v2/en_speaker_6",
    "device": os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
}

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI instance with middleware
app = FastAPI(middleware=[
    Middleware("slowapi.middleware.LimiterMiddleware", limiter=limiter)
])

# Setup logging
os.makedirs('/app/logs', exist_ok=True)
logging.basicConfig(
    filename='/app/logs/error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Helper Functions ---
def download_default_model():
    """Download default RVC model if missing"""
    model_dir = Path("/app/models/rvc")
    model_dir.mkdir(exist_ok=True)

    model_files = {
        "model.pth": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
        "model.index": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.index"
    }

    for filename, url in model_files.items():
        filepath = model_dir / filename
        if not filepath.exists():
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(response.content)
                logging.info(f"Downloaded {filename}")
            except Exception as e:
                logging.error(f"Failed to download {filename}: {str(e)}")
                raise

# --- Endpoints ---
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "device": CONFIG["device"]
    }

@app.get("/models")
async def list_models():
    model_dir = Path("/app/models/rvc")
    models = [f.stem for f in model_dir.glob("*.pth") if f.is_file()]
    return {"models": models}

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.error(f"Error processing {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

# --- Core Components ---
class BarkPipeline:
    def __init__(self):
        self.codec = None
        self.emo_detector = None
        self.init_models()

    def init_models(self):
        logging.info("Initializing Bark models")
        preload_models(
            text_use_gpu=True,
            text_use_small=False,
            coarse_use_gpu=True,
            fine_use_gpu=True,
            codec_use_gpu=True
        )
        self.codec = load_codec_model(use_gpu=CONFIG["device"] == "cuda")
        self.emo_detector = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=CONFIG["device"]
        )

    def process_emotion(self, text):
        result = self.emo_detector(text[:CONFIG["max_text_length"]], truncation=True)[0]
        label = result["label"].lower().replace("_", "")
        return label if result["score"] >= CONFIG["emotion_threshold"] else "neutral"

    async def generate_speech(self, text, emotion):
        params = {**GENERATION_PARAMS["high_quality"], **EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])}

        # Generate audio in background thread
        return await run_in_threadpool(self._generate_speech, text, params)

    def _generate_speech(self, text, params):
        semantic = generate_text_semantic(
            text,
            history_prompt=params["voice_preset"],
            temp=params["semantic_temp"],
            top_k=params["semantic_top_k"],
            top_p=params["semantic_top_p"]
        )

        coarse = generate_coarse(
            semantic,
            history_prompt=params["voice_preset"],
            temp=params["coarse_temp"],
            max_coarse_history=params["max_coarse_history"]
        )

        fine = generate_fine(
            coarse,
            history_prompt=params["voice_preset"],
            temp=params["fine_temp"]
        )

        audio = self.post_process(self.codec.decode(fine), params)
        return audio

    def post_process(self, audio, params):
        shifted = librosa.effects.pitch_shift(
            audio,
            sr=24000,
            n_steps=params["pitch_shift"]
        )
        stretched = librosa.effects.time_stretch(
            shifted,
            rate=params["speed_factor"]
        )
        breath = np.random.normal(0, 0.001, int(0.2 * 24000))
        insert_point = len(stretched) // 3
        stretched[insert_point:insert_point+len(breath)] += breath
        return stretched

class VCProcessor:
    def __init__(self, model_name="model"):
        model_name = Path(model_name).stem  # Prevent path traversal
        self.model_path = f"/app/models/rvc/{model_name}.pth"
        self.index_path = f"/app/models/rvc/{model_name}.index"
        self._validate_paths()

    def _validate_paths(self):
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model {self.model_path} not found")
        if not os.path.exists(self.index_path):
            raise ValueError(f"Index {self.index_path} not found")

    async def convert(self, audio):
        return await run_in_threadpool(self._convert, audio)

    def _convert(self, audio):
        audio_40k = librosa.resample(audio, orig_sr=24000, target_sr=CONFIG["rvc_sample_rate"])
        return infer_vc(
            audio=audio_40k,
            model_path=self.model_path,
            index_path=self.index_path,
            sr=CONFIG["rvc_sample_rate"],
            device=CONFIG["device"]
        )

# --- Application Setup ---
GENERATION_PARAMS = {
    "high_quality": {
        "semantic_temp": 0.7,
        "semantic_top_k": 50,
        "semantic_top_p": 0.95,
        "coarse_temp": 0.7,
        "fine_temp": 0.5,
        "max_coarse_history": 630,
        "breathing_interval": 8
    }
}

EMOTION_CONFIG = {
    # ... (same as original) ...
}

# Initialize components
pipeline = BarkPipeline()

@app.post("/generate")
@limiter.limit("10/minute")
async def generate_endpoint(
    request: Request,
    text: str = Query(..., min_length=1, max_length=CONFIG["max_text_length"]),
    model: str = Query("model", regex=r'^[a-zA-Z0-9_-]+$')
):
    try:
        # Validate model exists
        vc_processor = VCProcessor(model)

        # Process emotion
        emotion = pipeline.process_emotion(text)

        # Generate and process audio
        raw_audio = await pipeline.generate_speech(text, emotion)
        converted_audio = await vc_processor.convert(raw_audio)

        # Save output
        output_path = os.path.join(CONFIG["workspace"], f"{hash(text)}.wav")
        sf.write(output_path, converted_audio, CONFIG["rvc_sample_rate"])

        return FileResponse(
            output_path,
            media_type="audio/wav",
            headers={"Cache-Control": "max-age=3600"}
        )

    except ValueError as e:
        raise HTTPException(404, str(e))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(503, "Insufficient GPU memory")
    except Exception as e:
        logging.error(f"Generation error: {str(e)}", exc_info=True)
        raise HTTPException(500, "Internal processing error")

if __name__ == "__main__":
    download_default_model()
    uvicorn.run(app, host="0.0.0.0", port=5001)