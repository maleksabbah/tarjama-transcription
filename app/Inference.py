import os, json, time, torch
import soundfile as sf
import numpy as np
from transformers import WhisperForConditionalGeneration, AutoProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_processor = None

def load_model():
    global _model, _processor
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    start = time.time()
    _model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16
    ).to(DEVICE)
    _processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f"Model loaded in {time.time()-start:.1f}s on {DEVICE}")

def transcribe(audio_path, language="ar", dialect="auto"):
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    inputs = _processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        ids = _model.generate(input_features, language="ar", task="transcribe",
            no_repeat_ngram_size=3, repetition_penalty=1.2)
    text = _processor.batch_decode(ids, skip_special_tokens=True)[0]
    return {"text": text.strip(), "segments": [], "duration_seconds": 0, "confidence": None}

def save_result(result, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)