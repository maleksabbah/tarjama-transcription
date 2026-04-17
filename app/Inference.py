import os
import json
import time
import torch
from transformers import (
    WhisperForConditionalGeneration,
    AutoProcessor,
    pipeline,
)

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_pipe = None


def load_model():
    """Load the fine-tuned Whisper model as a HuggingFace pipeline with chunking."""
    global _pipe
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    start = time.time()

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
        chunk_length_s=30,
        stride_length_s=(5, 5),
        return_timestamps=True,
    )

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s on {DEVICE}")
    return _pipe


def transcribe(audio_path: str, language: str = "ar", dialect: str = "auto") -> dict:
    """Transcribe an audio file of arbitrary length with per-segment timestamps."""
    if _pipe is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    generate_kwargs = {
        "language": language,
        "task": "transcribe",
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
    }

    result = _pipe(
        audio_path,
        generate_kwargs=generate_kwargs,
        return_timestamps=True,
    )

    text = (result.get("text") or "").strip()
    chunks = result.get("chunks") or []

    segments = []
    duration = 0.0
    for c in chunks:
        seg_text = (c.get("text") or "").strip()
        if not seg_text:
            continue
        ts = c.get("timestamp") or (None, None)
        start = ts[0] if ts[0] is not None else 0.0
        end = ts[1] if ts[1] is not None else start
        segments.append({
            "start": float(start),
            "end": float(end),
            "text": seg_text,
        })
        if end and end > duration:
            duration = float(end)

    # Fallback: if pipeline returned text but no usable chunks, keep a single segment
    if not segments and text:
        segments = [{"start": 0.0, "end": duration or 0.0, "text": text}]

    return {
        "text": text,
        "segments": segments,
        "duration_seconds": duration,
        "confidence": None,
    }


def save_result(result: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path
