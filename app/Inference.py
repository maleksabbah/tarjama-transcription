"""
Unified transcription using faster-whisper.
Handles both batch (full audio file) and live (short numpy buffer).
Silero VAD built in via vad_filter=True.
"""
import os
import numpy as np
from faster_whisper import WhisperModel

MODEL_PATH = os.getenv("MODEL_PATH_CT2", "/app/model-ct2")

_model: WhisperModel | None = None


def _strip_bom_if_present(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(path, "wb") as f:
            f.write(data[3:])
        print(f"  [INFERENCE] Stripped BOM from {path}")


def load_model():
    global _model
    if _model is not None:
        return
    _strip_bom_if_present(os.path.join(MODEL_PATH, "preprocessor_config.json"))
    _strip_bom_if_present(os.path.join(MODEL_PATH, "config.json"))
    print(f"  [INFERENCE] Loading faster-whisper from {MODEL_PATH}...")
    _model = WhisperModel(
        MODEL_PATH,
        device="cuda",
        compute_type="float16",
    )
    print("  [INFERENCE] Model loaded.")


# Live convenience alias — same model, same underlying call
def load_model_live():
    load_model()


def transcribe(audio, dialect="auto", initial_prompt=None):
    """
    Accepts either:
      - str path to a WAV/audio file (batch)
      - np.ndarray float32 mono 16kHz audio (live)
    Returns a dict with 'text' and 'segments' for batch,
    or a plain str of joined text for live.
    """
    if _model is None:
        raise RuntimeError("Model not loaded")

    segments, info = _model.transcribe(
        audio,
        language="ar",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
        word_timestamps=True,
        initial_prompt=initial_prompt,
    )

    # Drain generator — faster-whisper returns lazily
    seg_list = []
    text_parts = []
    for seg in segments:
        t = seg.text.strip()
        if not t:
            continue
        seg_list.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": t,
        })
        text_parts.append(t)

    full_text = " ".join(text_parts).strip()

    # Live path (numpy in) — return plain string
    if isinstance(audio, np.ndarray):
        return full_text

    # Batch path (file path in) — return structured result
    return {
        "text": full_text,
        "segments": seg_list,
        "duration_seconds": float(info.duration),
        "language": info.language,
    }


# Used by Worker.py for batch
def save_result(result: dict, path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)