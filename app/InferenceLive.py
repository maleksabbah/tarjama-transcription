"""
Live inference using faster-whisper.
Silero VAD is built in via vad_filter=True — silent chunks return "".
Accepts raw Float32 numpy audio (no WAV round-trip).
"""
import os
import numpy as np
from faster_whisper import WhisperModel

MODEL_PATH_CT2 = os.getenv("MODEL_PATH_CT2", "/app/model-ct2")

_model: WhisperModel | None = None


def load_model_live():
    global _model
    if _model is not None:
        return
    print(f"  [LIVE] Loading faster-whisper from {MODEL_PATH_CT2}...")
    _model = WhisperModel(
        MODEL_PATH_CT2,
        device="cuda",
        compute_type="float16",
    )
    print("  [LIVE] Model loaded.")


def transcribe_live(audio: np.ndarray) -> str:
    """
    Transcribe a live buffer.
    Returns concatenated text of detected speech, or "" if VAD found no speech.
    """
    if _model is None:
        raise RuntimeError("Live model not loaded")

    segments, _info = _model.transcribe(
        audio,
        language="ar",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
    )

    # segments is a generator — drain it
    text_parts = [seg.text for seg in segments]
    return " ".join(p.strip() for p in text_parts).strip()