"""
Live inference using faster-whisper.
Silero VAD is built in via vad_filter=True.
Auto-strips UTF-8 BOM from preprocessor_config.json to prevent silent
fallback to 80-mel (our fine-tune is 128-mel).
"""
import os
import numpy as np
from faster_whisper import WhisperModel

MODEL_PATH_CT2 = os.getenv("MODEL_PATH_CT2", "/app/model-ct2")

_model: WhisperModel | None = None


def _strip_bom_if_present(path: str) -> None:
    """Remove UTF-8 BOM from a file if it has one. Idempotent."""
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(path, "wb") as f:
            f.write(data[3:])
        print(f"  [LIVE] Stripped BOM from {path}")


def load_model_live():
    global _model
    if _model is not None:
        return

    # Ensure config files have no UTF-8 BOM (breaks faster-whisper silently)
    _strip_bom_if_present(os.path.join(MODEL_PATH_CT2, "preprocessor_config.json"))
    _strip_bom_if_present(os.path.join(MODEL_PATH_CT2, "config.json"))

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

    text_parts = [seg.text for seg in segments]
    return " ".join(p.strip() for p in text_parts).strip()