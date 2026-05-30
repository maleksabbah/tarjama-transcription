# transcription: app/Services/InferenceService.py
"""
faster-whisper inference, shared by both batch and live workers.
Silero VAD built in via vad_filter=True.
"""
import json
import os

import numpy as np
from faster_whisper import WhisperModel

from app.Config.Config import config


def _strip_bom_if_present(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(path, "wb") as f:
            f.write(data[3:])
        print(f"  [INFERENCE] Stripped BOM from {path}")


class InferenceService:
    def __init__(self):
        self.model: WhisperModel | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        _strip_bom_if_present(os.path.join(config.MODEL_PATH, "preprocessor_config.json"))
        _strip_bom_if_present(os.path.join(config.MODEL_PATH, "config.json"))
        print(f"  [INFERENCE] Loading faster-whisper from {config.MODEL_PATH}...")
        self.model = WhisperModel(
            config.MODEL_PATH,
            device=config.DEVICE,
            compute_type=config.COMPUTE_TYPE,
        )
        print("  [INFERENCE] Model loaded.")

    def transcribe_file(self, audio_path: str) -> dict:
        """Batch — full audio file. Returns {text, segments, duration_seconds, language}."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        segments, info = self.model.transcribe(
            audio_path,
            language="ar",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            word_timestamps=True,
        )

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

        return {
            "text": " ".join(text_parts).strip(),
            "segments": seg_list,
            "duration_seconds": float(info.duration),
            "language": info.language,
        }

    def transcribe_buffer(self, audio: np.ndarray) -> str:
        """Live — float32 mono 16kHz numpy buffer. Returns just the text."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        segments, _ = self.model.transcribe(
            audio,
            language="ar",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            word_timestamps=True,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()

    @staticmethod
    def save_result(result: dict, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)