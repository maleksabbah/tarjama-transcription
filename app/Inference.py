import os
import time
import logging
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


class InferenceEngine:
    def __init__(self):
        self.pipe = None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading model from {MODEL_PATH} on {DEVICE} ({TORCH_DTYPE})")
        start = time.time()

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(DEVICE)

        processor = AutoProcessor.from_pretrained(MODEL_PATH)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
        )

        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")

    def transcribe(self, audio_path: str, language: str = "ar") -> dict:
        """
        Transcribe an audio file and return segments with timestamps.

        Returns:
            {
                "text": str,
                "segments": [{"timestamp": [start, end], "text": str}, ...]
            }
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded")

        logger.info(f"Transcribing: {audio_path}")
        start = time.time()

        result = self.pipe(
            audio_path,
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=(5, 5),
            generate_kwargs={
                "language": language,
                "task": "transcribe",
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.2,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": False,
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            },
        )

        elapsed = time.time() - start
        logger.info(f"Transcription done in {elapsed:.1f}s")

        # result["chunks"] contains real Whisper segment timestamps
        segments = []
        full_text = result.get("text", "").strip()

        for chunk in result.get("chunks", []):
            ts = chunk.get("timestamp", [None, None])
            segments.append({
                "timestamp": [ts[0], ts[1]],
                "text": chunk.get("text", "").strip(),
            })

        return {
            "text": full_text,
            "segments": segments,
            "duration_seconds": elapsed,
        }


# Singleton
_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def transcribe(audio_path: str, language: str = "ar") -> dict:
    return get_engine().transcribe(audio_path, language)