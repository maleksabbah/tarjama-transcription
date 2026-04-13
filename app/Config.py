"""
Transcription Service Configuration
"""
import os


class Config:
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Queues
    QUEUE_TRANSCRIBE = "queue:transcribe"
    QUEUE_COMPLETED = "queue:completed"

    # Model
    MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/lora_023_20260312_181433/merged")
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/whisper-small")
    DEVICE = os.getenv("DEVICE", "cuda")

    # Inference
    LANGUAGE = "ar"
    TASK = "transcribe"
    NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM", "3"))

    # Storage
    STORAGE_BASE = os.getenv("STORAGE_BASE", "./storage")
    RESULTS_DIR = os.getenv("RESULTS_DIR", "results")


config = Config()