# transcription: app/Config/Config.py
"""
Transcription worker config.
"""
import os


class Config:
    # Kafka (batch)
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    TOPIC_TRANSCRIBE_TASKS: str = os.getenv(
        "TOPIC_TRANSCRIBE_TASKS", "tarjama.transcribe.tasks",
    )
    TOPIC_COMPLETED: str = os.getenv("TOPIC_COMPLETED", "tarjama.completed")
    GROUP_TRANSCRIBE_WORKER: str = os.getenv(
        "GROUP_TRANSCRIBE_WORKER", "tarjama.transcribe",
    )

    # Redis (live)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Storage HTTP
    STORAGE_URL: str = os.getenv("STORAGE_URL", "http://storage:8002")

    # Model
    MODEL_PATH: str = os.getenv("MODEL_PATH_CT2", "/app/model-ct2")
    DEVICE: str = os.getenv("DEVICE", "cuda")
    COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "float16")


config = Config()