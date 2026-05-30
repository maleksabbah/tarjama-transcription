# transcription: app/main.py
"""
Transcription worker entrypoint.
Loads the model once, then runs the batch (Kafka) and live (Redis) workers
as concurrent asyncio tasks.
"""
import asyncio

import httpx

from app.Config.Config import config
from app.Config.Kafka import get_producer, make_consumer, close_producer
from app.Config.Redis import get_text_client, get_binary_client, close_redis
from app.Repositories import (
    EventConsumer,
    EventPublisher,
    LiveSessionRepository,
    S3Client,
    StorageClient,
)
from app.Services import InferenceService, BatchWorkerService, LiveWorkerService


async def main() -> None:
    print("Starting Transcription Service...")

    # ── Shared model (loaded once, used by both workers) ────────────
    inference = InferenceService()
    inference.load()

    # ── Batch deps (Kafka + S3 + storage HTTP) ──────────────────────
    producer = await get_producer()
    publisher = EventPublisher(producer)

    consumer = EventConsumer(
        make_consumer(
            topics=[config.TOPIC_TRANSCRIBE_TASKS],
            group_id=config.GROUP_TRANSCRIBE_WORKER,
        )
    )

    s3 = S3Client()
    http_client = httpx.AsyncClient(timeout=10.0)
    storage = StorageClient(http_client)

    batch = BatchWorkerService(
        consumer=consumer,
        publisher=publisher,
        s3=s3,
        storage=storage,
        inference=inference,
    )

    # ── Live deps (Redis only) ──────────────────────────────────────
    text_client = await get_text_client()
    binary_client = await get_binary_client()
    sessions = LiveSessionRepository(text_client, binary_client)

    live = LiveWorkerService(sessions=sessions, inference=inference)

    # ── Run both ────────────────────────────────────────────────────
    print("Transcription Service ready.")
    batch_task = asyncio.create_task(batch.run())
    live_task = asyncio.create_task(live.run())

    try:
        await asyncio.gather(batch_task, live_task)
    except KeyboardInterrupt:
        pass
    finally:
        batch_task.cancel()
        live_task.cancel()
        await http_client.aclose()
        await close_producer()
        await close_redis()
        print("Transcription Service stopped.")


if __name__ == "__main__":
    asyncio.run(main())