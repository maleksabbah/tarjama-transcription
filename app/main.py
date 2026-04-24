"""
Transcription Service entry point.
Loads models, starts workers, handles shutdown.
"""
import asyncio
from app import Redis_client as rc
from app.Inference import load_model
from app.InferenceLive import load_model_live
from app.BatchWorker import batch_worker
from app.LiveWorker import live_worker


async def main():
    print("Starting Transcription Service...")

    load_model()          # HF pipeline (batch)
    load_model_live()     # faster-whisper (live)

    await rc.init_redis()
    print("  Redis connected")
    print("Transcription Service ready. Waiting for tasks...")

    batch_task = asyncio.create_task(batch_worker())
    live_task = asyncio.create_task(live_worker())

    try:
        await asyncio.gather(batch_task, live_task)
    finally:
        print("Shutting down Transcription Service...")
        await rc.close_redis()
        print("Transcription Service stopped.")


if __name__ == "__main__":
    asyncio.run(main())