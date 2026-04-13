"""
ASR Transcription Service
==========================
Loads the fine-tuned Whisper model once, then processes chunks from the queue.
Runs as a background worker — no HTTP server. GPU-bound.

Run:
  python -m app.main
"""
import asyncio
from app.Config import config
from app import Redis_client as rc
from app.Inference import load_model
from app.Worker import process_task


async def main():
    """Main worker loop."""
    print("Starting Transcription Service...")

    # Load model once on startup (takes a few seconds)
    load_model()

    await rc.init_redis()
    print("  Redis connected")
    print("Transcription Service ready. Waiting for tasks...")

    try:
        while True:
            try:
                message = await rc.pop_transcribe_task(timeout=5)
                if message:
                    print(f"  [TRANSCRIBE] Received task for job {message.get('job_id')} "
                          f"chunk {message.get('chunk_index')}")
                    await process_task(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  [TRANSCRIBE] Error: {e}")
                await asyncio.sleep(1)
    finally:
        print("Shutting down Transcription Service...")
        await rc.close_redis()
        print("Transcription Service stopped.")


if __name__ == "__main__":
    asyncio.run(main())