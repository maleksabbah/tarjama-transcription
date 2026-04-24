"""
Batch transcription worker.
Pops jobs from queue:transcribe, runs them through process_task
(which uses the HF pipeline in Inference.py).
"""
import asyncio
from app import Redis_client as rc
from app.Worker import process_task


async def batch_worker():
    print("  Batch worker started.")
    while True:
        try:
            message = await rc.pop_transcribe_task(timeout=5)
            if message:
                print(f"  [BATCH] Received task for job {message.get('job_id')}")
                await process_task(message)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"  [BATCH] Error: {e}")
            await asyncio.sleep(1)