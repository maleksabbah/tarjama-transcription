"""
ASR Transcription Service
==========================
Loads the fine-tuned Whisper model once, then processes:
1. Batch chunks from queue:transcribe (file upload pipeline)
2. Live audio chunks from live:audio:{session_id} (real-time pipeline)

Run:
  python -m app.main
"""
import asyncio
import json
import tempfile
import os
import subprocess
from app.Config import config
from app import Redis_client as rc
from app.Inference import load_model, transcribe
from app.Worker import process_task


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any audio format to wav using ffmpeg."""
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-f", "wav",
            output_path
        ], capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        print(f"  [LIVE] ffmpeg error: {e}")
        return False


async def process_live_audio(session_id: str, audio_bytes: bytes):
    """Transcribe a live audio chunk and push result back to Redis."""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save raw webm bytes
            webm_path = os.path.join(tmp_dir, "live_chunk.webm")
            wav_path = os.path.join(tmp_dir, "live_chunk.wav")

            with open(webm_path, "wb") as f:
                f.write(audio_bytes)

            # Convert webm -> wav
            if not convert_to_wav(webm_path, wav_path):
                print(f"  [LIVE] ffmpeg conversion failed for session {session_id[:16]}")
                return

            result = transcribe(wav_path, dialect="auto")
            text = result.get("text", "").strip()

            if text:
                print(f"  [LIVE] Session {session_id[:16]}: '{text[:60]}'")
                await rc.push_live_result(session_id, {"type": "final", "text": text})

    except Exception as e:
        print(f"  [LIVE] Error for session {session_id[:16]}: {e}")
        await rc.push_live_result(session_id, {"type": "error", "message": str(e)})


async def live_worker():
    """Poll Redis for live audio chunks from any active session."""
    print("  Live transcription worker started.")
    while True:
        try:
            sessions = await rc.get_live_sessions()
            processed = False
            for session_id in sessions:
                chunk = await rc.pop_live_audio(session_id)
                if chunk:
                    await process_live_audio(session_id, chunk)
                    processed = True
            if not processed:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"  [LIVE] Worker error: {e}")
            await asyncio.sleep(1)


async def batch_worker():
    """Process batch transcription tasks from queue:transcribe."""
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


async def main():
    """Main worker loop."""
    print("Starting Transcription Service...")

    load_model()

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
