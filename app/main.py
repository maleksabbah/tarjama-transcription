"""
ASR Transcription Service
==========================
Loads the fine-tuned Whisper model once, then processes:
1. Batch chunks from queue:transcribe (file upload pipeline)
2. Live audio chunks from live:audio:{session_id} (real-time pipeline)
   - Accumulates chunks until 3 seconds of audio
   - Skips silent chunks using energy-based VAD
   - Converts webm to wav with ffmpeg before transcribing

Run:
  python -m app.main
"""
import asyncio
import json
import tempfile
import os
import subprocess
import time
import numpy as np
from app.Config import config
from app import Redis_client as rc
from app.Inference import load_model, transcribe
from app.Worker import process_task

# Accumulate chunks per session before transcribing
# Key: session_id, Value: {"chunks": [...], "last_flush": timestamp}
_session_buffers = {}
ACCUMULATE_SECONDS = 3.0   # wait until we have ~3 seconds of audio
FLUSH_TIMEOUT = 2.0        # flush if no new chunk for 2 seconds
SILENCE_THRESHOLD = 0.01   # RMS energy below this = silence


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any audio format to wav 16kHz mono using ffmpeg."""
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


def is_silent(wav_path: str) -> bool:
    """Check if audio is silent using RMS energy."""
    try:
        import soundfile as sf
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        return rms < SILENCE_THRESHOLD
    except Exception:
        return False


async def process_live_audio(session_id: str, audio_bytes: bytes):
    """Buffer audio chunks and transcribe when enough data is accumulated."""
    now = time.time()

    if session_id not in _session_buffers:
        _session_buffers[session_id] = {"chunks": [], "last_flush": now, "total_bytes": 0}

    buf = _session_buffers[session_id]
    buf["chunks"].append(audio_bytes)
    buf["total_bytes"] += len(audio_bytes)
    buf["last_update"] = now

    # Each 500ms chunk at ~32kbps opus = ~2000 bytes
    # 3 seconds = ~6 chunks = ~12000 bytes — use time-based flush
    should_flush = (
        buf["total_bytes"] >= 12000 or  # enough data
        (now - buf["last_flush"]) >= ACCUMULATE_SECONDS  # timeout
    )

    if should_flush:
        chunks = buf["chunks"][:]
        buf["chunks"] = []
        buf["total_bytes"] = 0
        buf["last_flush"] = now
        await _transcribe_chunks(session_id, chunks)


async def _transcribe_chunks(session_id: str, chunks: list):
    """Concatenate chunks, convert to wav, run VAD, transcribe."""
    if not chunks:
        return

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write all chunks concatenated
            webm_path = os.path.join(tmp_dir, "live_chunk.webm")
            wav_path = os.path.join(tmp_dir, "live_chunk.wav")

            with open(webm_path, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)

            # Convert to wav
            if not convert_to_wav(webm_path, wav_path):
                print(f"  [LIVE] ffmpeg failed for {session_id[:16]}")
                return

            # Skip silent audio
            if is_silent(wav_path):
                return

            # Transcribe
            result = transcribe(wav_path, dialect="auto")
            text = result.get("text", "").strip()

            # Filter known hallucinations
            hallucinations = [
                "اشتركوا في القناة",
                "اشترك في القناة",
                "subscribe",
                "thank you",
                "شكراً للمشاهدة",
                "موسيقى",
            ]
            if any(h.lower() in text.lower() for h in hallucinations):
                print(f"  [LIVE] Filtered hallucination: '{text[:40]}'")
                return

            if text:
                print(f"  [LIVE] {session_id[:16]}: '{text[:60]}'")
                await rc.push_live_result(session_id, {"type": "final", "text": text})

    except Exception as e:
        print(f"  [LIVE] Transcribe error for {session_id[:16]}: {e}")


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

            # Flush stale buffers (session stopped sending)
            now = time.time()
            for session_id, buf in list(_session_buffers.items()):
                if (buf.get("chunks") and
                        now - buf.get("last_update", now) > FLUSH_TIMEOUT):
                    chunks = buf["chunks"][:]
                    buf["chunks"] = []
                    buf["total_bytes"] = 0
                    buf["last_flush"] = now
                    await _transcribe_chunks(session_id, chunks)

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
