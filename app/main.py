"""
ASR Transcription Service
==========================
Loads the fine-tuned Whisper model once, then processes:
1. Batch chunks from queue:transcribe (file upload pipeline)
2. Live audio chunks from live:audio:{session_id} (real-time pipeline)
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

_session_buffers = {}
MIN_BYTES = 10000       # ~3 seconds of opus audio
FLUSH_TIMEOUT = 2.5     # flush if no new chunk for 2.5 seconds
SILENCE_THRESHOLD = 0.01


def convert_to_wav(input_path: str, output_path: str) -> bool:
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
    try:
        import soundfile as sf
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        return rms < SILENCE_THRESHOLD
    except Exception:
        return False


HALLUCINATIONS = [
    "اشتركوا في القناة",
    "اشترك في القناة",
    "subscribe",
    "thank you for watching",
    "شكراً للمشاهدة",
    "موسيقى",
]


async def _transcribe_and_send(session_id: str, header_chunk: bytes, chunks: list):
    if not chunks:
        return
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            webm_path = os.path.join(tmp_dir, "chunk.webm")
            wav_path = os.path.join(tmp_dir, "chunk.wav")

            with open(webm_path, "wb") as f:
                # Always prepend the header chunk so ffmpeg can parse the webm
                f.write(header_chunk)
                for c in chunks:
                    f.write(c)

            if not convert_to_wav(webm_path, wav_path):
                print(f"  [LIVE] ffmpeg failed for {session_id[:16]}")
                return

            if is_silent(wav_path):
                print(f"  [LIVE] Silent chunk skipped for {session_id[:16]}")
                return

            result = transcribe(wav_path, dialect="auto")
            text = result.get("text", "").strip()

            if any(h.lower() in text.lower() for h in HALLUCINATIONS):
                print(f"  [LIVE] Hallucination filtered: '{text[:40]}'")
                return

            if text:
                print(f"  [LIVE] {session_id[:16]}: '{text[:60]}'")
                await rc.push_live_result(session_id, {"type": "final", "text": text})

    except Exception as e:
        print(f"  [LIVE] Transcribe error: {e}")


async def live_worker():
    print("  Live transcription worker started.")
    while True:
        try:
            sessions = await rc.get_live_sessions()
            got_chunk = False

            for session_id in sessions:
                chunk = await rc.pop_live_audio(session_id)
                if chunk:
                    got_chunk = True
                    now = time.time()

                    if session_id not in _session_buffers:
                        # First chunk — save as header, don't add to buffer yet
                        _session_buffers[session_id] = {
                            "header": chunk,
                            "chunks": [],
                            "total_bytes": 0,
                            "first_chunk_time": now,
                            "last_chunk_time": now
                        }
                        print(f"  [LIVE] New session {session_id[:16]}, saved header ({len(chunk)} bytes)")
                        continue

                    buf = _session_buffers[session_id]
                    buf["chunks"].append(chunk)
                    buf["total_bytes"] += len(chunk)
                    buf["last_chunk_time"] = now

                    # Flush when enough data accumulated
                    if buf["total_bytes"] >= MIN_BYTES:
                        chunks_to_process = buf["chunks"][:]
                        buf["chunks"] = []
                        buf["total_bytes"] = 0
                        buf["first_chunk_time"] = now
                        await _transcribe_and_send(session_id, buf["header"], chunks_to_process)

            # Flush stale buffers (user paused speaking)
            now = time.time()
            for session_id, buf in list(_session_buffers.items()):
                if (buf["chunks"] and
                        now - buf["last_chunk_time"] > FLUSH_TIMEOUT):
                    chunks_to_process = buf["chunks"][:]
                    buf["chunks"] = []
                    buf["total_bytes"] = 0
                    buf["first_chunk_time"] = now
                    await _transcribe_and_send(session_id, buf["header"], chunks_to_process)

            # Clean up sessions that no longer exist in Redis
            for session_id in list(_session_buffers.keys()):
                if session_id not in sessions:
                    del _session_buffers[session_id]
                    print(f"  [LIVE] Session {session_id[:16]} ended, cleaned up")

            if not got_chunk:
                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"  [LIVE] Worker error: {e}")
            await asyncio.sleep(1)


async def batch_worker():
    while True:
        try:
            message = await rc.pop_transcribe_task(timeout=5)
            if message:
                print(f"  [TRANSCRIBE] Received task for job {message.get('job_id')}")
                await process_task(message)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"  [TRANSCRIBE] Error: {e}")
            await asyncio.sleep(1)


async def main():
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
