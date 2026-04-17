"""
ASR Transcription Service
==========================
Loads the fine-tuned Whisper model once, then processes:
1. Batch chunks from queue:transcribe (file upload pipeline)
2. Live audio chunks from live:audio:{session_id} (real-time pipeline)

Live audio format: raw Int16 PCM @ 16kHz mono (little-endian), sent directly
from the browser's Web Audio API. No webm, no ffmpeg.
"""
import asyncio
import tempfile
import os
import time
import numpy as np
import soundfile as sf
from app.Config import config
from app import Redis_client as rc
from app.Inference import load_model, transcribe
from app.Worker import process_task

_session_buffers = {}
MIN_BYTES = 32000       # 16000 Hz * 2 bytes/sample * 1 second = 32000 bytes/sec; ~1s of audio
FLUSH_TIMEOUT = 2.5
SESSION_TIMEOUT = 30.0
SAMPLE_RATE = 16000


async def _transcribe_pcm_and_send(session_id: str, chunks: list):
    """chunks is a list of bytes where each element is raw Int16 PCM @ 16kHz mono."""
    if not chunks:
        return
    try:
        # Concatenate and convert Int16 PCM → Float32
        pcm_bytes = b"".join(chunks)
        if len(pcm_bytes) < 2:
            return
        # Ensure even byte count
        if len(pcm_bytes) % 2:
            pcm_bytes = pcm_bytes[:-1]
        int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio = int16.astype(np.float32) / 32768.0

        duration = len(audio) / SAMPLE_RATE
        if duration < 0.3:  # too short to be meaningful
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "chunk.wav")
            sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")

            result = transcribe(wav_path, dialect="auto")
            text = result.get("text", "").strip()

            print(f"  [LIVE] {session_id[:16]}: '{text[:80]}'")

            if text:
                await rc.push_live_result(session_id, {"type": "final", "text": text})

    except Exception as e:
        print(f"  [LIVE] Transcribe error: {e}")


async def live_worker():
    print("  Live transcription worker started.")
    while True:
        try:
            sessions = await rc.get_live_sessions()
            got_chunk = False

            active_this_cycle = set(sessions)
            active_this_cycle.update(_session_buffers.keys())

            for session_id in active_this_cycle:
                chunk = await rc.pop_live_audio(session_id)
                if chunk:
                    got_chunk = True
                    now = time.time()

                    if session_id not in _session_buffers:
                        _session_buffers[session_id] = {
                            "chunks": [chunk],
                            "total_bytes": len(chunk),
                            "first_chunk_time": now,
                            "last_chunk_time": now,
                        }
                        print(f"  [LIVE] New session {session_id[:16]} ({len(chunk)} bytes)")
                        continue

                    buf = _session_buffers[session_id]
                    buf["chunks"].append(chunk)
                    buf["total_bytes"] += len(chunk)
                    buf["last_chunk_time"] = now

                    if buf["total_bytes"] >= MIN_BYTES:
                        chunks_to_process = buf["chunks"][:]
                        buf["chunks"] = []
                        buf["total_bytes"] = 0
                        await _transcribe_pcm_and_send(session_id, chunks_to_process)

            # Flush stale buffers (inactivity)
            now = time.time()
            for session_id, buf in list(_session_buffers.items()):
                if buf["chunks"] and now - buf["last_chunk_time"] > FLUSH_TIMEOUT:
                    chunks_to_process = buf["chunks"][:]
                    buf["chunks"] = []
                    buf["total_bytes"] = 0
                    await _transcribe_pcm_and_send(session_id, chunks_to_process)

            # Clean up truly ended sessions
            now = time.time()
            for session_id in list(_session_buffers.keys()):
                buf = _session_buffers[session_id]
                if now - buf["last_chunk_time"] > SESSION_TIMEOUT:
                    del _session_buffers[session_id]
                    print(f"  [LIVE] Session {session_id[:16]} ended (timeout)")

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
