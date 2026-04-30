"""
Live transcription worker.
Pops PCM chunks from live:audio:{session_id}, accumulates buffers,
hands them to faster-whisper (which has Silero VAD built in),
pushes text back to live:result:{session_id}.
"""
import asyncio
import time
import numpy as np
from app import Redis_client as rc
from app.Inference import transcribe as transcribe_live

SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2        # Int16 mono

MIN_BYTES = BYTES_PER_SEC * 3          # ~3 seconds
FLUSH_TIMEOUT = 2.5
SESSION_TIMEOUT = 30.0
MIN_SPEECH_DURATION = 0.5

_session_buffers: dict[str, dict] = {}


def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]
    int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int16.astype(np.float32) / 32768.0


async def _transcribe_and_send(session_id: str, chunks: list[bytes]):
    if not chunks:
        return
    try:
        pcm_bytes = b"".join(chunks)
        if len(pcm_bytes) < 2:
            return

        audio = _pcm_to_float32(pcm_bytes)
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_SPEECH_DURATION:
            return

        text = await transcribe_live(audio)

        if not text:
            # VAD found no speech — skip silently
            print(f"  [LIVE] {session_id[:16]}: no speech detected ({duration:.1f}s)")
            return

        print(f"  [LIVE] {session_id[:16]}: '{text[:80]}'")
        await rc.push_live_result(session_id, {"type": "final", "text": text})

    except Exception as e:
        print(f"  [LIVE] Transcribe error: {e}")


async def live_worker():
    print("  Live worker started.")
    while True:
        try:
            sessions = await rc.get_live_sessions()
            got_chunk = False

            active = set(sessions)
            active.update(_session_buffers.keys())

            for session_id in active:
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
                        to_process = buf["chunks"][:]
                        buf["chunks"] = []
                        buf["total_bytes"] = 0
                        await _transcribe_and_send(session_id, to_process)

            # Flush stragglers that haven't hit MIN_BYTES but have gone quiet
            now = time.time()
            for session_id, buf in list(_session_buffers.items()):
                if buf["chunks"] and now - buf["last_chunk_time"] > FLUSH_TIMEOUT:
                    to_process = buf["chunks"][:]
                    buf["chunks"] = []
                    buf["total_bytes"] = 0
                    await _transcribe_and_send(session_id, to_process)

            # Clean up dead sessions
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