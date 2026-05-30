# transcription: app/Services/LiveWorkerService.py
"""
Live transcription worker.
Polls Redis live:audio:* keys, accumulates per-session buffers,
transcribes when buffer is full or has gone quiet, pushes results back.
"""
import asyncio
import time

import numpy as np

from app.Repositories import LiveSessionRepository
from app.Services.InferenceService import InferenceService


SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2          # int16 mono
MIN_BYTES = BYTES_PER_SEC * 3            # ~3 seconds
FLUSH_TIMEOUT = 2.5
SESSION_TIMEOUT = 30.0
MIN_SPEECH_DURATION = 0.5


def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]
    int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int16.astype(np.float32) / 32768.0


class LiveWorkerService:
    def __init__(
        self,
        sessions: LiveSessionRepository,
        inference: InferenceService,
    ):
        self.sessions = sessions
        self.inference = inference
        self._buffers: dict[str, dict] = {}

    async def run(self) -> None:
        print("  [LIVE] Worker started")
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  [LIVE] Worker error: {e}")
                await asyncio.sleep(1)

    async def _tick(self) -> None:
        active_sessions = await self.sessions.list_active_sessions()
        active = set(active_sessions)
        active.update(self._buffers.keys())

        got_chunk = False
        for session_id in active:
            chunk = await self.sessions.pop_audio_chunk(session_id)
            if chunk:
                got_chunk = True
                await self._absorb_chunk(session_id, chunk)

        await self._flush_quiet_buffers()
        self._evict_dead_sessions()

        if not got_chunk:
            await asyncio.sleep(0.05)

    async def _absorb_chunk(self, session_id: str, chunk: bytes) -> None:
        now = time.time()

        if session_id not in self._buffers:
            self._buffers[session_id] = {
                "chunks": [chunk],
                "total_bytes": len(chunk),
                "first_chunk_time": now,
                "last_chunk_time": now,
            }
            print(f"  [LIVE] New session {session_id[:16]} ({len(chunk)} bytes)")
            return

        buf = self._buffers[session_id]
        buf["chunks"].append(chunk)
        buf["total_bytes"] += len(chunk)
        buf["last_chunk_time"] = now

        if buf["total_bytes"] >= MIN_BYTES:
            to_process = buf["chunks"][:]
            buf["chunks"] = []
            buf["total_bytes"] = 0
            await self._transcribe_and_send(session_id, to_process)

    async def _flush_quiet_buffers(self) -> None:
        now = time.time()
        for session_id, buf in list(self._buffers.items()):
            if buf["chunks"] and now - buf["last_chunk_time"] > FLUSH_TIMEOUT:
                to_process = buf["chunks"][:]
                buf["chunks"] = []
                buf["total_bytes"] = 0
                await self._transcribe_and_send(session_id, to_process)

    def _evict_dead_sessions(self) -> None:
        now = time.time()
        for session_id in list(self._buffers.keys()):
            if now - self._buffers[session_id]["last_chunk_time"] > SESSION_TIMEOUT:
                del self._buffers[session_id]
                print(f"  [LIVE] Session {session_id[:16]} ended (timeout)")

    async def _transcribe_and_send(
        self, session_id: str, chunks: list[bytes],
    ) -> None:
        try:
            pcm_bytes = b"".join(chunks)
            if len(pcm_bytes) < 2:
                return

            audio = _pcm_to_float32(pcm_bytes)
            duration = len(audio) / SAMPLE_RATE
            if duration < MIN_SPEECH_DURATION:
                return

            text = self.inference.transcribe_buffer(audio)

            if not text:
                print(f"  [LIVE] {session_id[:16]}: no speech ({duration:.1f}s)")
                return

            print(f"  [LIVE] {session_id[:16]}: '{text[:80]}'")
            await self.sessions.push_result(
                session_id, {"type": "final", "text": text},
            )
        except Exception as e:
            print(f"  [LIVE] Transcribe error: {e}")