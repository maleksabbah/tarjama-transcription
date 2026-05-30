# transcription: app/Repositories/LiveSessionRepository.py
"""
Redis live session storage — read audio chunks, write transcript results.
Mirrors the orchestrator's repo so both sides agree on key shapes.
"""
from redis import asyncio as aioredis


class LiveSessionRepository:
    def __init__(
        self,
        text_client: aioredis.Redis,
        binary_client: aioredis.Redis,
    ):
        self.text = text_client
        self.binary = binary_client

    async def list_active_sessions(self) -> list[str]:
        """Return all session IDs that currently have an audio buffer."""
        keys = await self.text.keys("live:audio:*")
        return [k.replace("live:audio:", "") for k in keys]

    async def pop_audio_chunk(self, session_id: str) -> bytes | None:
        """Pop one raw audio chunk (bytes) from a session's buffer."""
        return await self.binary.lpop(f"live:audio:{session_id}")

    async def push_result(self, session_id: str, result: dict) -> None:
        """Push a transcription result back for the orchestrator to drain."""
        import json
        key = f"live:result:{session_id}"
        await self.text.lpush(key, json.dumps(result))
        await self.text.expire(key, 120)