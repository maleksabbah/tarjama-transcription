"""
Transcription Service Redis Client
Pop from queue:transcribe, push to queue:completed.
Supports both batch and live transcription queues.
"""

import json
import redis.asyncio as redis
from app.Config import config

client: redis.Redis = None
binary_client: redis.Redis = None


async def init_redis():
    global client, binary_client
    # Text client for JSON messages
    client = redis.from_url(config.REDIS_URL, decode_responses=True)
    # Binary client for raw audio bytes
    binary_client = redis.from_url(config.REDIS_URL, decode_responses=False)


async def close_redis():
    global client, binary_client
    if client:
        await client.close()
    if binary_client:
        await binary_client.close()


async def pop_transcribe_task(timeout: int = 5) -> dict | None:
    """Pop a task from the transcribe queue."""
    result = await client.brpop(config.QUEUE_TRANSCRIBE, timeout=timeout)
    if result:
        _, data = result
        return json.loads(data)
    return None


async def push_completed(message: dict):
    """Push completion message to queue:completed."""
    await client.lpush(config.QUEUE_COMPLETED, json.dumps(message))


async def pop_live_audio(session_id: str) -> bytes | None:
    """Pop a raw audio chunk from a live session queue."""
    result = await binary_client.lpop(f"live:audio:{session_id}")
    return result


async def push_live_result(session_id: str, result: dict):
    """Push transcription result back to live session."""
    await client.lpush(f"live:result:{session_id}", json.dumps(result))
    await client.expire(f"live:result:{session_id}", 120)


async def get_live_sessions() -> list:
    """Get all active live audio session keys."""
    keys = await client.keys("live:audio:*")
    return [k.replace("live:audio:", "") for k in keys]
