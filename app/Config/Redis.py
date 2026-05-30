# transcription: app/Config/Redis.py
"""
Redis client factories. One text client (decode_responses=True) for JSON
results, one binary client for raw audio bytes.
"""
from typing import Optional

from redis import asyncio as aioredis

from app.Config.Config import config


_text_client: Optional[aioredis.Redis] = None
_binary_client: Optional[aioredis.Redis] = None


async def get_text_client() -> aioredis.Redis:
    global _text_client
    if _text_client is None:
        _text_client = aioredis.from_url(config.REDIS_URL, decode_responses=True)
    return _text_client


async def get_binary_client() -> aioredis.Redis:
    global _binary_client
    if _binary_client is None:
        _binary_client = aioredis.from_url(config.REDIS_URL, decode_responses=False)
    return _binary_client


async def close_redis() -> None:
    global _text_client, _binary_client
    if _text_client:
        await _text_client.close()
        _text_client = None
    if _binary_client:
        await _binary_client.close()
        _binary_client = None