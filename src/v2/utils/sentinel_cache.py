import hashlib
import json
import pickle
from types import TracebackType
from typing import Any, Optional, Type, Union

import redis.asyncio as redis
from loguru import logger

from src.v2.config import RedisCacheConfig, RedisSentinelCacheConfig


class SentinelCompatibleCache:
    """
    A Redis cache implementation that supports both standard Redis and Redis Sentinel connections.

    This class provides the same interface as aiocache.RedisCache but with Redis Sentinel support
    for high availability deployments.
    """

    def __init__(
        self,
        namespace: str = "",
        serializer: Optional[Any] = None,
        config: Optional[Union[RedisCacheConfig, RedisSentinelCacheConfig]] = None,
        timeout: int = 60,
    ):
        """
        Initialize the Sentinel-compatible cache.

        Args:
            namespace: Cache namespace to prefix all keys
            serializer: Serializer for cache values (pickle, json, or custom)
            config: Redis configuration (RedisCacheConfig or RedisSentinelCacheConfig)
            timeout: Connection timeout in seconds
        """
        self.namespace = namespace
        self.serializer = serializer
        self.config = config
        self.timeout = timeout
        self._redis_client: Optional[redis.Redis] = None
        self._sentinel: Optional[redis.Sentinel] = None

    def _get_namespaced_key(self, key: str) -> str:
        """Add namespace prefix to key."""
        if self.namespace:
            return f"{self.namespace}:{key}"
        return key

    def _hash_key(self, key: str) -> str:
        """Hash a key using SHA-256 for consistent key naming."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _serialize_value(self, value: Any) -> Union[str, bytes]:
        """Serialize value based on the configured serializer."""
        if self.serializer is None:
            # Default to JSON serialization for simple types, pickle for complex
            try:
                return json.dumps(value)
            except (TypeError, ValueError):
                # Return binary data directly for pickle
                return pickle.dumps(value)
        elif hasattr(self.serializer, "dumps"):
            # aiocache serializer interface
            return self.serializer.dumps(value)  # type: ignore
        elif callable(self.serializer):
            # Custom callable serializer
            return self.serializer(value)  # type: ignore
        else:
            # Assume it's a string representation
            return str(value)

    def _deserialize_value(self, value: Union[str, bytes]) -> Any:
        """Deserialize value based on the configured serializer."""
        if self.serializer is None:
            # Handle both string and bytes
            if isinstance(value, bytes):
                # Try to unpickle first
                try:
                    return pickle.loads(value)
                except (ValueError, pickle.UnpicklingError):
                    # Try to decode as UTF-8 and then parse as JSON
                    try:
                        return json.loads(value.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        return value
            else:
                # String value - try JSON first, fallback to pickle from hex
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    try:
                        return pickle.loads(bytes.fromhex(value))
                    except (ValueError, pickle.UnpicklingError):
                        return value
        elif hasattr(self.serializer, "loads"):
            # aiocache serializer interface
            return self.serializer.loads(value)
        elif callable(self.serializer):
            # For custom deserializer, assume it can handle the value
            return value
        else:
            return value

    async def _get_redis_client(self) -> redis.Redis:
        """Get or create Redis client based on configuration."""
        if self._redis_client is not None:
            return self._redis_client

        if not self.config:
            raise ValueError("Redis configuration is required")

        if isinstance(self.config, RedisSentinelCacheConfig):
            # Create Sentinel connection
            self._sentinel = redis.Sentinel(
                sentinels=self.config.hosts,
                password=self.config.password,
                decode_responses=False,  # Keep binary data as bytes
                protocol=3,
                db=self.config.db_index,
            )  # type: ignore
            master_client = self._sentinel.master_for(self.config.master)
            # Check if master_for returns a coroutine and await if needed
            if hasattr(master_client, "__await__"):
                self._redis_client = await master_client
            else:
                self._redis_client = master_client
        elif isinstance(self.config, RedisCacheConfig):
            # Create standard Redis connection
            self._redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db_index,
                decode_responses=False,  # Keep binary data as bytes
                protocol=3,
                password=self.config.password,
            )
        else:
            raise ValueError("Invalid configuration type")

        return self._redis_client

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache."""
        client = await self._get_redis_client()
        namespaced_key = self._get_namespaced_key(key)
        hashed_key = self._hash_key(namespaced_key)
        serialized_value = self._serialize_value(value)

        await client.set(name=hashed_key, value=serialized_value, ex=ttl)
        logger.debug(f"Set cache key: {namespaced_key} (hashed: {hashed_key[:8]}...)")

    async def get(self, key: str) -> Any:
        """Get a value from the cache."""
        client = await self._get_redis_client()
        namespaced_key = self._get_namespaced_key(key)
        hashed_key = self._hash_key(namespaced_key)

        value = await client.get(hashed_key)
        if value is None:
            logger.debug(f"Cache miss for key: {namespaced_key}")
            return None

        logger.debug(f"Cache hit for key: {namespaced_key}")
        return self._deserialize_value(value)

    async def multi_get(self, keys: list[str]) -> list[Any]:
        """Get multiple values from the cache."""
        if not keys:
            return []

        client = await self._get_redis_client()
        namespaced_keys = [self._get_namespaced_key(key) for key in keys]
        hashed_keys = [self._hash_key(key) for key in namespaced_keys]

        values = await client.mget(hashed_keys)
        logger.debug(f"Multi-get for {len(keys)} keys, found {sum(1 for v in values if v is not None)} values")

        result: list[Any] = []
        for value in values:
            if value is None:
                result.append(None)
            else:
                result.append(self._deserialize_value(value))

        return result

    async def multi_set(self, pairs: list[tuple[str, Any]], ttl: Optional[int] = None) -> None:
        """Set multiple key-value pairs in the cache."""
        if not pairs:
            return

        client = await self._get_redis_client()

        # Process pairs for bulk operation
        mapping = {}
        for key, value in pairs:
            namespaced_key = self._get_namespaced_key(key)
            hashed_key = self._hash_key(namespaced_key)
            serialized_value = self._serialize_value(value)
            mapping[hashed_key] = serialized_value

        # Use pipeline for better performance
        async with client.pipeline() as pipe:
            await pipe.mset(mapping)
            if ttl:
                for hashed_key in mapping.keys():
                    await pipe.expire(hashed_key, ttl)
            await pipe.execute()

        logger.debug(f"Multi-set {len(pairs)} key-value pairs")

    async def delete(self, key: str) -> None:
        """Delete a key from the cache."""
        client = await self._get_redis_client()
        namespaced_key = self._get_namespaced_key(key)
        hashed_key = self._hash_key(namespaced_key)

        await client.delete(hashed_key)
        logger.debug(f"Deleted cache key: {namespaced_key}")

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        client = await self._get_redis_client()
        namespaced_key = self._get_namespaced_key(key)
        hashed_key = self._hash_key(namespaced_key)

        result = await client.exists(hashed_key)
        return bool(result)

    async def clear(self, namespace: Optional[str] = None) -> None:
        """Clear all keys in the cache or a specific namespace."""
        client = await self._get_redis_client()

        # For simplicity, we'll use the FLUSHDB command
        # In production, you might want to be more selective
        logger.warning("Clearing entire Redis database")
        await client.flushdb()

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis_client:
            try:
                await self._redis_client.aclose()
                logger.debug("Closed Redis client connection")
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
            finally:
                self._redis_client = None

        if self._sentinel:
            try:
                for sentinel_conn in self._sentinel.sentinels:
                    await sentinel_conn.aclose()
                logger.debug("Closed Sentinel connections")
            except Exception as e:
                logger.warning(f"Error closing Sentinel connections: {e}")
            finally:
                self._sentinel = None

    async def __aenter__(self) -> "SentinelCompatibleCache":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        await self.close()
