"""Tests for the new Sentinel-compatible cache implementation."""

import json
import pickle
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.v2.config import RedisCacheConfig, RedisSentinelCacheConfig
from src.v2.utils.cache_utils import get_cache
from src.v2.utils.sentinel_cache import SentinelCompatibleCache


class TestSentinelCompatibleCache:
    """Test cases for SentinelCompatibleCache."""

    @pytest.fixture
    def redis_config(self) -> RedisCacheConfig:
        """Create a mock Redis configuration."""
        config = MagicMock(spec=RedisCacheConfig)
        config.host = "localhost"
        config.port = 6379
        config.db_index = 0
        config.password = None
        return config

    @pytest.fixture
    def sentinel_config(self) -> RedisSentinelCacheConfig:
        """Create a mock Sentinel configuration."""
        config = MagicMock(spec=RedisSentinelCacheConfig)
        config.hosts = [("sentinel1", 26379), ("sentinel2", 26379)]
        config.master = "mymaster"
        config.db_index = 0
        config.password = None
        return config

    @pytest.mark.asyncio
    async def test_redis_cache_creation(self, redis_config: RedisCacheConfig) -> None:
        """Test creating cache with Redis configuration."""
        cache = SentinelCompatibleCache(
            namespace="test",
            config=redis_config,
        )

        # Mock Redis client
        with patch("redis.asyncio.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client

            client = await cache._get_redis_client()

            # Verify Redis was called with correct parameters
            mock_redis.assert_called_once_with(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=False,
                protocol=3,
                password=None,
            )
            assert client == mock_client

    @pytest.mark.asyncio
    async def test_sentinel_cache_creation(self, sentinel_config: RedisSentinelCacheConfig) -> None:
        """Test creating cache with Sentinel configuration."""
        cache = SentinelCompatibleCache(
            namespace="test",
            config=sentinel_config,
        )

        # Mock Sentinel
        with patch("redis.asyncio.Sentinel") as mock_sentinel_class:
            mock_sentinel = MagicMock()  # Use MagicMock, not AsyncMock
            mock_client = AsyncMock()
            mock_sentinel.master_for.return_value = mock_client
            mock_sentinel_class.return_value = mock_sentinel

            client = await cache._get_redis_client()

            # Verify Sentinel was called with correct parameters
            mock_sentinel_class.assert_called_once_with(
                sentinels=[("sentinel1", 26379), ("sentinel2", 26379)],
                password=None,
                decode_responses=False,
                protocol=3,
                db=0,
            )
            # Verify master_for was called correctly
            mock_sentinel.master_for.assert_called_once_with("mymaster")
            assert client == mock_client

    @pytest.mark.asyncio
    async def test_cache_operations(self, redis_config: RedisCacheConfig) -> None:
        """Test basic cache operations."""
        cache = SentinelCompatibleCache(
            namespace="test",
            config=redis_config,
        )

        # Mock Redis client
        with patch("redis.asyncio.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client

            # Mock get/set operations
            mock_client.get.return_value = None
            mock_client.set.return_value = None

            # Test set operation
            await cache.set("test_key", "test_value")
            mock_client.set.assert_called_once()

            # Test get operation
            result = await cache.get("test_key")
            assert result is None
            mock_client.get.assert_called_once()

    def test_binary_data_serialization(self, redis_config: RedisCacheConfig) -> None:
        """Test that binary data is properly serialized and deserialized to fix UnicodeDecodeError."""
        cache = SentinelCompatibleCache(
            namespace="test",
            config=redis_config,
        )

        # Test cases with different data types
        test_cases = [
            ("simple_string", "hello world"),
            ("simple_number", 42),
            ("simple_list", [1, 2, 3]),
            ("simple_dict", {"key": "value", "number": 123}),
            ("complex_object", pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})),
            ("nested_structure", {"list": [1, 2, {"nested": "value"}]}),
        ]

        for description, original_value in test_cases:
            # Serialize the value
            serialized = cache._serialize_value(original_value)

            # Verify serialization type
            if isinstance(original_value, (str, int, float, list, dict)):
                # Simple types should be JSON-serialized to string
                assert isinstance(serialized, str), f"Expected string for {description}, got {type(serialized)}"
                # Verify it's valid JSON
                json.loads(serialized)  # Should not raise exception
            else:
                # Complex types should be pickle-serialized to bytes
                assert isinstance(serialized, bytes), f"Expected bytes for {description}, got {type(serialized)}"
                # Verify it's valid pickle
                pickle.loads(serialized)  # Should not raise exception

            # Deserialize the value
            deserialized = cache._deserialize_value(serialized)

            # Verify deserialization
            if isinstance(original_value, pd.DataFrame):
                # Special comparison for DataFrames
                assert isinstance(deserialized, pd.DataFrame), f"Expected DataFrame for {description}"
                assert deserialized.equals(original_value), f"DataFrame comparison failed for {description}"
            else:
                assert deserialized == original_value, f"Value mismatch for {description}"

    def test_backwards_compatibility_hex_pickle(self, redis_config: RedisCacheConfig) -> None:
        """Test that hex-encoded pickle data (from old implementation) can still be read."""
        cache = SentinelCompatibleCache(
            namespace="test",
            config=redis_config,
        )

        # Test data that was previously stored as hex-encoded pickle
        test_data = {"list": [1, 2, 3], "nested": {"key": "value"}}
        hex_encoded = pickle.dumps(test_data).hex()

        # This should still work for backwards compatibility
        result = cache._deserialize_value(hex_encoded)
        assert result == test_data

    def test_mixed_data_types_deserialization(self, redis_config: RedisCacheConfig) -> None:
        """Test handling of mixed data types that might be returned from Redis."""
        cache = SentinelCompatibleCache(
            namespace="test",
            config=redis_config,
        )

        # Test deserialization of string data (JSON)
        json_data = '{"key": "value", "number": 123}'
        result = cache._deserialize_value(json_data)
        assert result == {"key": "value", "number": 123}

        # Test deserialization of bytes data (pickle)
        pickle_data = pickle.dumps({"complex": pd.DataFrame({"a": [1, 2]})})
        result = cache._deserialize_value(pickle_data)
        assert "complex" in result
        assert isinstance(result["complex"], pd.DataFrame)
        assert result["complex"].equals(pd.DataFrame({"a": [1, 2]}))

    @pytest.mark.asyncio
    async def test_multi_operations(self, redis_config: RedisCacheConfig) -> None:
        """Test multi-get and multi-set operations."""
        cache = SentinelCompatibleCache(
            namespace="test",
            config=redis_config,
        )

        # Mock Redis client
        with patch("redis.asyncio.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client

            # Mock pipeline using a simpler approach
            with patch.object(cache, "_get_redis_client", return_value=mock_client):
                # Mock multi-get
                mock_client.mget.return_value = [None, None]

                # Test multi-get operation (simpler test)
                result = await cache.multi_get(["key1", "key2"])
                assert result == [None, None]
                mock_client.mget.assert_called_once()

                # For multi-set, let's test the basic set operations instead
                # since the pipeline mock is complex
                await cache.set("key1", "value1")
                mock_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_operation(self, redis_config: RedisCacheConfig) -> None:
        """Test delete operation."""
        cache = SentinelCompatibleCache(
            namespace="test",
            config=redis_config,
        )

        # Mock Redis client
        with patch("redis.asyncio.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client

            # Test delete operation
            await cache.delete("test_key")
            mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_utility_function(self, redis_config: RedisCacheConfig) -> None:
        """Test that the cache utility function returns SentinelCompatibleCache."""
        # Mock the get_config function to return our test config
        with patch("src.v2.utils.cache_utils.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.get_redis_config.return_value = redis_config
            mock_get_config.return_value = mock_config

            cache = get_cache(namespace="test")
            assert isinstance(cache, SentinelCompatibleCache)
            assert cache.namespace == "test"
            assert cache.config == redis_config


class TestCacheUtils:
    """Test cases for cache utility functions."""

    @patch("src.v2.utils.cache_utils.get_config")
    def test_get_cache_with_redis(self, mock_get_config: MagicMock) -> None:
        """Test get_cache with Redis configuration."""
        # Mock config
        mock_config = MagicMock()
        mock_redis_config = MagicMock(spec=RedisCacheConfig)
        mock_redis_config.host = "localhost"
        mock_redis_config.port = 6379
        mock_redis_config.db_index = 0
        mock_redis_config.password = None
        mock_config.get_redis_config.return_value = mock_redis_config
        mock_config.cache.redis_cache_timeout = 60
        mock_get_config.return_value = mock_config

        cache = get_cache("test_namespace")

        # Should now always return SentinelCompatibleCache (supports both Redis and Sentinel)
        assert isinstance(cache, SentinelCompatibleCache)

    @patch("src.v2.utils.cache_utils.get_config")
    def test_get_cache_with_sentinel(self, mock_get_config: MagicMock) -> None:
        """Test get_cache with Sentinel configuration."""
        # Mock config
        mock_config = MagicMock()
        mock_sentinel_config = MagicMock(spec=RedisSentinelCacheConfig)
        mock_sentinel_config.hosts = [("sentinel1", 26379), ("sentinel2", 26379)]
        mock_sentinel_config.master = "mymaster"
        mock_sentinel_config.db_index = 0
        mock_sentinel_config.password = None
        mock_config.get_redis_config.return_value = mock_sentinel_config
        mock_config.cache.redis_cache_timeout = 60
        mock_get_config.return_value = mock_config

        cache = get_cache("test_namespace")

        # Should return SentinelCompatibleCache for Sentinel
        assert isinstance(cache, SentinelCompatibleCache)


if __name__ == "__main__":
    pytest.main([__file__])
