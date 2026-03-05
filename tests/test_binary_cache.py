"""
Test script to verify that the SentinelCompatibleCache correctly handles binary data and serialization/deserialization.
Covers single/multi set/get, serialization, and backwards compatibility.
"""

import json
import pickle
from typing import Any, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.v2.config import RedisCacheConfig
from src.v2.utils.sentinel_cache import SentinelCompatibleCache


@pytest.mark.asyncio
async def test_sentinel_compatible_cache_binary_and_serialization() -> None:
    """Test SentinelCompatibleCache for binary data, serialization, and backwards compatibility."""
    # Use MagicMock for Redis client to avoid real Redis dependency
    config = MagicMock(spec=RedisCacheConfig)
    cache = SentinelCompatibleCache(namespace="test_binary", config=config)

    # --- Serialization/Deserialization tests ---
    test_cases: List[Tuple[str, Any]] = [
        ("simple_string", "hello world"),
        ("simple_number", 42),
        ("simple_list", [1, 2, 3]),
        ("simple_dict", {"key": "value", "number": 123}),
        ("complex_object", pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})),
        ("nested_structure", {"list": [1, 2, {"nested": "value"}], "df": pd.DataFrame({"x": [1, 2]})}),
    ]

    for description, original_value in test_cases:
        # Serialize
        serialized = cache._serialize_value(original_value)
        # Check serialization type based on actual result
        if isinstance(serialized, str):
            # Should be valid JSON
            json.loads(serialized)  # Should not raise
        elif isinstance(serialized, bytes):
            # Should be valid pickle
            pickle.loads(serialized)  # Should not raise
        else:
            pytest.fail(f"Unexpected serialization type for {description}: {type(serialized)}")
        # Deserialize
        deserialized = cache._deserialize_value(serialized)
        if isinstance(original_value, pd.DataFrame):
            assert isinstance(deserialized, pd.DataFrame), f"Expected DataFrame for {description}"
            assert deserialized.equals(original_value), f"DataFrame comparison failed for {description}"
        elif isinstance(original_value, dict) and "df" in original_value:
            # Special case for nested dict with DataFrame
            assert isinstance(deserialized, dict)
            assert deserialized["df"].equals(original_value["df"])
            d1 = dict(original_value)
            d2 = dict(deserialized)
            d1.pop("df")
            d2.pop("df")
            assert d1 == d2
        else:
            assert deserialized == original_value, f"Value mismatch for {description}"

    # --- Mixed data types deserialization ---
    json_data = '{"key": "value", "number": 123}'
    result = cache._deserialize_value(json_data)
    assert result == {"key": "value", "number": 123}
    pickle_data = pickle.dumps({"complex": pd.DataFrame({"a": [1, 2]})})
    result = cache._deserialize_value(pickle_data)
    assert "complex" in result
    assert isinstance(result["complex"], pd.DataFrame)
    assert result["complex"].equals(pd.DataFrame({"a": [1, 2]}))

    # --- Backwards compatibility: hex-encoded pickle ---
    test_data = {"list": [1, 2, 3], "nested": {"key": "value"}}
    hex_encoded = pickle.dumps(test_data).hex()
    result = cache._deserialize_value(hex_encoded)
    assert result == test_data

    # --- Async cache operations (mocked) ---
    # Mock the Redis client with AsyncMock
    client = AsyncMock()
    cache._get_redis_client = AsyncMock(return_value=client)  # type: ignore[method-assign]

    # Set/get
    client.get.return_value = None
    client.set.return_value = None
    await cache.set("test_key", "test_value")
    client.set.assert_called_once()
    result = await cache.get("test_key")
    client.get.assert_called_once()

    # Multi-get
    client.mget.return_value = [None, None]
    result = await cache.multi_get(["key1", "key2"])
    client.mget.assert_called_once()

    # Delete
    client.delete.return_value = None
    await cache.delete("test_key")
    client.delete.assert_called_once()
