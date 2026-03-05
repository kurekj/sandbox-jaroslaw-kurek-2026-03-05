# Redis Sentinel Cache Implementation

This implementation provides Redis Sentinel support for the web recommendation system, allowing for high availability Redis deployments.

## Overview

The `aiocache.RedisCache` is not compatible with Redis Sentinel out of the box. This implementation provides:

1. **SentinelCompatibleCache**: A drop-in replacement for aiocache.RedisCache that supports Redis Sentinel
2. **Backward compatibility**: Existing code using `get_redis_cache()` will continue to work with standard Redis
3. **Migration path**: New `get_cache()` function that automatically chooses the right cache implementation

## Architecture

### Standard Redis vs Redis Sentinel

- **Standard Redis**: Direct connection to a single Redis server
- **Redis Sentinel**: High availability setup with automatic failover

### Cache Classes

1. **SentinelCompatibleCache**: Our custom implementation supporting both Redis and Sentinel
2. **RedisCache (aiocache)**: Standard aiocache implementation (Redis only)

## Configuration

### Standard Redis Configuration

```python
# In your environment or config
REDIS__HOST=localhost
REDIS__PORT=6379
REDIS__DB_INDEX=0
REDIS__PASSWORD=optional_password
```

### Redis Sentinel Configuration

```python
# In your environment or config
REDIS_SENTINEL__HOSTS=[("sentinel1", 26379), ("sentinel2", 26379), ("sentinel3", 26379)]
REDIS_SENTINEL__MASTER=mymaster
REDIS_SENTINEL__DB_INDEX=0
REDIS_SENTINEL__PASSWORD=optional_password
```

## Usage

### Recommended Approach (New)

```python
from src.v2.utils.cache_utils import get_cache
from aiocache.serializers import PickleSerializer

# This automatically uses Sentinel if configured, or Redis otherwise
cache = get_cache(namespace="my_namespace", serializer=PickleSerializer())

# Use cache normally
await cache.set("key", value, ttl=3600)
result = await cache.get("key")

# Multi operations
keys = ["key1", "key2", "key3"]
values = await cache.multi_get(keys)

pairs = [("key1", value1), ("key2", value2)]
await cache.multi_set(pairs, ttl=3600)

# Clean up (only needed for SentinelCompatibleCache)
if hasattr(cache, 'close'):
    await cache.close()
```

### Migration from Old Code

```python
# OLD (deprecated but still works for standard Redis)
from src.v2.utils.cache_utils import get_redis_cache
cache = get_redis_cache(namespace="my_namespace")

# NEW (supports both Redis and Sentinel)
from src.v2.utils.cache_utils import get_cache
cache = get_cache(namespace="my_namespace")
```

### Direct Usage

```python
from src.v2.utils.sentinel_cache import SentinelCompatibleCache
from src.v2.config import get_config

config = get_config()
redis_config = config.get_redis_config()

cache = SentinelCompatibleCache(
    namespace="my_namespace",
    serializer=PickleSerializer(),
    config=redis_config,
    timeout=60,
)

# Use as context manager for automatic cleanup
async with cache:
    await cache.set("key", value)
    result = await cache.get("key")
```

## API Compatibility

The `SentinelCompatibleCache` provides the same interface as `aiocache.RedisCache`:

### Core Methods

- `set(key, value, ttl=None)`: Set a key-value pair
- `get(key)`: Get a value by key  
- `multi_get(keys)`: Get multiple values
- `multi_set(pairs, ttl=None)`: Set multiple key-value pairs
- `delete(key)`: Delete a key
- `exists(key)`: Check if key exists
- `clear()`: Clear all keys (use with caution)

### Additional Methods

- `close()`: Close connections (important for Sentinel)
- Context manager support: `async with cache:`

## Serialization

Supports the same serializers as aiocache:

```python
from aiocache.serializers import PickleSerializer, JsonSerializer

# Pickle serializer (recommended for complex objects)
cache = get_cache("namespace", serializer=PickleSerializer())

# JSON serializer (for simple objects)  
cache = get_cache("namespace", serializer=JsonSerializer())

# No serializer (auto-detection)
cache = get_cache("namespace")
```

## Error Handling

```python
from src.v2.utils.cache_utils import get_cache
from src.v2.utils.sentinel_cache import SentinelCompatibleCache

try:
    cache = get_cache("namespace")
    
    # Use cache...
    await cache.set("key", "value")
    
except Exception as e:
    logger.error(f"Cache operation failed: {e}")
    
finally:
    # Clean up if using Sentinel
    if isinstance(cache, SentinelCompatibleCache):
        await cache.close()
```

## Key Features

### Automatic Configuration Detection

- Detects Redis vs Sentinel configuration automatically
- No code changes needed when switching between environments

### Connection Management

- Proper connection pooling and cleanup
- Sentinel failover support
- Automatic reconnection

### Performance Optimizations

- Bulk operations support (`multi_get`, `multi_set`)
- Key hashing for consistent naming
- Efficient serialization

### Security

- Password support for both Redis and Sentinel
- Key hashing to prevent injection

## Environment Examples

### Development (Standard Redis)

```bash
# .env
REDIS__HOST=localhost
REDIS__PORT=6379
REDIS__DB_INDEX=0
```

### Production (Redis Sentinel)

```bash
# .env
REDIS_SENTINEL__HOSTS=[("redis-sentinel-1", 26379), ("redis-sentinel-2", 26379), ("redis-sentinel-3", 26379)]
REDIS_SENTINEL__MASTER=mymaster
REDIS_SENTINEL__DB_INDEX=0
REDIS_SENTINEL__PASSWORD=production_password
```

## Migration Checklist

1. **Update imports**: Change `get_redis_cache` to `get_cache`
2. **Add cleanup**: Add `await cache.close()` for Sentinel caches
3. **Test both configurations**: Verify with both Redis and Sentinel setups
4. **Update CI/CD**: Ensure environment variables are set correctly
5. **Monitor**: Watch for connection and performance issues

## Troubleshooting

### Common Issues

1. **"Cannot connect to Sentinel"**
   - Check Sentinel host/port configuration
   - Verify Sentinel is running and accessible

2. **"Master not found"**
   - Check master name in Sentinel configuration
   - Verify Sentinel is monitoring the correct master

3. **"Connection timeout"**
   - Increase timeout value in cache configuration
   - Check network connectivity

4. **"Authentication failed"**
   - Verify password is set correctly for both Redis and Sentinel
   - Check Redis/Sentinel ACL configuration

### Debugging

```python
import logging
logging.getLogger("redis").setLevel(logging.DEBUG)

# Enable debug logging for cache operations
cache = get_cache("namespace")
# Cache operations will now log debug information
```

## Performance Considerations

- **Sentinel Overhead**: Slight latency increase due to service discovery
- **Connection Pooling**: Reuse cache instances when possible
- **Bulk Operations**: Use `multi_get`/`multi_set` for better performance
- **TTL Management**: Set appropriate TTLs to manage memory usage
