# Caching Mechanism Documentation

## Overview

The recommendation system implements a multi-layered caching mechanism designed to optimize performance and reduce computational overhead. The caching system handles three main types of data: property information, user lead data, and computed recommendation scores.

## Architecture Overview

The caching system consists of several layers:

1. **Redis/Redis Sentinel Backend**: Distributed cache storage
2. **Namespace Separation**: Logical separation of different data types
3. **TTL Management**: Automatic expiration of stale data
4. **Batch Operations**: Efficient bulk read/write operations
5. **Serialization**: Optimized data storage and retrieval

## Core Components

### 1. SentinelCompatibleCache

**Location**: `src/v2/utils/sentinel_cache.py`

This is the foundation of the caching system, providing a unified interface for both standard Redis and Redis Sentinel configurations.

#### Key Features

- **High Availability**: Supports Redis Sentinel for failover scenarios
- **Namespace Support**: Automatic key prefixing for logical separation
- **Multiple Serializers**: Supports pickle, JSON, and custom serialization
- **Connection Management**: Automatic connection pooling and reconnection

#### Configuration

```python
# Standard Redis Configuration
RedisCacheConfig(
    host="localhost",
    port=6379,
    db=0,
    password="optional_password"
)

# Redis Sentinel Configuration  
RedisSentinelCacheConfig(
    sentinels=[("sentinel1", 26379), ("sentinel2", 26379)],
    service_name="mymaster",
    password="optional_password"
)
```

### 2. Cache Utilities

**Location**: `src/v2/utils/cache_utils.py`

Provides high-level functions for common caching patterns used throughout the recommendation system.

#### Key Functions

##### `get_cache(namespace, serializer=None)`

Creates cache instances with proper configuration.

```python
# Example usage
properties_cache = get_cache("property_data", PickleSerializer())
scores_cache = get_cache("get_scores_df")
```

##### `get_or_create_cached_df(keys, cache, cache_key_format, load_func, ...)`

Generic function for DataFrame caching with bulk operations.

```python
# Loads data from cache or database as needed
df = await get_or_create_cached_df(
    keys=[1, 2, 3],  # Property IDs
    cache=properties_cache,
    cache_key_format="property:{key}",
    load_func=load_properties_from_db,
    id_column="property_id",
    ttl=604800  # 7 days
)
```

##### `batch_get_or_set_cache(items, cache, load_func, ...)`

Handles batch operations for key-value caching scenarios.

## Cache Namespaces and Data Types

The system uses different namespaces to logically separate different types of cached data:

### 1. Property Data Cache

- **Namespace**: `"property_data"`
- **Key Format**: `"property:{property_id}"`
- **TTL**: 7 days (604,800 seconds)
- **Serializer**: PickleSerializer (for complex DataFrames)
- **Data**: Preprocessed property information with features

### 2. User Leads Cache

- **Namespace**: `"load_leads_data"`
- **Key Format**: `"leads:{user_id}"`
- **TTL**: 24 hours (86,400 seconds)
- **Serializer**: PickleSerializer
- **Data**: User application/lead history

### 3. Recommendation Scores Cache

- **Namespace**: `"get_scores_df"`
- **Key Format**: `"score:{user_id}:{property_id}"`
- **TTL**: 24 hours (86,400 seconds)
- **Serializer**: Default (JSON-compatible)
- **Data**: Computed similarity scores between users and properties

## TTL (Time-To-Live) Strategy

Different data types have different update frequencies and importance levels:

```python
class CacheConfig:
    redis_cache_timeout: int = 3600        # 1 hour - Connection timeout
    default_ttl: int = 43200               # 12 hours - General purpose
    property_ttl: int = 604800             # 7 days - Property data (stable)
    user_ttl: int = 86400                  # 24 hours - User data (moderate)
    score_ttl: int = 86400                 # 24 hours - Computed scores
    poi_ttl: int = 2592000                 # 30 days - POI data (very stable)
```

### Rationale

- **Property Data (7 days)**: Property features change infrequently
- **User Data (24 hours)**: User behavior patterns evolve regularly
- **Scores (24 hours)**: Balance between performance and freshness
- **POI Data (30 days)**: Geographic points of interest are very stable

## Caching Patterns

### 1. Bulk Data Loading Pattern

Used for loading multiple properties or users efficiently:

```python
async def _load_data_cached(ids: list[int], overwrite: bool = False) -> pd.DataFrame:
    """Load and cache property data in bulk."""
    if not ids:
        return await _load_data(ids)
    
    properties_cache = get_properties_cache()
    return await get_or_create_cached_df(
        keys=ids,
        cache=properties_cache,
        cache_key_format=PROPERTY_CACHE_KEY_FORMAT,
        load_func=_load_data,
        id_column="property_id",
        ttl=get_config().cache.property_ttl,
        overwrite=overwrite,
    )
```

### 2. Score Caching Pattern

Used for caching computed recommendation scores:

```python
async def get_scores_df_cached(df: pd.DataFrame, overwrite_cache: bool = False) -> pd.DataFrame:
    """Cache computed scores for user-property pairs."""
    
    # Create cache keys for each user-property pair
    cache_keys = {}
    for _, row in df.iterrows():
        user_id = row["user_id"]
        property_id = row["property_id"]
        pair_key = f"{user_id}:{property_id}"
        cache_keys[pair_key] = SCORE_CACHE_KEY_FORMAT.format(key=pair_key)
    
    # Batch load missing scores
    async def load_missing_scores(keys: list[str]) -> dict[str, float]:
        # Convert keys back to DataFrame and compute scores
        pairs_to_calculate = []
        for key in keys:
            user_id, property_id = key.split(":")
            pairs_to_calculate.append({"user_id": user_id, "property_id": int(property_id)})
        
        calc_df = pd.DataFrame(pairs_to_calculate)
        calculated_df = await get_scores_df(calc_df)
        
        # Convert back to key-value format
        result = {}
        for _, row in calculated_df.iterrows():
            pair_key = f"{row['user_id']}:{row['property_id']}"
            result[pair_key] = float(row["score"]) if not np.isnan(row["score"]) else None
        
        return result
    
    # Use batch cache operation
    scores_cache = get_scores_cache()
    score_results = await batch_get_or_set_cache(
        items=cache_keys,
        cache=scores_cache,
        load_func=load_missing_scores,
        ttl=get_config().cache.score_ttl,
        overwrite=overwrite_cache,
    )
```

## Performance Optimizations

### 1. Bulk Operations

The caching system is designed around bulk operations to minimize Redis round-trips:

- **Multi-Get**: Retrieve multiple keys in a single Redis command
- **Batch Processing**: Process and cache data in groups
- **Connection Pooling**: Reuse Redis connections efficiently

### 2. Smart Cache Miss Handling

When cache misses occur, the system:

1. **Groups Missing Keys**: Collects all missing keys before database queries
2. **Bulk Database Loading**: Makes fewer, larger database queries
3. **Immediate Caching**: Caches results as they're computed
4. **Partial Success**: Handles partial failures gracefully

### 3. Memory-Efficient Processing

```python
# Process groups and cache immediately to avoid memory buildup
for typed_key, group in tqdm(groups_to_cache.items(), desc="Processing and caching data"):
    cache_key = key_to_cache_key.get(typed_key)
    
    # Process records in bulk
    if process_loaded_item:
        group_records = group.to_dict("records")
        processed_records = [process_loaded_item(record) for record in group_records]
    else:
        processed_records = group.to_dict("records")
    
    # Save to cache immediately after processing each group
    await cache.set(cache_key, processed_records, ttl=ttl)
```

## Cache Invalidation Strategies

### 1. Time-Based Expiration (TTL)

Primary mechanism for cache invalidation:

- Automatic expiration based on data type
- No manual invalidation needed for most cases
- Balances freshness with performance

### 2. Manual Cache Overwrite

For scenarios requiring fresh data:

```python
# Force refresh of property data
df = await _load_data_cached(property_ids, overwrite=True)

# Force refresh of scores
scores = await get_scores_df_cached(user_property_pairs, overwrite_cache=True)
```

### 3. Selective Invalidation

The system supports targeted cache clearing for specific keys or patterns.

## Error Handling and Resilience

### 1. Serialization Errors

Handles serialization/deserialization failures:

```python
try:
    return self.serializer.loads(value)
except (pickle.PickleError, json.JSONDecodeError) as e:
    logger.error(f"Failed to deserialize cached value: {e}")
    return None  # Trigger cache miss and reload
```

### 2. Partial Failures

The system can handle partial cache failures:

- Some keys retrieved from cache, others from database
- Graceful degradation when Redis is unavailable
- Automatic retry logic for transient failures

## Monitoring and Observability

### 1. Logging

The caching system provides detailed logging:

```python
logger.info(f"Loading data for {len(keys)} items with caching")
logger.debug(f"Found {len(cached_data)} items in cache, need to load {len(keys_to_load)} items")
logger.debug(f"Processing {len(groups_to_cache)} groups for caching")
```

## Cache Warming Strategies

For optimal performance, the system includes cache warming mechanisms:

### 1. Prefill Cache Utility

**Location**: `src/v2/utils/prefill_cache.py`

Proactively loads frequently accessed data:

```python
async def prefill_cache(overwrite_visible_properties: bool = True):
    """Prefill cache with commonly accessed data."""
    
    # Load user leads (last 120 days)
    leads_df = await load_leads_data_db()
    
    # Load property data for leads
    await _load_data_cached(
        leads_df["property_id"].unique().tolist(), 
        overwrite=True
    )
    
    # Load user data
    await load_leads_data_db_cached(
        leads_df["algolytics_uuid"].unique().tolist(), 
        overwrite=True
    )
```

### 2. Scheduled Cache Warming

Recommended to run cache warming:

- **Daily**: For user data and recent properties
- **Weekly**: For full property dataset refresh
- **After Model Updates**: When embeddings change

### 3. Cache Warming API Endpoints

The system provides REST API endpoints for cache management and warming operations.

#### Start Cache Prefill Task

**Endpoint**: `POST /prefill_cache/start`

Initiates a background cache warming task using Celery:

```python
# Request body
{
    "overwrite_visible_properties": true,  # Force refresh property data
    "overwrite_pois": false               # Keep existing POI data
}

# Response (HTTP 202 Accepted)
{
    "task_id": "celery-task-uuid-12345",
    "state": "PENDING"
}
```

**Usage Examples**:

```bash
# Start cache prefill with default settings
curl -X POST "https://api.example.com/prefill_cache/start" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'

# Force refresh of all data
curl -X POST "https://api.example.com/prefill_cache/start" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "overwrite_visible_properties": true,
    "overwrite_pois": true
  }'
```

#### Monitor Task State

**Endpoint**: `GET /prefill_cache/state?task_id={task_id}`

Check the current status of a cache prefill task:

```python
# Response examples
{
    "task_id": "celery-task-uuid-12345",
    "state": "PENDING",    # Task is queued
    "ready": false,        # Task not yet completed
    "successful": false    # Task not successful yet
}

{
    "task_id": "celery-task-uuid-12345",
    "state": "STARTED",    # Task is running
    "ready": false,        # Task not yet completed
    "successful": false    # Task not successful yet
}

{
    "task_id": "celery-task-uuid-12345",
    "state": "SUCCESS",    # Task completed successfully
    "ready": true,         # Task completed
    "successful": true     # Task was successful
}

{
    "task_id": "celery-task-uuid-12345",
    "state": "FAILURE",    # Task failed
    "ready": true,         # Task completed (with failure)
    "successful": false    # Task was not successful
}
```

**Usage**:

```bash
# Check task status
curl -X GET "https://api.example.com/prefill_cache/state?task_id=celery-task-uuid-12345" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### Operational Considerations

**Resource Usage**:

- Cache prefill is memory and CPU intensive
- Recommended to run during low-traffic periods
- Consider Redis memory limits when scheduling
