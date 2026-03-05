from typing import Any, Awaitable, Callable, Hashable, Optional, TypeVar

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.v2.config import get_config
from src.v2.utils.sentinel_cache import SentinelCompatibleCache

T = TypeVar("T")


def get_cache(namespace: str, serializer: Any | None = None) -> SentinelCompatibleCache:
    """
    Get a cache instance that supports both Redis and Redis Sentinel configurations.

    Args:
        namespace: The namespace for the cache
        serializer: Optional serializer for the cache
    Returns:
        A SentinelCompatibleCache instance compatible with the current Redis configuration
    """
    config = get_config()
    redis_config = config.get_redis_config()

    # Always return SentinelCompatibleCache as it handles both Redis and Sentinel
    return SentinelCompatibleCache(
        namespace=namespace,
        serializer=serializer,
        config=redis_config,
        timeout=config.cache.redis_cache_timeout,
    )


async def get_or_create_cached_df(
    keys: list[T],
    cache: SentinelCompatibleCache,
    cache_key_format: str,
    load_func: Callable[[list[T]], Awaitable[pd.DataFrame]],
    id_column: str,
    ttl: int = get_config().cache.default_ttl,
    process_loaded_item: Optional[Callable[[dict[Hashable, Any]], dict[Hashable, Any]]] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Generic function to get or create a cached DataFrame.

    Args:
        keys: List of keys to load (user IDs, property IDs, etc.)
        cache: Redis cache instance
        cache_key_format: Format string for cache keys (e.g., "user:{key}")
        load_func: Function to load missing data
        id_column: Column name that contains the key in the loaded DataFrame
        ttl: Time-to-live for cache entries in seconds
        process_loaded_item: Optional function to process each loaded item before caching
        overwrite: Whether to overwrite existing cache entries

    Returns:
        Combined DataFrame of cached and newly loaded data
    """
    if not keys:
        return pd.DataFrame()

    logger.info(f"Loading data for {len(keys)} items with caching")

    # Format all cache keys
    cache_keys = [cache_key_format.format(key=key) for key in keys]
    key_to_cache_key = dict(zip(keys, cache_keys))

    # If overwrite is True, skip cache lookup and load everything
    if overwrite:
        keys_to_load: list[T] = keys
        cached_data: dict[T, Any] = {}
    else:
        # Bulk get from cache using multi_get
        logger.debug(f"Retrieving {len(keys)} items from cache")
        cached_values = await cache.multi_get(cache_keys)

        # Map results back to original keys - optimized version without loop
        cached_indices = [i for i, val in enumerate(cached_values) if val is not None]
        missing_indices = [i for i, val in enumerate(cached_values) if val is None]

        cached_data = {keys[i]: cached_values[i] for i in cached_indices}
        keys_to_load = [keys[i] for i in missing_indices]

        logger.debug(f"Found {len(cached_data)} items in cache, need to load {len(keys_to_load)} items")

    # Create DataFrame from cached data
    if cached_data:
        all_records = []
        for key, item_data in cached_data.items():
            if item_data:  # Check if the cached data isn't empty
                if isinstance(item_data, list):
                    all_records.extend(item_data)
                else:
                    all_records.append(item_data)

        cached_df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    else:
        cached_df = pd.DataFrame()

    # Load missing data
    loaded_df = pd.DataFrame()
    if keys_to_load:
        loaded_df = await load_func(keys_to_load)

        # Cache the newly loaded data using bulk operations
        if not loaded_df.empty:
            # Group loaded data by the id_column for faster processing
            grouped_data = loaded_df.groupby(id_column)

            # Optimize: First filter keys that need to be cached, then process only those groups
            keys_to_cache = set(keys_to_load)

            # Pre-compute which groups need caching to avoid repeated lookups
            groups_to_cache = {key: group for key, group in grouped_data if key in keys_to_cache}

            logger.debug(f"Processing {len(groups_to_cache)} groups for caching")

            # Process groups and cache immediately as they are processed
            for typed_key, group in tqdm(groups_to_cache.items(), desc="Processing and caching data", unit="group"):
                cache_key = key_to_cache_key.get(typed_key)  # type: ignore
                if not cache_key:
                    continue

                # Process records in bulk if possible
                if process_loaded_item:
                    # Convert once to records for better performance
                    group_records = group.to_dict("records")
                    processed_records = [process_loaded_item(record) for record in group_records]
                else:
                    processed_records = group.to_dict("records")

                # Save to cache immediately after processing each group
                # This ensures partial caching even if the process is interrupted
                await cache.set(cache_key, processed_records, ttl=ttl)

                # No need to collect in cache_items dictionary anymore
                # as we're saving each item immediately

    # Combine cached and newly loaded data
    if cached_df.empty:
        return loaded_df
    elif loaded_df.empty:
        return cached_df
    else:
        return pd.concat([cached_df, loaded_df], ignore_index=True)


async def batch_get_or_set_cache(
    items: dict[str, str],  # Dict of item_id -> cache_key
    cache: SentinelCompatibleCache,
    load_func: Callable[[list[str]], Awaitable[dict[str, Any]]],
    ttl: int = get_config().cache.default_ttl,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Batch get or set cache entries.

    Args:
        items: Dictionary mapping item IDs to their cache keys
        cache: Redis cache instance
        load_func: Function to load missing data
        ttl: Time-to-live for cache entries in seconds
        overwrite: Whether to overwrite existing cache entries

    Returns:
        Dictionary mapping item IDs to their values
    """
    if not items:
        return {}

    # Prepare lists for bulk operations
    item_ids = list(items.keys())
    cache_keys = list(items.values())

    if not overwrite:
        # Bulk get all items from cache
        cached_values = await cache.multi_get(cache_keys)
        logger.debug(f"Retrieved {len(cached_values)} items from cache")
    else:
        cached_values = [None] * len(cache_keys)
        logger.debug("Overwrite mode: skipping cache lookup")

    # Process results - vectorized approach
    cached_results = {}
    items_to_load = []
    keys_to_load = []

    # Create mappings in one go
    for i, item_id in enumerate(item_ids):
        if cached_values[i] is not None:
            cached_results[item_id] = cached_values[i]
        else:
            items_to_load.append(item_id)
            keys_to_load.append(cache_keys[i])

    # If all items were in cache, return cached results
    if not items_to_load:
        return cached_results

    logger.debug(f"Found {len(cached_results)} items in cache, need to load {len(items_to_load)} items")

    # Load missing items
    loaded_values = await load_func(items_to_load)

    # Create bulk cache items directly
    cache_items = [
        (keys_to_load[idx], loaded_values[item_id])
        for idx, item_id in enumerate(items_to_load)
        if item_id in loaded_values
    ]

    # Bulk cache newly loaded items
    if cache_items:
        await cache.multi_set(pairs=cache_items, ttl=ttl)

    # Combine cached and loaded results
    return {**cached_results, **loaded_values}
