import asyncio
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from aiocache.serializers import PickleSerializer  # type: ignore
from loguru import logger

from src.v2.api.services.calculate_scores import calculate_all_users_scores
from src.v2.api.services.embed_properties import embed_properties
from src.v2.api.services.load_leads_df import load_leads_data_db_cached
from src.v2.autoencoder.preprocess_data import load_current_properties_data, preprocess_properties_data
from src.v2.config import get_config
from src.v2.utils.cache_utils import batch_get_or_set_cache, get_cache, get_or_create_cached_df
from src.v2.utils.sentinel_cache import SentinelCompatibleCache

# Cache namespaces
SCORES_CACHE_NAMESPACE = "get_scores_df"
PROPERTIES_CACHE_NAMESPACE = "property_data"

# Cache key formats
SCORE_CACHE_KEY_FORMAT = "score:{key}"  # {key} will be {user_id}:{property_id}
PROPERTY_CACHE_KEY_FORMAT = "property:{key}"  # {key} will be property_id


@lru_cache
def get_scores_cache() -> SentinelCompatibleCache:
    """
    Get a cache instance for scores that supports both Redis and Redis Sentinel configurations.
    """
    return get_cache(namespace=SCORES_CACHE_NAMESPACE)


@lru_cache
def get_properties_cache() -> SentinelCompatibleCache:
    """
    Get a cache instance for properties that supports both Redis and Redis Sentinel configurations.
    """
    return get_cache(namespace=PROPERTIES_CACHE_NAMESPACE, serializer=PickleSerializer())


async def _load_data(ids: Optional[list[int]] = None) -> pd.DataFrame:
    df = await load_current_properties_data(ids)
    df = await preprocess_properties_data(df)
    return df


async def _load_data_cached(ids: Optional[list[int]] = None, overwrite: bool = False) -> pd.DataFrame:
    """
    Cached version of _load_data that checks Redis cache first before loading
    and preprocessing property data. Uses the generic caching utilities.
    """

    if ids is None:
        # If no IDs provided, load all data without caching
        return await _load_data()

    if len(ids) == 0:
        logger.warning("No property IDs provided. Returning empty DataFrame.")
        return pd.DataFrame()

    # Use the generic caching function
    properties_cache = get_properties_cache()
    df = await get_or_create_cached_df(
        keys=ids,
        cache=properties_cache,
        cache_key_format=PROPERTY_CACHE_KEY_FORMAT,
        load_func=_load_data,
        id_column="property_id",
        ttl=get_config().cache.property_ttl,
        overwrite=overwrite,
    )

    return df


async def get_scores_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating scores for user_id <-> property_id pairs")
    # calculate scores for a dataframe of user_id <-> property_id pairs

    # First load leads for the users
    leads_df = await load_leads_data_db_cached(user_ids=df["user_id"].unique().tolist())
    logger.info(f"Loaded {len(leads_df)} leads for {len(df['user_id'].unique())} users")

    # load the data for leads properties and properties defined in the df
    if leads_df.empty:
        logger.warning("Leads dataframe is empty! Returning original df with NaN scores.")
        df["score"] = np.nan
        return df

    leads_properties_df = await _load_data_cached(ids=leads_df["property_id"].unique().tolist())
    scoring_properties_df = await _load_data_cached(ids=df["property_id"].unique().tolist())
    logger.info(
        f"Loaded {len(leads_properties_df)} leads properties and {len(scoring_properties_df)} scoring properties"
    )

    # if any of the dfs are empty, return the original df with NaN scores
    if leads_properties_df.empty or scoring_properties_df.empty:
        logger.warning("Leads properties or scoring properties are empty! Returning original df with NaN scores.")
        df["score"] = np.nan
        return df

    # embed the properties
    leads_properties_df = embed_properties(leads_properties_df)
    scoring_properties_df = embed_properties(scoring_properties_df)
    logger.info("Embedded leads properties and scoring properties")

    # calculate the scores for the properties and users
    all_scores, user_ids = calculate_all_users_scores(
        leads_df=leads_df,
        leads_properties_df=leads_properties_df,
        properties_embeddings=np.array(scoring_properties_df["embeddings"].tolist()),
    )
    logger.info("Calculated scores")

    property_ids = scoring_properties_df["property_id"].tolist()

    # Convert the scores matrix to a DataFrame with proper indices
    scores_df = pd.DataFrame(all_scores, index=user_ids, columns=property_ids)

    # Reset the index and melt the dataframe to get the user_id, property_id, score format
    scores_df = (
        scores_df.reset_index()
        .melt(id_vars="index", var_name="property_id", value_name="score")
        .rename(columns={"index": "user_id"})
    )

    # merge the scores with the original dataframe
    df = df.merge(scores_df, on=["user_id", "property_id"], how="left")

    return df


async def get_scores_df_cached(df: pd.DataFrame, overwrite_cache: bool = False) -> pd.DataFrame:
    """
    The idea behind this function is to check if given user_id and property_id pair already
    exists in the cache retrieve it and remove the pair from the df. The resulting df will be
    processed as normal. Calculated scores will be added to the cache for the given user_id and
    property_id pair. This way we can avoid recalculating scores for the same user_id and
    property_id pair multiple times.

    After the receiving the scores we will merge them with cached ones.
    """
    logger.info(f"Processing {len(df)} user_id <-> property_id pairs with caching")

    # Create a copy of the original dataframe
    original_df = df.copy()

    # Create cache keys for each user-property pair
    cache_keys = {}
    for _, row in df.iterrows():
        user_id = row["user_id"]
        property_id = row["property_id"]
        pair_key = f"{user_id}:{property_id}"
        cache_keys[pair_key] = SCORE_CACHE_KEY_FORMAT.format(key=pair_key)

    # Use batch get/set cache function to retrieve cached scores
    async def load_missing_scores(keys: list[str]) -> dict[str, Optional[float]]:
        # Extract user_id and property_id from the composite keys
        pairs_to_calculate = []
        key_to_pair = {}

        for key in keys:
            user_id, property_id = key.split(":")
            pairs_to_calculate.append({"user_id": user_id, "property_id": int(property_id)})
            key_to_pair[key] = {"user_id": user_id, "property_id": int(property_id)}

        # Calculate missing scores
        calc_df = pd.DataFrame(pairs_to_calculate)
        calculated_df = await get_scores_df(calc_df)

        # Convert back to dictionary of scores by pair key
        result = {}
        for _, row in calculated_df.iterrows():
            pair_key = f"{row['user_id']}:{row['property_id']}"
            result[pair_key] = float(row["score"]) if not np.isnan(row["score"]) else None

        return result

    # Get cached scores and calculate missing ones
    scores_cache = get_scores_cache()
    score_results = await batch_get_or_set_cache(
        items=cache_keys,
        cache=scores_cache,
        load_func=load_missing_scores,
        ttl=get_config().cache.score_ttl,
        overwrite=overwrite_cache,
    )

    # Convert results back to DataFrame
    result_data = []
    for pair_key, score in score_results.items():
        user_id, property_id = pair_key.split(":")
        if score is not None:  # Skip None values (could be NaN scores)
            result_data.append({"user_id": user_id, "property_id": int(property_id), "score": score})

    result_df = pd.DataFrame(result_data, columns=["user_id", "property_id", "score"])

    # Ensure we have all the original pairs in the result
    # This is important if some pairs couldn't be scored and aren't in the result yet
    final_df = original_df.merge(result_df, on=["user_id", "property_id"], how="left")

    return final_df


if __name__ == "__main__":
    import time

    # Example usage
    df = pd.DataFrame(
        {
            "user_id": [
                "0f51390e-63aa-4791-845e-97ef6dd92bfe",
                "0f51390e-63aa-4791-845e-97ef6dd92bfe",
                "c0dcface-0489-45cc-8a45-bc2ef2d243de",
                "c0dcface-0489-45cc-8a45-bc2ef2d243de",
                "c0dcface-0489-45cc-8a45-bc2ef2d243de",
            ],
            "property_id": [
                # 1181688,
                671261,
                1114792,
                988320,
                1221875,
                17407,
            ],
        }
    )

    start_time = time.time()
    # Test both caching implementations
    # result = asyncio.run(get_scores_df_cached(df))
    result = asyncio.run(get_scores_df_cached(df))  # Try the fully cached version
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    print(result)
