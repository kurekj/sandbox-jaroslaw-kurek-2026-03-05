import asyncio
import gc
import time

from loguru import logger

from src.v2.api.services.get_scores import _load_data, _load_data_cached
from src.v2.api.services.load_leads_df import load_leads_data_db, load_leads_data_db_cached
from src.v2.autoencoder.get_poi_data import get_all_pois
from src.v2.autoencoder.preprocess_data import _round_coords, load_current_properties_data


async def prefill_cache(overwrite_visible_properties: bool = True, overwrite_pois: bool = False) -> None:
    """Prefills the cache with user and property data for the recommendation system.

    This asynchronous function loads, processes, and caches multiple datasets required for
    the property recommendation system. It executes a series of data loading and caching
    operations in sequential stages, including leads data, apartment data, POI (Points of
    Interest) data, and establishes user-property relationships in the cache.

    **Note:** The function logs the progress and timing for each stage of execution. Stage 8
    (score calculation) is currently commented out as it's computationally heavy.

    Args:
        overwrite_visible_properties (bool, optional): Whether to overwrite existing visible
            properties data in the cache. Defaults to True.
        overwrite_pois (bool, optional): Whether to overwrite existing POI data.
            Defaults to False.

    Returns:
        None: This function doesn't return any values but updates the system cache.
    """
    start_time_total = time.time()
    logger.info("Prefilling cache with user_id <-> property_id pairs")

    # Load all users with leads from last 120 days
    start_time = time.time()
    logger.info("Stage 1: Loading leads data")
    leads_df = await load_leads_data_db()
    logger.info(f"Stage 1: Loaded {len(leads_df)} leads with valid property_id. Time: {time.time() - start_time:.2f}s")

    # Load and process apartments data
    start_time = time.time()
    logger.info("Stage 2: Loading and processing apartments data")
    raw_visible_apartments_df = await load_current_properties_data(only_for_sale=True)
    raw_visible_apartments_df = _round_coords(raw_visible_apartments_df)
    logger.info(
        f"Stage 2: Processed {len(raw_visible_apartments_df)} apartments. Time: {time.time() - start_time:.2f}s"
    )
    gc.collect()  # Clean up memory

    # Process location data
    start_time = time.time()
    logger.info("Stage 3: Getting POI data")
    unique_lat_lon = raw_visible_apartments_df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    unique_lat_lon["id"] = unique_lat_lon.index
    unique_lat_lon.sort_values(by=["lat", "lon"], inplace=True)
    # Get POI data
    await get_all_pois(unique_lat_lon, batch_size=50, overwrite=overwrite_pois)
    logger.info(f"Stage 3: Retrieved POI data for locations. Time: {time.time() - start_time:.2f}s")
    gc.collect()  # Clean up memory

    # Load data into cache
    start_time = time.time()
    logger.info("Stage 4: Loading visible properties data")
    all_visible_properties_df = await _load_data()
    logger.info(
        f"Stage 4: Loaded {len(all_visible_properties_df)} visible properties. Time: {time.time() - start_time:.2f}s"
    )

    start_time = time.time()
    logger.info("Stage 5: Loading leads (cached)")
    # NOTE: we want to always update the leads
    await load_leads_data_db_cached(leads_df["algolytics_uuid"].unique().tolist(), overwrite=True)
    logger.info(f"Stage 5: Loaded leads. Time: {time.time() - start_time:.2f}s")
    gc.collect()  # Clean up memory

    start_time = time.time()
    logger.info("Stage 6: Loading leads property data")
    # NOTE: we want to always update the leads
    await _load_data_cached(leads_df["property_id"].unique().tolist(), overwrite=True)
    logger.info(f"Stage 6: Loaded property data for leads. Time: {time.time() - start_time:.2f}s")
    gc.collect()  # Clean up memory

    start_time = time.time()
    logger.info("Stage 7: Loading all visible properties data (cached)")
    await _load_data_cached(
        all_visible_properties_df["property_id"].unique().tolist(),
        overwrite=overwrite_visible_properties,
    )
    logger.info(f"Stage 7: Loaded all visible properties. Time: {time.time() - start_time:.2f}s")
    gc.collect()  # Clean up memory

    # NOTE: for now the step 8 is too heavy to be run
    # # Generate scores
    # start_time = time.time()
    # logger.info("Stage 8: Calculating and caching scores")
    # user_ids = leads_df["algolytics_uuid"].unique().tolist()
    # property_ids = all_visible_properties_df["property_id"].unique().tolist()
    # # remove None values
    # user_ids = [user_id for user_id in user_ids if user_id is not None]
    # property_ids = [property_id for property_id in property_ids if property_id is not None]

    # logger.info(f"Stage 8: Preparing to calculate scores for {len(user_ids)} users/{len(property_ids)} properties")

    # # Process users in batches of 100
    # batch_size = 100
    # total_user_batches = (len(user_ids) + batch_size - 1) // batch_size  # Ceiling division

    # for batch_idx in tqdm(range(total_user_batches), desc="Processing batches of users", unit="batch"):
    #     batch_start = batch_idx * batch_size
    #     batch_end = min((batch_idx + 1) * batch_size, len(user_ids))
    #     batch_user_ids = user_ids[batch_start:batch_end]

    #     # Create cross product for current batch of users
    #     batch_scores_df = pd.DataFrame(
    #         itertools.product(batch_user_ids, property_ids), columns=["user_id", "property_id"]
    #     )

    #     # Calculate and cache scores for current batch
    #     await get_scores_df_cached(batch_scores_df)

    # logger.info(f"Stage 8: All score calculations completed. Total time: {time.time() - start_time:.2f}s")

    logger.info(f"Total prefill cache time: {time.time() - start_time_total:.2f}s")


if __name__ == "__main__":
    asyncio.run(prefill_cache())
