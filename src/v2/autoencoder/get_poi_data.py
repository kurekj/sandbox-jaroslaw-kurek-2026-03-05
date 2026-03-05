import asyncio
from functools import lru_cache
from typing import Any

import httpx
import pandas as pd
from aiocache.serializers import PickleSerializer  # type: ignore
from tqdm import tqdm

from src.v2.config import get_config
from src.v2.utils.cache_utils import get_cache
from src.v2.utils.get_poi_count import get_poi_count
from src.v2.utils.sentinel_cache import SentinelCompatibleCache

from loguru import logger

# Cache namespaces
POI_CACHE_NAMESPACE = "poi_data"

AGG_POI = {
    "childcare": "EDUCATION",
    "kindergarten": "EDUCATION",
    "school": "EDUCATION",
    "college": "EDUCATION",
    "university": "EDUCATION",
    "library": "ALL",
    "atm": "ALL",
    "bank": "ALL",
    "parcel_locker": "ALL",
    "playground": "ENTERTAINMENT",
    "ars_centre": "ENTERTAINMENT",
    "cinema": "ALL",
    "theatre": "ALL",
    "museum": "ALL",
    "all_sports": "SPORT",
    "fitness_centre": "SHOPS",
    "fitness_station": "SPORT",
    "ice_rink": "SPORT",
    "pitch": "SPORT",
    "sports_centre": "SPORT",
    "stadium": "SPORT",
    "swimming_pool": "SPORT",
    "track": "SPORT",
    "woods": "ALL",
    "park": "ALL",
    "water": "ALL",
    "grass": "ALL",
    "all_food": "FOOD",
    "pubs": "FOOD",
    "clubs": "ALL",
    "amusement": "ALL",
    "parking": "ALL",
    "fuel_carwash": "ALL",
    "roads": "ALL",
    "bus_stop": "TRANSPORT",
    "tram_stop": "TRANSPORT",
    "subway_stop": "TRANSPORT",
    "train_stop": "TRANSPORT",
    "all_shops": "SHOPS",
    "local_shops": "SHOPS",
    "food_shops": "SHOPS",
    "city": "ALL",
    "post": "ALL",
    "all_doctor": "HEALTH",
    "doctor": "HEALTH",
    "pharmacy": "ALL",
    "vet": "ALL",
}


@lru_cache
def get_poi_cache() -> SentinelCompatibleCache:
    return get_cache(namespace=POI_CACHE_NAMESPACE, serializer=PickleSerializer())


async def _call_numlabs_api(lat: float, lon: float, distance: int, url: str) -> pd.DataFrame:
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        response = await client.get(url, params={"point": f"{lat},{lon}", "dist": distance})
        response.raise_for_status()
        poi = response.json()
        df = pd.DataFrame(poi[1:])
        try:
            df["agg_type"] = df["type"].map(AGG_POI)
        except KeyError:
            raise KeyError(
                f"The response from NumLABS does not contain a 'type' column for lat: {lat}, lon: {lon}, "
                + f"distance: {distance}"
            )
        return df


async def _get_pois_data(lat: float, lon: float, overwrite_cache: bool = False) -> dict[str, dict[str, int]]:
    # Check if the data is already cached
    logger.info(f"START… calling _call_numlabs_api for coordinates ({lat}, {lon})")
    poi_cache = get_poi_cache()

    logger.info(f"AFTER get_poi_cache() ({lat}, {lon}) overwrite_cache:{overwrite_cache}")
    cache_key = f"pois:{lat}:{lon}"

    if not overwrite_cache:
        cached_data = await poi_cache.get(cache_key)
        if cached_data:
            return cached_data  # type: ignore

    logger.info(f"BEFORE get_config().numlabs ({lat}, {lon})")
    config = get_config().numlabs

    logger.info(f"Numlabs config: distance={config.transport_distance}, url={config.poi_url}")

    logger.info(f"Calling Numlabs API for lat={lat}, lon={lon}")

    pois = await _call_numlabs_api(lat, lon, config.transport_distance, config.poi_url)

    logger.info(f"Received {len(pois)} rows from Numlabs, preview:\n{pois.head()}")

    pois_transport = pois[pois["agg_type"] == "TRANSPORT"]
    pois_non_transport = pois[(pois["agg_type"] != "TRANSPORT") & (pois["dist"] <= config.poi_distance)]

    df = pd.concat([pois_non_transport, pois_transport])

    poi_data = get_poi_count(df.to_dict(orient="records"))

    # Save to cache
    await poi_cache.set(cache_key, poi_data, ttl=get_config().cache.poi_ttl)
    return poi_data  # type: ignore


async def _process_batch(batch: pd.DataFrame, overwrite_cache: bool = False) -> list[dict[str, Any]]:
    tasks = []
    batch_results = []
    for _, row in batch.iterrows():
        tasks.append(_get_pois_data(lat=row["lat"], lon=row["lon"], overwrite_cache=overwrite_cache))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for row, result in zip(batch.iterrows(), results):  # type: ignore
        try:
            if isinstance(result, Exception):
                raise result
            batch_results.append({"id": int(row[1]["id"]), "pois": result})
        except Exception as e:
            print(f"Error processing row {int(row[1]['id'])} - ({row[1]['lat']}, {row[1]['lon']}): {e}")

    return batch_results


async def get_all_pois(df: pd.DataFrame, batch_size: int = 10, overwrite: bool = False) -> pd.DataFrame:
    all_results = []

    for start in tqdm(range(0, df.shape[0], batch_size), desc="Loading POIs"):
        end = min(start + batch_size, df.shape[0])
        batch = df.iloc[start:end]
        batch_results = await _process_batch(batch, overwrite_cache=overwrite)
        all_results.extend(batch_results)

    # Convert the list of results to a DataFrame
    result_df = pd.DataFrame(all_results, columns=["id", "pois"])

    return result_df
