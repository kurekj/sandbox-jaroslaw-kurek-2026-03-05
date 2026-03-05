from functools import lru_cache

import pandas as pd
from aiocache.serializers import PickleSerializer  # type: ignore

from src.v2.config import get_config
from src.v2.db_utils import execute_query
from src.v2.utils.cache_utils import get_cache, get_or_create_cached_df
from src.v2.utils.sentinel_cache import SentinelCompatibleCache

# LEADS_QUERY = """
# select
# 	a.property_id::float::int as property_id,
# 	a.offer_id::float::int as offer_id,
#     to_timestamp(a.ext_timestamp::bigint / 1000) as ext_time,
# 	a.ext_timestamp::bigint,
# 	c.algolytics_uuid::varchar(255) as algolytics_uuid,
#     provide_city_name(oo.region_id) as city_name,
# 	st_x(st_centroid(st_transform(oo.geo_point, 4326))) as lon,
# 	st_y(st_centroid(st_transform(oo.geo_point, 4326))) as lat
# from
# 	web_events.application a
# left join web_events.crossuid c on
# 	a.session_id = c.session_id
# left join rds2.offers_offer oo on
# 	a.offer_id::float::int = oo.id
# left join rds2.regions_region rr on
# 	oo.region_id = rr.id
# where
# 	a.property_id is not null
# 	and c.algolytics_uuid = any(%s)
# """

ALL_LEADS_QUERY = """
select
	aa.create_date,
	aa.property_id,
	uu2."uuid"::varchar(255) as algolytics_uuid
from
	rds2.applications_application aa
left join rds2.users_userclient uu on
	aa.client_id = uu.client_id
left join km.users_user uu2 on
	uu.user_id = uu2.id
left join rds2.offers_offer oo on
	aa.offer_id = oo.id
left join rds2.regions_region rr on
	oo.region_id = rr.id
where
    aa.property_id is not null
    and uu2."uuid" is not null
    and aa.create_date >= current_date - interval '120 days'
    and rr.country = 1
    and aa.source NOT IN (203)
"""

LEADS_QUERY = f"""
{ALL_LEADS_QUERY}
	and uu2."uuid" = any(%s)
"""


LEADS_RAGE_QUERY = """
select
	aa.create_date,
	aa.property_id,
	uu2."uuid"::varchar(255) as algolytics_uuid
from
	rds2.applications_application aa
left join rds2.users_userclient uu on
	aa.client_id = uu.client_id
left join km.users_user uu2 on
	uu.user_id = uu2.id
where
	aa.create_date > %(start_date)s
	and aa.create_date < %(end_date)s;
"""

# Cache namespaces and formats
LEADS_CACHE_KEY_FORMAT = "leads:{key}"  # Cache key format for user leads
LEAD_CACHE_NAMESPACE = "load_leads_data"


@lru_cache
def get_leads_cache() -> SentinelCompatibleCache:
    """
    Get a cache instance for leads that supports both Redis and Redis Sentinel configurations.
    """
    return get_cache(namespace=LEAD_CACHE_NAMESPACE, serializer=PickleSerializer())


async def load_leads_data_db(user_ids: list[str] | None = None) -> pd.DataFrame:
    if user_ids is None:
        leads_data = await execute_query(ALL_LEADS_QUERY)
    else:
        leads_data = await execute_query(LEADS_QUERY, (user_ids,))
    leads_df = pd.DataFrame(leads_data)
    return leads_df


async def load_leads_data_db_cached(user_ids: list[str], overwrite: bool = False) -> pd.DataFrame:
    """
    Cached version of load_leads_data_db that checks Redis cache first before querying
    the database for user leads data. Uses the generic caching utilities.
    """
    return await get_or_create_cached_df(
        keys=user_ids,
        cache=get_leads_cache(),
        cache_key_format=LEADS_CACHE_KEY_FORMAT,
        load_func=load_leads_data_db,
        id_column="algolytics_uuid",
        ttl=get_config().cache.user_ttl,
        overwrite=overwrite,
    )


async def load_leads_data_db_in_range(start_date: str, end_date: str) -> pd.DataFrame:
    leads_data = await execute_query(
        LEADS_RAGE_QUERY,
        {
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    leads_df = pd.DataFrame(leads_data)
    return leads_df
