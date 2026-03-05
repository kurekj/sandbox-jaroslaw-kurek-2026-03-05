"""
Utilities for loading Tinder-like user interactions from the database.

This module introduces a new data loader for the "Tinder" mode of
``km.homematch_prelead`` (where ``mode = 3``).  The loader fetches
interactions for a list of user UUIDs and maps the underlying
``status`` values into continuous weights that capture the strength
and polarity of the user's feedback.  These weights follow the
Tinder-like re‑ranking proposal described in the design documents and
allow us to treat swipe gestures, clicks, shortlists and leads as
graded signals rather than a binary lead/no‑lead indicator.

The returned dataframe contains one row per user‑property interaction
with the following columns:

* ``create_date``: Timestamp of the interaction (timezone aware)
* ``property_id``: Identifier of the property (integer)
* ``algolytics_uuid``: UUID of the user
* ``status`` and ``status_name``: Original prelead status values
* ``event_code``: Intermediate code representing the type of feedback
* ``event_weight``: Continuous weight derived from ``event_code``

The mapping from ``status`` to ``event_code`` and the mapping from
``event_code`` to ``event_weight`` are defined below and can be
fine‑tuned in A/B tests.  See docs/Research-tinder-approach2.docx for
background and rationale.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import pandas as pd
from aiocache.serializers import PickleSerializer  # type: ignore

from src.v2.config import get_config
from src.v2.db_utils import execute_query
from src.v2.utils.cache_utils import get_or_create_cached_df, get_cache
from src.v2.utils.sentinel_cache import SentinelCompatibleCache

__all__ = [
    "load_tinder_interactions_db",
    "load_tinder_interactions_db_cached",
]

# Mapping of prelead status (km__homematch_prelead__status.id) to
# abstract event codes.  Negative codes indicate negative feedback,
# positive codes indicate increasing levels of positive feedback.  A
# value of 0 corresponds to neutral (ignored) interactions.  These
# mappings are inspired by the Tinder-like re‑ranking proposal.
STATUS_TO_EVENT_CODE: dict[int, int] = {
    1: -1,  # Oczekujące (shown, no action) → weak negative
    2: -2,  # Odrzucone (rejected) → strong negative (swipe left)
    3: 4,   # Utworzono zgłoszenie (lead created) → strongest positive
    4: 4,   # Zduplikowane zgłoszenie → treat as a strong positive
    5: 0,   # Problem z utworzeniem zgłoszenia → neutral
}

# Mapping of event codes to continuous weights.  These weights can be
# tuned; values below are taken directly from the research doc.  Note
# that event codes +1, +2 and +3 are included for completeness even
# though they are not currently produced by the status mapping.
EVENT_WEIGHTS: dict[int, float] = {
    -2: -0.5,
    -1: -0.02,
    0: 0.0,
    1: 0.2,
    2: 0.4,
    3: 0.7,
    4: 1.0,
}

# SQL query to fetch interactions from km.homematch_prelead for the
# Tinder mode.  The ``ANY(%s)`` placeholder allows passing a list of
# UUID strings directly via the psycopg driver.  The date filter of
# 120 days mirrors the behaviour of the existing leads query to keep
# the dataset fresh.
TINDER_INTERACTIONS_QUERY: str = """
SELECT
    hm.create_date,
    hm.property_id,
    u.uuid AS algolytics_uuid,
    hm.status AS status,
    ds.name AS status_name
FROM km.homematch_prelead hm
LEFT JOIN dictionary.km__homematch_prelead__status ds ON hm.status = ds.id
JOIN km.homematch_homematchconfiguration hc ON hm.configuration_id = hc.id
JOIN km.users_user u ON hc.user_id = u.id
WHERE hm.mode = 3
  AND u.uuid = ANY(%s)
  AND hm.create_date >= current_date - interval '120 days'
"""

# Cache namespace and key pattern for tinder interactions.  Each user
# UUID is used as a cache key to avoid expensive repeated queries.
CACHE_NAMESPACE: str = "load_tinder_interactions"
CACHE_KEY_FORMAT: str = "tinder:{key}"


@lru_cache
def get_tinder_cache() -> SentinelCompatibleCache:
    """Return a Sentinel-compatible cache instance for tinder interactions."""
    return get_cache(namespace=CACHE_NAMESPACE, serializer=PickleSerializer())


async def load_tinder_interactions_db(user_ids: List[str] | None = None) -> pd.DataFrame:
    """
    Load tinder interactions for a list of user UUIDs directly from the database.

    Args:
        user_ids: A list of user UUIDs.  If ``None`` or empty, an empty
            DataFrame is returned.

    Returns:
        A DataFrame containing a row per user‑property interaction with
        ``create_date``, ``property_id``, ``algolytics_uuid``, ``status``,
        ``status_name``, ``event_code`` and ``event_weight`` columns.  The
        latter two are derived columns used to build weighted user profiles.
    """
    if not user_ids:
        return pd.DataFrame()

    # Execute the query with the provided user IDs.  psycopg will convert
    # the Python list into the appropriate Postgres array format for ANY().
    rows = await execute_query(TINDER_INTERACTIONS_QUERY, (user_ids,))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Map status to event_code and then to event_weight.  Missing mappings
    # default to 0 (neutral) and a weight of 0.0.
    df["event_code"] = df["status"].map(STATUS_TO_EVENT_CODE).fillna(0).astype(int)
    df["event_weight"] = df["event_code"].map(EVENT_WEIGHTS).fillna(0.0).astype(float)

    return df


async def load_tinder_interactions_db_cached(user_ids: List[str], overwrite: bool = False) -> pd.DataFrame:
    """
    Cached wrapper around :func:`load_tinder_interactions_db`.

    This function uses the generic caching utilities to store interactions
    keyed by the user UUID.  The TTL for this cache is driven by
    ``get_config().cache.user_ttl`` to align with the existing user
    cache semantics.

    Args:
        user_ids: A list of user UUIDs to fetch interactions for.
        overwrite: If ``True``, forces a refresh from the database and
            overwrites the cached value.

    Returns:
        A DataFrame with the same structure as returned by
        :func:`load_tinder_interactions_db`.
    """
    return await get_or_create_cached_df(
        keys=user_ids,
        cache=get_tinder_cache(),
        cache_key_format=CACHE_KEY_FORMAT,
        load_func=load_tinder_interactions_db,
        id_column="algolytics_uuid",
        ttl=get_config().cache.user_ttl,
        overwrite=overwrite,
    )
