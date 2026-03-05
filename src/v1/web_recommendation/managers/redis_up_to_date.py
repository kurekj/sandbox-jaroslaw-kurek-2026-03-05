from datetime import datetime

from src.v1.web_recommendation.redis_pool import redis_client

TIMESTAMP_REDIS_KEY = "current_date"
DATASET_EXPIRATION_TIME = 6 * 60 * 60  # 6 hours


def get_redis_current_date(default: str = None) -> str:
    """
    Get the current date from redis.

    If the current date is not set in redis, the default value is set and returned.
    """
    if not (redis_current_date := redis_client.get(TIMESTAMP_REDIS_KEY)):
        # Try to initialize the value and read it again
        set_redis_current_date(default, overwrite=False)
        redis_current_date = redis_client.get(TIMESTAMP_REDIS_KEY)

    if isinstance(redis_current_date, bytes):
        redis_current_date = redis_current_date.decode("utf-8")
    return redis_current_date


def set_redis_current_date(string_to_set=None, overwrite=True):
    """
    Set the current date in redis.

    Parameter overwrite set to False will prevent overwriting the value already set in redis.
    """
    if string_to_set is None:
        string_to_set = datetime.utcnow().isoformat()
    redis_client.set(TIMESTAMP_REDIS_KEY, string_to_set, nx=not overwrite)
    redis_client.expire(TIMESTAMP_REDIS_KEY, DATASET_EXPIRATION_TIME)
