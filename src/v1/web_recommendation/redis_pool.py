from typing import Any, Dict

from kombu.utils.url import _parse_url
from redis.sentinel import Sentinel

from src.v1.web_recommendation.consts.config import (
    CELERY_REDIS_SENTINEL_MASTER_NAME,
    CELERY_REDIS_SENTINEL_PASSWORD,
    CELERY_RESULT_BACKEND,
)


def parse_sentinel_connection_string(conn_string: str) -> Dict[str, Any]:
    """
    Parse connection string to sentinel into a list of tuples (host, port).

    Connection string looks like this:
    sentinel://0.0.0.0:26347/3;sentinel://0.0.0.0:26348/3
    """
    connection_params: Dict[str, Any] = {
        "hosts": [],
    }
    for part in conn_string.split(";"):
        # use urlparse to parse sentinel://
        scheme, host, port, _, password, path, query = _parse_url(part)
        connection_params["hosts"].append((host, port))

        options: Dict[str, Any] = {
            "db": int(path) if path else 0,
            "password": password,
        }
        connection_params.update((key, value) for key, value in options.items() if value is not None)

    return connection_params


def get_sentinel_connection(conn_string, sentinel_kwargs=None):
    """
    Get a connection to a sentinel instance.
    """
    connection_params = parse_sentinel_connection_string(conn_string)
    hosts = connection_params.pop("hosts")
    sentinel = Sentinel(
        hosts,
        socket_timeout=0.1,
        sentinel_kwargs=sentinel_kwargs,
        **connection_params,
    )
    return sentinel


def get_redis_client(sentinel: Sentinel, master=True):
    connection_factory = sentinel.master_for if master else sentinel.slave_for

    return connection_factory(
        CELERY_REDIS_SENTINEL_MASTER_NAME,
    )


sentinel = get_sentinel_connection(
    CELERY_RESULT_BACKEND,
    sentinel_kwargs={
        "password": CELERY_REDIS_SENTINEL_PASSWORD,
    },
)

redis_client = get_redis_client(sentinel, master=True)
