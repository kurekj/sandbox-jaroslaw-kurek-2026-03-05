import logging
import os
from abc import ABC
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, cast

import toml
from fastapi import Depends
from pydantic import field_validator
from pydantic_core import Url
from pydantic_settings import BaseSettings, SettingsConfigDict

CURRENT_DIR = Path(__file__).resolve().parent


def get_project_version() -> str:
    """
    Retrieves the project version from the pyproject.toml file.

    This function constructs the path to the pyproject.toml file by navigating
    three directories up from the current file's location. It then loads the
    contents of the file and extracts the version information specified under
    the "tool.poetry.version" key.

    Returns:
        str: The version of the project as specified in the pyproject.toml file.

    Raises:
        FileNotFoundError: If the pyproject.toml file does not exist.
        KeyError: If the version key is not found in the pyproject.toml file.
    """
    pyproject_path = os.path.join(os.path.dirname(__file__), *[".."] * 2, "pyproject.toml")
    pyproject_data = toml.load(pyproject_path)
    return cast(str, pyproject_data["project"]["version"])


class AppConfig(BaseSettings):
    """Application-specific configuration."""

    api_name: str = "Scoring API"
    """The name of the API service."""
    version: Optional[str] = None
    """The version of the API service."""
    api_key: str = "<your-secret-api-key>"
    """The secret API key for authentication."""
    api_key_header: str = "X-API-KEY"
    """The header name for passing the API key."""
    host: str = "0.0.0.0"
    """The host of the application."""
    port: int = 8000
    """The port on which the application listens."""
    workers: Optional[int] = None
    """The number of worker processes. Defaults to none."""
    root_path: str = ""
    """The root path for the application, useful when running behind a reverse proxy."""

    @field_validator("version", mode="after")
    def load_version(cls, v: Optional[str]) -> str:
        """Load version from pyproject.toml if not provided."""
        if v is None:  # Check if the value is None
            return get_project_version()
        return v


class DBConfig(BaseSettings):
    """Configuration for the database with offer data."""

    host: str = "db6.prod.propertygroup"
    """The host of the database."""
    user: str = "<user>"
    """The user for the database."""
    password: str = "<password>"
    """The password for the database."""
    dbname: str = "algolytics"
    """The name of the database."""
    port: int = 5432
    """The port of the database."""


class BaseRedisConfig(BaseSettings, ABC):
    """Base configuration class for Redis settings."""

    db_index: int = 0
    """The index of the Redis database."""
    password: Optional[str] = None
    """The password for the Redis server (optional)."""

    def get_connection_string(self) -> str:
        """Get the connection string for the Redis server."""
        raise NotImplementedError("Subclasses must implement this method.")


class RedisSentinelConfig(BaseRedisConfig):
    """Configuration for Redis Sentinel."""

    hosts: list[tuple[str, int]]
    """A list of (host, port) tuples for Sentinel servers."""
    master: str
    """The name of the master service monitored by Sentinel."""

    def get_connection_string(self) -> str:
        """Get the connection string for the Redis server."""
        if self.password:
            connection_string = ";".join(
                f"sentinel://:{self.password}@{host}:{port}/{self.db_index}" for host, port in self.hosts
            )
        else:
            connection_string = ";".join(f"sentinel://{host}:{port}/{self.db_index}" for host, port in self.hosts)

        return connection_string


class RedisConfig(BaseRedisConfig):
    """Configuration for the Redis database."""

    host: str = "redis"
    """The host of the Redis server."""
    port: int = 6379
    """The port of the Redis server."""

    def get_connection_string(self) -> str:
        """Get the connection string for the Redis server."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db_index}"
        else:
            return f"redis://{self.host}:{self.port}/{self.db_index}"


class CacheConfig(BaseSettings):
    """Configuration for the cache."""

    redis_cache_timeout: int = 60 * 60  # 1 hour
    """Timeout for Redis cache in seconds."""
    default_ttl: int = 60 * 60 * 12  # 12 hours
    """Default TTL for cache items in seconds."""
    property_ttl: int = 60 * 60 * 24 * 7  # 7 days
    """TTL for property data in seconds."""
    user_ttl: int = 60 * 60 * 24  # 24 hours
    """TTL for user data in seconds."""
    score_ttl: int = 60 * 60 * 24  # 24 hours
    """TTL for score data in seconds."""
    poi_ttl: int = 60 * 60 * 24 * 30  # 30 days
    """TTL for POI data in seconds."""


class MLflowConfig(BaseSettings):
    """Configuration for the MLflow server."""

    uri: Url = Url("http://analytics1.stg.propertygroup:5000")
    """The URI of the MLflow server."""
    username: str | None = None
    """The username for the MLflow server."""
    password: str | None = None
    """The password for the MLflow server."""
    token: str | None = None
    """The token for the MLflow server."""

    def set_environ(self) -> None:
        os.environ["MLFLOW_TRACKING_URI"] = str(self.uri)
        if self.token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = self.token
        elif self.username and self.password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.password
        else:
            raise ValueError("Either token or username and password must be set.")


class NumLABSConfig(BaseSettings):
    """Configuration for the NumLABS API."""

    poi_distance: int = 1000
    """The distance in meters for the POI."""
    poi_url: str = "https://api.rynek.numlabs.com/geo-assets/poi/"
    """The URL for the POI."""
    transport_distance: int = 3000
    """The distance in meters for the transport POI."""
    cache_expiration: int = 30  # in days
    """The cache expiration time in days."""


class PropertiesEmbeddingModelConfig(BaseSettings):
    mlflow_artifact_path: str | None = None
    """The path to the MLflow artifact."""
    feature_spec_type: Literal["apartments", "full"] = "full"
    """The type of feature spec to use."""
    batch_size: int = 2048
    """The batch size for the data loader."""
    num_workers: int = 2
    """The number of workers for the data loader."""
    persistent_workers: bool = True
    """Whether to use persistent workers for the data loader."""


class UserScoreConfig(BaseSettings):
    """Configuration for the user score calculation."""

    time_decay_factor: float = 0.3
    """Controls how much timestamps influence weights."""
    temperature: float = 0.2
    """Controls the sharpness of the softmax distribution."""


class LoggerConfig(BaseSettings):
    """Configuration for the logger."""

    level: str | int = logging.INFO
    """The logging level for the logger."""


class CeleryConfig(BaseSettings):
    """Configuration for Celery."""

    broker_redis: RedisConfig | None = None
    """The Redis configuration for the broker. If Redis Sentinel is used, this should be None."""
    result_redis: RedisConfig | None = None
    """The Redis configuration for the result backend. If Redis Sentinel is used, this should be None."""
    broker_redis_sentinel: RedisSentinelConfig | None = None
    """The Redis Sentinel configuration for the broker, if used."""
    result_redis_sentinel: RedisSentinelConfig | None = None
    """The Redis Sentinel configuration for the result backend, if used."""
    timezone: str = "Europe/Warsaw"
    """The timezone."""
    task_time_limit: int = 60 * 60 * 2  # 2h
    """The time limit for the task."""
    task_track_started: bool = True
    """Flag to track the started task."""
    result_expires: int = 60 * 60 * 12  # 12h
    """The expiration time for the result."""
    max_retries: int = 3
    """The maximum number of retries for the task."""
    retry_backoff: int = 60 * 10  # 10 minutes
    """The backoff time for the retries."""
    retry_jitter: bool = True
    """Flag to enable jitter for the retries."""

    def get_celery_config(self) -> dict[str, Any]:
        config = {
            "timezone": self.timezone,
            "task_time_limit": self.task_time_limit,
            "task_track_started": self.task_track_started,
            "result_expires": self.result_expires,
            "max_retries": self.max_retries,
            "retry_backoff": self.retry_backoff,
            "retry_jitter": self.retry_jitter,
        }

        # NOTE: The check are done here, not in a model_validators, because model_validators are called
        # also on default values, which are Nones and we don't want to raise an error in that case.
        if self.broker_redis_sentinel and not self.broker_redis:
            config["broker_url"] = self.broker_redis_sentinel.get_connection_string()
            config["broker_transport_options"] = {"master_name": self.broker_redis_sentinel.master}
        elif self.broker_redis and not self.broker_redis_sentinel:
            config["broker_url"] = self.broker_redis.get_connection_string()
        elif self.broker_redis and self.broker_redis_sentinel:
            raise ValueError("Both broker_redis and broker_redis_sentinel are set. Please use only one.")
        else:
            raise ValueError("Neither broker_redis nor broker_redis_sentinel is provided.")

        if self.result_redis_sentinel and not self.result_redis:
            config["result_backend"] = self.result_redis_sentinel.get_connection_string()
            config["result_backend_transport_options"] = {"master_name": self.result_redis_sentinel.master}
        elif self.result_redis and not self.result_redis_sentinel:
            config["result_backend"] = self.result_redis.get_connection_string()
        elif self.result_redis and self.result_redis_sentinel:
            raise ValueError("Both result_redis and result_redis_sentinel are set. Please use only one.")
        else:
            raise ValueError("Neither result_redis nor result_redis_sentinel is provided.")

        return config


class BaseRedisCacheConfig(BaseSettings, ABC):
    """Configuration for the Redis database."""

    expire_time: Optional[int] = None
    """The expiration time in seconds for the Redis keys."""


class RedisCacheConfig(RedisConfig, BaseRedisCacheConfig):
    """Configuration for the Redis cache."""

    pass


class RedisSentinelCacheConfig(RedisSentinelConfig, BaseRedisCacheConfig):
    """Configuration for the Redis Sentinel cache."""

    pass


class Config(BaseSettings):
    """Main configuration class aggregating settings for different application components."""

    app: AppConfig = AppConfig()
    """Configuration specific to the application."""
    db: DBConfig = DBConfig()
    """Configuration for the database."""
    mlflow: MLflowConfig = MLflowConfig()
    """Configuration for the MLflow server."""
    numlabs: NumLABSConfig = NumLABSConfig()
    """Configuration for the NumLABS API."""
    properties_embedding_model: PropertiesEmbeddingModelConfig = PropertiesEmbeddingModelConfig()
    """Configuration for the properties embeddings."""
    logger: LoggerConfig = LoggerConfig()
    """Configuration for the logger."""
    redis: RedisCacheConfig | None = None
    """Configuration for the Redis client."""
    redis_sentinel: RedisSentinelCacheConfig | None = None
    """Configuration for the Redis Sentinel client."""
    cache: CacheConfig = CacheConfig()
    """Configuration for the cache."""
    user_score: UserScoreConfig = UserScoreConfig()
    """Configuration for the user score calculation."""
    celery: CeleryConfig = CeleryConfig()
    """Configuration for Celery."""

    # Env vars setup
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        # `.env.prod` takes priority over `.env`
        env_file=(os.path.join(CURRENT_DIR, *[".."] * 2, "config", ".env")),
    )

    def get_redis_config(self) -> RedisCacheConfig | RedisSentinelCacheConfig:
        """Get the Redis configuration based on the provided settings."""

        # NOTE: The check are done here, not in a model_validator, because model_validators are called
        # also on default values, which are Nones and we don't want to raise an error in that case.
        if self.redis_sentinel and not self.redis:
            return self.redis_sentinel
        elif self.redis and not self.redis_sentinel:
            return self.redis
        elif self.redis and self.redis_sentinel:
            raise ValueError("Both `redis` and `redis_sentinel` are set. Please use only one.")
        else:
            raise ValueError("Neither `redis` nor `redis_sentinel` is set. Please provide one.")


@lru_cache
def get_config() -> Config:
    """Return the app's configuration.

    :return Config: The configuration of the app
    """
    config = Config()  # type: ignore
    # set mlflow env vars
    config.mlflow.set_environ()
    return config


ConfigDep = Annotated[Config, Depends(get_config)]

if __name__ == "__main__":
    print(get_config())
