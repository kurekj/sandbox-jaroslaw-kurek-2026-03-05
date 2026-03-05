import asyncio
from typing import Any

from celery import Celery  # type: ignore[import-untyped]
from celery.signals import worker_process_init  # type: ignore[import-untyped]

from src.v2.api.utils.logging import init_loguru
from src.v2.config import get_config
from src.v2.utils.prefill_cache import prefill_cache

app_config = get_config()
celery_app = Celery("tasks")
celery_app.config_from_object(app_config.celery.get_celery_config())


@worker_process_init.connect  # type: ignore[misc]
def init_worker(**kwargs: Any) -> None:
    """Initialize loguru when worker process starts"""
    init_loguru()


@celery_app.task(autoretry_for=(Exception,))  # type: ignore
def prefill_cache_task(overwrite_visible_properties: bool = True, overwrite_pois: bool = False) -> None:
    return asyncio.run(
        prefill_cache(
            overwrite_visible_properties=overwrite_visible_properties,
            overwrite_pois=overwrite_pois,
        )
    )
