from celery import Celery

from src.v1.web_recommendation.api.module.sentry import initialize_sentry
from src.v1.web_recommendation.consts.config import (
    CELERY_BROKER_URL,
    CELERY_REDIS_SENTINEL_MASTER_NAME,
    CELERY_REDIS_SENTINEL_PASSWORD,
    CELERY_RESULT_BACKEND,
)

SENTINEL_KWARGS = (
    {
        "password": CELERY_REDIS_SENTINEL_PASSWORD,
    }
    if CELERY_REDIS_SENTINEL_PASSWORD
    else {}
)

initialize_sentry()

app = Celery(__name__)

app.conf.broker_url = CELERY_BROKER_URL
app.conf.broker_transport_options = {
    "master_name": CELERY_REDIS_SENTINEL_MASTER_NAME,
    "sentinel_kwargs": SENTINEL_KWARGS,
}
app.conf.result_backend = CELERY_RESULT_BACKEND
app.conf.result_backend_transport_options = {
    "master_name": CELERY_REDIS_SENTINEL_MASTER_NAME,
    "sentinel_kwargs": SENTINEL_KWARGS,
}
app.autodiscover_tasks(
    packages=[
        "src.v1.web_recommendation.api",
    ]
)
