from typing import cast

import pandas as pd
from celery import Task  # type: ignore[import-untyped]
from fastapi import Body, Depends, FastAPI, Response, status

from src.v2.api.middlewares.time_request import TimeRequest
from src.v2.api.models.prefill_cache import PrefillCacheRequest
from src.v2.api.models.scores import ScoresRequest, ScoresResponse
from src.v2.api.models.task_statuses import CeleryTask
from src.v2.api.security import validate_api_key
from src.v2.api.services.get_scores import get_scores_df
from src.v2.api.tasks import celery_app, prefill_cache_task
from src.v2.api.utils import get_scores_metadata, get_task_result, get_task_state, init_loguru
from src.v2.config import get_config

init_loguru()

app = FastAPI(
    title=get_config().app.api_name,
    version=cast(str, get_config().app.version),
    dependencies=[Depends(validate_api_key)],
)

app.add_middleware(TimeRequest)


@app.post(
    "/calculate_scores",
    summary="Calculate scores for user_id and property_id pairs",
    tags=["Scores"],
    response_model=ScoresResponse,
)
async def post_calculate_scores(body: ScoresRequest) -> ScoresResponse:
    input_df = pd.DataFrame(body.data)

    scores_df = await get_scores_df(input_df)

    return ScoresResponse(
        scores=scores_df.to_dict(orient="records"),  # type: ignore
        metadata=get_scores_metadata(),
    )


@app.post(
    "/prefill_cache/start",
    summary="Enqueue cache prefill as a Celery background task",
    tags=["Cache Prefill"],
    response_model=CeleryTask,
    response_description="The ID of the started task.",
    status_code=status.HTTP_202_ACCEPTED,
)
async def post_prefill_cache(body: PrefillCacheRequest = Body(default_factory=PrefillCacheRequest)) -> CeleryTask:
    return CeleryTask(
        cast(Task, prefill_cache_task).delay(
            overwrite_visible_properties=body.overwrite_visible_properties,
            overwrite_pois=body.overwrite_pois,
        )
    )


@app.get(
    "/prefill_cache/state",
    summary="Get the state of the prefill_cache task.",
    tags=["Cache Prefill"],
    response_model=CeleryTask,
)
def get_prefill_cache_state(task_id: str) -> CeleryTask:
    return get_task_state(task_id, celery_app)


@app.get(
    "/prefill_cache/result",
    summary="Get the result of the prefill_cache task.",
    tags=["Cache Prefill"],
)
def get_prefill_cache_result(task_id: str) -> Response:
    return get_task_result(task_id, celery_app)
