from typing import cast

from celery import Celery  # type: ignore[import-untyped]
from celery.result import AsyncResult  # type: ignore[import-untyped]
from fastapi import Response

from src.v2.api.errors.task_result_exceptions import TaskNotSuccessfulException
from src.v2.api.models.task_statuses import CeleryTask


def get_task_state(task_id: str, celery_app: Celery) -> CeleryTask:
    """
    Retrieve the state of a Celery task given its task ID.

    Args:
        task_id (str): The unique identifier of the Celery task.

    Returns:
        CeleryTask: An instance of CeleryTask representing the state of the task.
    """
    return CeleryTask(cast(AsyncResult, celery_app.AsyncResult(task_id)))


def get_task_result(task_id: str, celery_app: Celery) -> Response:
    """
    Retrieve the result of a Celery task by its task ID.

    Args:
        task_id (str): The unique identifier of the Celery task.

    Returns:
        Response: A FastAPI Response object containing the task result as JSON.

    Raises:
        TaskNotSuccessfulException: If the task was not successful.
    """
    task = cast(AsyncResult, celery_app.AsyncResult(task_id))
    # type ignore because CeleryTask is needed since computed_filed is used
    if CeleryTask(task).successful:  # type: ignore
        # content is already a JSON string
        return Response(content=task.result, media_type="application/json")  # type: ignore
    raise TaskNotSuccessfulException()
