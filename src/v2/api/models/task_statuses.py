from enum import Enum

from celery.result import AsyncResult  # type: ignore[import-untyped]
from celery.states import READY_STATES  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, computed_field


class TaskStatesEnum(str, Enum):
    PENDING = "PENDING"
    RECEIVED = "RECEIVED"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    REVOKED = "REVOKED"
    REJECTED = "REJECTED"
    RETRY = "RETRY"
    IGNORED = "IGNORED"


class CeleryTask(BaseModel):
    """
    Represents a Celery task with its current state.

    Task state meanings:
    - PENDING: Task state is unknown (assumed pending since you know the id).
    - RECEIVED: Task was received by a worker (only used in events).
    - STARTED: Task was started by a worker.
    - SUCCESS: Task succeeded.
    - FAILURE: Task failed.
    - REVOKED: Task was revoked.
    - REJECTED: Task was rejected (only used in events).
    - RETRY: Task is waiting for retry.
    - IGNORED: Task is ignored.
    """

    task_id: str
    state: TaskStatesEnum = Field(..., description="The current state of the task.")

    def __init__(self, task: AsyncResult) -> None:
        super().__init__(task_id=task.id, state=task.state)

    @computed_field(description="Whether the task is ready (SUCCESS, FAILURE or REVOKED).")
    def ready(self) -> bool:
        return self.state in READY_STATES

    @computed_field(description="Whether the task is successful.")
    def successful(self) -> bool:
        return self.state == TaskStatesEnum.SUCCESS
