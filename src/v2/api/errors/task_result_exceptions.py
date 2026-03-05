from fastapi import HTTPException, status


class TaskNotSuccessfulException(HTTPException):
    status_code: int = status.HTTP_400_BAD_REQUEST
    detail: str = "Requested task does not have a successful state."

    def __init__(
        self,
    ) -> None:
        super().__init__(status_code=self.status_code, detail=self.detail)
