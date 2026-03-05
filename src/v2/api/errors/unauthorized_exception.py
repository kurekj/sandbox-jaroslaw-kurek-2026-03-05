from fastapi import HTTPException, status


class UnauthorizedException(HTTPException):
    status_code: int = status.HTTP_401_UNAUTHORIZED
    detail: str = "Missing or incorrect API KEY"

    def __init__(
        self,
    ) -> None:
        super().__init__(
            status_code=self.status_code,
            detail=self.detail,
            headers={"WWW-Authenticate": "API-KEY"},
        )
