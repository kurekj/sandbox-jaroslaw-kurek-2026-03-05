import secrets
from typing import Optional

from fastapi import Request, Security
from fastapi.security import APIKeyHeader

from src.v2.api.errors import UnauthorizedException
from src.v2.config import ConfigDep, get_config

api_key_header = APIKeyHeader(
    name=get_config().app.api_key_header,
    auto_error=False,
)


def validate_api_key(
    _: Request,
    config: ConfigDep,
    api_key: Optional[str] = Security(api_key_header),
) -> None:
    """Validates if an upcoming request has a correct API key.

    Args:
        _: An upcoming request.
        config: The app configuration.
        api_key: An API key extracted from a request (default: Security(api_key_header)).

    Raises:
        UnauthorizedException: Raised whenever a user does not have a valid API key.
    """
    if api_key is None:
        raise UnauthorizedException()

    request_api_key_bytes = api_key.encode()
    server_api_key_bytes = config.app.api_key.encode()

    # compare_digest is used as a good practice against timing attacks.
    if not secrets.compare_digest(request_api_key_bytes, server_api_key_bytes):
        raise UnauthorizedException()
