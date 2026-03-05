import logging

import uvicorn

from src.v2.config import get_config

if __name__ == "__main__":
    app_config = get_config().app
    logging_config = get_config().logger
    uvicorn.run(
        "src.v2.api.app:app",
        host=app_config.host,
        port=app_config.port,
        log_level=logging_config.level
        if not isinstance(logging_config.level, str)
        else logging._nameToLevel[logging_config.level],
        workers=app_config.workers,
        forwarded_allow_ips="*",  # Needed when running behind a reverse proxy like NGINX
        root_path=app_config.root_path,  # Needed for reverse proxy setups with stripping of the root path
    )
