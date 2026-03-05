import os

import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.flask import FlaskIntegration

from web_recommendation import __version__


def initialize_sentry():
    if SENTRY_DSN := os.getenv("SENTRY_DSN"):
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[FlaskIntegration(), CeleryIntegration()],
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            # We recommend adjusting this value in production.
            traces_sample_rate=0.1,
            environment=ENVIRONMENT,
            release=__version__,
        )


ENVIRONMENT = os.getenv("ENVIRONMENT")
