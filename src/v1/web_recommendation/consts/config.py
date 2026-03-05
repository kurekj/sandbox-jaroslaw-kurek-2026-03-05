import os

# db connection
DB_USER = os.environ["db_username"]
DB_PASSWORD = os.environ["db_password"]
DB_DATABASE = os.environ["db_database"]
DB_HOST = os.environ["db_host"]
DB_ENGINE_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_DATABASE}"
DB_CONN_STRING = f"host={DB_HOST} dbname='{DB_DATABASE}' user={DB_USER} password={DB_PASSWORD}"

DB_HOST_GH = os.environ["db_gh_host"]
DB_USER_GH = os.environ["db_gh_username"]
DB_PASS_GH = os.environ["db_gh_password"]
DB_ENGINE_STRING_GH = f"postgresql://{DB_USER_GH}:{DB_PASS_GH}@{DB_HOST_GH}/{DB_DATABASE}"
DB_CONN_STRING_GH = f"host={DB_HOST_GH} dbname='{DB_DATABASE}' user={DB_USER_GH} password={DB_PASS_GH}"

DB = {"RP": DB_CONN_STRING, "GH": DB_CONN_STRING_GH}

# celery
CELERY_BROKER_URL = os.environ["celery_broker_url"]
CELERY_RESULT_BACKEND = os.environ["celery_result_backend"]
CELERY_REDIS_SENTINEL_MASTER_NAME = os.environ["celery_redis_sentinel_master_name"]
CELERY_REDIS_SENTINEL_PASSWORD = os.environ.get("celery_redis_sentinel_password")
