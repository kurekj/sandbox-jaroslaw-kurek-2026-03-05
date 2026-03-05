import logging
import os
import tempfile
import time
from datetime import datetime, timedelta

import pandas as pd
from filelock import FileLock, Timeout

from sql.scripts import (
    APPLICATIONS_AGG_SCRIPT,
    APPLICATIONS_GET_DM,
    VIEWS_AGG_SCRIPT,
    VIEWS_GET_DM,
)
from src.v1.web_recommendation.consts.config import DB_ENGINE_STRING
from src.v1.web_recommendation.consts.const import (
    APPLICATIONS_CHUNK_SIZE,
    LEADS_DAYS_AGO_FROM,
    VIEWS_CHUNK_SIZE,
)
from src.v1.web_recommendation.consts.paths import (
    DIR_CSV_APPLICATIONS,
    DIR_CSV_VIEWS,
    DIR_REDIS_TIME,
)
from src.v1.web_recommendation.managers.db_manager import query_fetch
from src.v1.web_recommendation.managers.redis_up_to_date import get_redis_current_date


def _from_date():
    return datetime.combine(
        datetime.now() - timedelta(days=LEADS_DAYS_AGO_FROM),
        datetime.min.time(),
    )


def _to_date():
    return datetime.combine(datetime.today(), datetime.min.time())


def pivot_lead_table(activities):
    """
    The function makes a pivot table from the data frame.
    :param activities: users activities like leads or views
    :return: pivot table from given data frame
    """
    pivot_df = activities.groupby(["uuid", "offer"]).size().unstack().notnull().astype(int)
    del activities
    return pivot_df


def user_views_agg(from_day=_from_date(), to_day=_to_date(), user_uuid=None):
    """
    The function groups log_data about users views basing on a copy of the production database.
    :param from_day: the date from which log_data are aggregated
    :param to_day: the date to which log_data are aggregated
    :param user_uuid: by Default is NULL - then all users are included, otherwise specific user.
    :return: user_uuid and viewed offers_id with number of room
    """
    views_sql = (
        VIEWS_AGG_SCRIPT.replace(
            """AND coalesce(nullif(split_part(split_part(ext_cookie, 'algolytics_uuid=', 2),';',1),'')::varchar(36),
            m.algolytics_uuid::varchar(36), v.session_id::varchar(36)) = %(user_uuid)s""",
            "",
        )
        if user_uuid is None
        else VIEWS_AGG_SCRIPT
    )

    views_agg_results = pd.DataFrame(
        query_fetch(
            views_sql,
            {"from_date": from_day, "to_date": to_day, "user_uuid": user_uuid},
        ),
        columns=["uuid", "offer"],
    )
    return views_agg_results


def user_applications_tracker_agg(from_day=_from_date(), to_day=_to_date(), user_uuid=None):
    """
    The function groups log_data about users applications basing on a log_data from tracker.
    :param from_day: the date from which log_data are aggregated
    :param to_day: the date to which log_data are aggregated
    :param user_uuid: by Default is NULL - then all users are included, otherwise specific user.
    :return: user_uuid and quered offers_id with number of room
    """
    applications_sql = (
        APPLICATIONS_AGG_SCRIPT.replace(
            """AND coalesce(nullif(split_part(split_part(ext_cookie, 'algolytics_uuid=', 2),';',1),'')::varchar(36),
            m.algolytics_uuid::varchar(36), v.session_id::varchar(36)) = %(user_uuid)s""",
            "",
        )
        if user_uuid is None
        else APPLICATIONS_AGG_SCRIPT
    )

    apps_agg_results = pd.DataFrame(
        query_fetch(
            applications_sql,
            {"from_date": from_day, "to_date": to_day, "user_uuid": user_uuid},
        ),
        columns=["uuid", "offer"],
    )

    return apps_agg_results


def _save_sql_in_chunks_to_csv(sql_query, chunk, filename):
    temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv", encoding="utf-8")

    is_first_loop = True
    for data_chunk in pd.read_sql_query(sql_query, DB_ENGINE_STRING, chunksize=chunk):
        data_chunk.to_csv(temp_file, mode="a", index=False, header=is_first_loop)
        is_first_loop = False

    os.replace(temp_file.name, filename)


def is_datamart_up_to_date():
    """
    The function verify if the data ara up to date. If not run update of csv files.
    """
    for i in range(20):
        try:
            with open(DIR_REDIS_TIME, "r") as file:
                datamart_current_date = file.read()
                if not get_redis_current_date() == datamart_current_date:
                    lock = FileLock(DIR_REDIS_TIME)
                    with lock.acquire(timeout=120):
                        if not get_redis_current_date() == datamart_current_date:
                            update_applications_views_dm()
                break
        except Timeout:
            logging.error(f"Another instance of this application currently holds the lock. Attempt number:{i+1}/20")
        time.sleep(6)


def update_applications_views_dm():
    """
    The function downloads previously generated views and applications.
    """
    _save_sql_in_chunks_to_csv(VIEWS_GET_DM, VIEWS_CHUNK_SIZE, DIR_CSV_VIEWS)
    _save_sql_in_chunks_to_csv(APPLICATIONS_GET_DM, APPLICATIONS_CHUNK_SIZE, DIR_CSV_APPLICATIONS)

    # Save the UTC time to a file
    redis_current_date = get_redis_current_date()
    with open(DIR_REDIS_TIME, "w") as file:
        file.write(redis_current_date)


def open_activities_dm(user_uuid, chunk=APPLICATIONS_CHUNK_SIZE, filename=DIR_CSV_APPLICATIONS):
    """
    The function reads previously generated activities by user.
    Applications by default, but may be also views.
    :param user_uuid: user id
    :param chunk: number of fetched records during a one loop
    :param filename: path to the file
    :return: activities genereted by user_uuid
    """
    if user_uuid is not None:
        user_activities = pd.DataFrame(columns=["uuid", "offer"])
        for activities_chunk in pd.read_csv(filename, chunksize=chunk):
            activities_chunk = activities_chunk[activities_chunk["uuid"] == user_uuid]
            if not activities_chunk.empty:
                user_activities = user_activities.append(activities_chunk)
        del activities_chunk
        return user_activities
