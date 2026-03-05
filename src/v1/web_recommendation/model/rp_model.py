import logging
import time
from datetime import date, datetime

from numpy import meshgrid
from pandas import DataFrame, concat, read_csv
from psycopg2.extensions import AsIs
from scipy.sparse import csr_matrix

from sql.scripts import FIND_PROPERTIES
from src.v1.web_recommendation.consts import const
from src.v1.web_recommendation.consts.paths import DIR_CSV_APPLICATIONS, DIR_CSV_VIEWS
from src.v1.web_recommendation.managers.data_generation import (
    open_activities_dm,
    pivot_lead_table,
    user_applications_tracker_agg,
    user_views_agg,
)
from src.v1.web_recommendation.managers.db_manager import query_fetch


def _neighbours(sim):
    """
    Filter for prediction only users with similarity above the set threshold.
    :param sim: calculated jaccard similarity
    :return: users above the set threshold of similarity
    """
    return [i for i, v in enumerate(sim["similarity"]) if v >= const.KNN_THRESHOLD]


def _jaccard_distance(x, y=None):
    """
    Computes the Jaccard similarity between two sparse matrices or between all pairs in one sparse matrix.
    :param x: A sparse matrix.
    :param y: A sparse matrix (optional).
    :return: Jaccard similarity
    """
    if y is None:
        y = x
    assert x.shape[1] == y.shape[1]

    x = x.astype(bool).astype(int)
    y = y.astype(bool).astype(int)

    intersect = x.dot(y.T)

    x_sum = x.sum(axis=1).A1
    y_sum = y.sum(axis=1).A1
    xx, yy = meshgrid(x_sum, y_sum)
    union = (xx + yy).T - intersect
    return DataFrame((intersect / union).A, columns=["similarity"])


def _today(min_time=True):
    return datetime.combine(date.today(), datetime.min.time() if min_time else datetime.max.time())


def _get_top_5_offers(old_results, new_results):
    """
    The function compares new recommendations with the old ones and returns only the TOP 5 with the highest score.
    :param old_results: current TOP5 recommendations
    :param new_results: potential better recommendations
    :return: views history of the forcasted users
    """
    old_res = DataFrame(old_results.items(), columns=["offer", "sim"])
    new_res = DataFrame(new_results.items(), columns=["offer", "sim"])
    concat_res = (
        concat([old_res, new_res])
        .sort_values(by=["offer", "sim"], ascending=False)
        .drop_duplicates()
        .sort_values(by=["sim"], ascending=False)
        .head(5)
    )
    del old_res, new_res
    concat_res = concat_res.set_index("offer")
    return concat_res.to_dict()["sim"]


def score_user(history_of_user):
    """
    The function is searching for best offers to user by using Collaborative Filtering (user-based technique).
    :param history_of_user: leads history for the forecasted user
    :return: offers and number of room - that interested the most similar users
    """
    start_time = time.time()
    user_uuid = history_of_user["uuid"].iloc[0]
    to_exclude = list(set(list(history_of_user["offer"]) + _find_latest_views(user_uuid)))

    final_results = {}
    for applications_chunk in read_csv(DIR_CSV_APPLICATIONS, chunksize=const.APPLICATIONS_CHUNK_SIZE):
        if applications_chunk is not None and len(applications_chunk) > 0:
            chunk_history = pivot_lead_table(concat([history_of_user, applications_chunk]))
            chunk_history_all_users = chunk_history[chunk_history.index != user_uuid]
            history_users = chunk_history[chunk_history.index == user_uuid]
            sim = _jaccard_distance(csr_matrix(chunk_history_all_users), csr_matrix(history_users))
            neighbor = chunk_history_all_users.reset_index()[sim.index.isin(_neighbours(sim))]
            results = neighbor.drop(to_exclude, axis="columns", errors="ignore").set_index("uuid").mean()
            results = results[results > 0]

            new_results = results.sort_values(ascending=False).head(const.TOP_N_RECOMMENDATION).to_dict()
            final_results = _get_top_5_offers(final_results, new_results)

    del (
        sim,
        neighbor,
        history_of_user,
        to_exclude,
        results,
        new_results,
        chunk_history,
        chunk_history_all_users,
        history_users,
    )
    logging.debug(f"Time:{str(round(time.time() - start_time, 2))} - score user")
    return final_results.keys()


def _include_latest_leads(user_uuid):
    """
    :param user_uuid: User uuid from km.users_user
    :return: leads history of the forcasted users
    """
    start_time = time.time()
    new_leads = user_applications_tracker_agg(from_day=_today(), to_day=_today(min_time=False), user_uuid=user_uuid)
    logging.debug(f"Time:{str(round(time.time() - start_time, 2))} - get latest leads from query")

    applications_rec_dm = open_activities_dm(user_uuid)
    leads = (
        concat([applications_rec_dm, new_leads]).drop_duplicates().reset_index(drop=True)
        if len(new_leads) > 0
        else applications_rec_dm
    )

    if leads is None or len(leads) == 0:
        logging.debug("include views")
        old_agg_views = open_activities_dm(user_uuid, chunk=const.VIEWS_CHUNK_SIZE, filename=DIR_CSV_VIEWS)
        new_views = user_views_agg(from_day=_today(), to_day=_today(min_time=False), user_uuid=user_uuid)
        leads = (
            concat([old_agg_views, new_views]).drop_duplicates().reset_index(drop=True)
            if len(new_views) > 0
            else old_agg_views
        )
        del new_views, old_agg_views
    del start_time, new_leads, applications_rec_dm

    if leads is not None and len(leads) > 0:
        return leads


def _find_latest_views(user_id):
    """
    :param user_id: User id from km.users_user
    :return: views history of the forcasted users
    """
    views = open_activities_dm(user_id, chunk=const.VIEWS_CHUNK_SIZE, filename=DIR_CSV_VIEWS).offer.tolist()
    start_time = time.time()
    new_views = user_views_agg(from_day=_today(), to_day=_today(min_time=False), user_uuid=user_id).offer.tolist()

    logging.debug(f"Time:{str(round(time.time() - start_time, 2))} - latest leads - query sql get lastest views")

    return list(set(views + new_views))


def recommend_properties(offers):
    """
    The function takes list of investments id and run function which returns list of recommended properties
    :param offers: list of investments: {offer_id}_{room no}
    :return: list of recommended properties
    """
    offer_list = "','".join(offers)
    property_list = query_fetch(FIND_PROPERTIES, {"offer_list": AsIs(offer_list)})
    del offer_list
    if len(property_list) > 0:
        return [recommended_property[0] for recommended_property in property_list]


def get_best_properties_from_rp(user_uuid):
    """
    The function is searching for best offers for selected user.
    :param user_uuid: uuid from km.users_user or session_id from tracker
    :return: TOP 5 offers from RynekPierwotny.pl
    """
    logging.debug("New recommendation for RP")
    total_time_start = time.time()

    user_history = _include_latest_leads(user_uuid)
    if user_history is not None and len(user_history) > 0:
        score_results = score_user(user_history)
        logging.debug(f"Finish - RP time:{str(round(time.time() - total_time_start, 2))}")
        del user_history, total_time_start, user_uuid
        return recommend_properties(score_results)
    else:
        del user_history, total_time_start, user_uuid
