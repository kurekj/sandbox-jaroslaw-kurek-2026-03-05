import logging
import time

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from sql import scripts
from src.v1.web_recommendation.consts.const import NUMBER_OF_GH_OFFERS
from src.v1.web_recommendation.managers.db_manager import query_fetch
from src.v1.web_recommendation.managers.knn_data_processing import (
    get_user_params,
    normalize_data,
    prepare_sql_script,
)


def _convert_into_list(input_list):
    return list(set([item[0] for item in input_list]))


def _exlude_gh_offers(user_uuid):
    offers_leaded_agg = _convert_into_list(
        query_fetch(scripts.GH_OFFERS_USER_APPLICATION_AGG, {"user_id": user_uuid}, db="GH")
    )
    offers_viewed_agg = _convert_into_list(
        query_fetch(scripts.GH_OFFERS_USER_VIEWS_AGG, {"user_id": user_uuid}, db="GH")
    )
    offers_leaded_last = _convert_into_list(
        query_fetch(scripts.GH_OFFERS_USER_APPLICATION_LAST, {"user_id": user_uuid}, db="GH")
    )
    offers_viewed_last = _convert_into_list(query_fetch(scripts.GH_OFFERS_USER_VIEWS, {"user_id": user_uuid}, db="GH"))
    offers = list(set(offers_leaded_agg + offers_viewed_agg + offers_leaded_last + offers_viewed_last))

    if len(offers) > 0:
        offers_text = ", ".join(str(item) for item in offers)
    else:
        offers_text = ""

    return offers_text


def _process_data(user_uuid):
    """
    The function retrieves user parameters using the _get_user_params function, prepares an SQL script using
    _prepare_sql_script, executes the SQL query to fetch similar properties from the database,
    and normalizes the data using Min Max.
    :param user_uuid: User ID from the housing account
    :return: normalized similar properties, and user preferences
    """
    user_params = get_user_params(user_uuid)

    if user_params is not None and user_params["geopoints"][0] is not None:
        sql, params, user_data = prepare_sql_script(user_params, _exlude_gh_offers(user_uuid))
        similar_properties = pd.DataFrame(
            query_fetch(sql, params=params, db="GH"),
            columns=["id", "distance", "price"],
        )
        similar_properties = similar_properties.dropna()
        if len(similar_properties) > 0:
            similar_properties_normalized, property_data_normalized = normalize_data(
                similar_properties, user_data, ["distance", "price"]
            )
            return similar_properties_normalized, property_data_normalized
        else:
            return None, None
    else:
        return None, None


def get_best_properties_from_gh(user_uuid):
    """
    The function calls _process_data to obtain the normalized similar properties and property data.
    It determines the value of k (number of neighbors) based on the square root of the number of similar properties.
    It trains a k-nearest neighbors (KNN) model using the normalized similar properties.
    It then uses the trained model to predict clusters for the user's property data.
    Finally, it retrieves the top properties from the normalized similar properties based on the predicted clusters
    and returns their IDs as a list.
    :param user_uuid: User ID from the housing account
    :return: The function returns the top N properties that are most similar to the user's preferences.
    """
    logging.debug("New recommendation for Gethome")
    total_time_start = time.time()

    similar_properties_normalized, property_data_normalized = _process_data(user_uuid)
    if similar_properties_normalized is not None:
        neighbors_number = min([len(similar_properties_normalized), NUMBER_OF_GH_OFFERS])

        # KNN model training
        knn_model = NearestNeighbors(n_neighbors=neighbors_number)
        knn_model.fit(similar_properties_normalized[["distance", "price"]])

        # Predicting clusters for user datas
        indices = knn_model.kneighbors(property_data_normalized[["distance", "price"]], return_distance=False)

        top_properties = similar_properties_normalized.iloc[indices[0]]["id"].tolist()

        logging.debug(f"Finish - Gethome time:{str(round(time.time() - total_time_start, 2))}")
        return top_properties
