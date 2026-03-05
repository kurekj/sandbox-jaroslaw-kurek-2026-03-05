import pandas as pd
from psycopg2.extensions import AsIs
from sklearn.preprocessing import MinMaxScaler

from sql.scripts import (
    FIND_GH_PROPERTIES_ONE_AREA,
    FIND_GH_PROPERTIES_TWO_AREAS,
    GET_DATA_FROM_USER_VECTOR_GH_OFFERS,
    GET_LAST_2_GH_LEADS,
    GET_LAST_2_RP_LEADS,
)
from src.v1.web_recommendation.consts import const
from src.v1.web_recommendation.managers.db_manager import query_fetch


def _find_dict_columns(df):
    """
    This function takes a DataFrame df as input.
    It checks each element of the DataFrame to identify dictionary columns and returns the names of those columns.
    :param df: DataFrame to verify
    :return: The function returns columns of dictionary type.
    """
    is_dict = df.applymap(lambda x: type(x).__name__ == "dict" and bool(x))
    dict_cols = is_dict.any(axis=0).index[is_dict.any(axis=0)]
    return dict_cols


def _cnt_descr_stats_from_key_value(df, descr_stat="min"):
    """
    This function takes a dictionary-like object df and a descr_stat parameter specifying the desired descriptive
    statistic. It retrieves the minimum key, maximum key, or the top key from the dictionary based on the descr_stat
    value.
    :param df: DataFrame for modification
    :param descr_stat: Min/Max/TOP - most popular value
    :return: the expected value for the selected measure.
    """
    if "None" in df:
        del df["None"]
    if df is not None and type(df).__name__ == "dict" and len(df) > 0:
        if descr_stat == "min":
            return min(list(df.keys()))
        if descr_stat == "max":
            return max(list(df.keys()))
        if descr_stat == "top_key":
            return max(df, key=df.get)
    else:
        return None


def _get_latest_location(client_id, user_id):
    """
    This function takes client_id and user_id as input and retrieves the latest location information from two separate
    databases: RP and GH. It merges the results into a DataFrame, sorts it based on the create_date column
    in descending order, and returns location based on last two leads.
    :param client_id: client_id from rds2.applications_application
    :param user_id: User ID from the housing account
    :return: geo_points based on last two leads
    """
    if client_id is not None and user_id is not None:
        rp_leads = pd.DataFrame(
            query_fetch(GET_LAST_2_RP_LEADS, {"client_id": int(client_id)}),
            columns=["offer_id", "create_date", "geo_point"],
        )
        gh_leads = pd.DataFrame(
            query_fetch(GET_LAST_2_GH_LEADS, {"user_id": int(user_id)}, db="GH"),
            columns=["offer_id", "create_date", "geo_point"],
        )
        geopoints = pd.concat([rp_leads, gh_leads]).sort_values(by=["create_date"], ascending=False).head(2)
        return ",".join("'" + geopoints["geo_point"] + "'")


def _lead_first(first, second, cast_to_int=True):
    """
    This function takes two values first and second. It checks if the first value is not None and returns it.
    If the first value is None, it returns the second value. If both values are None, it returns None.
    :param first: first lead value
    :param second: second views value
    :return: first existing value
    """
    if bool(first[0]):
        selected = first[0]
    elif bool(second[0]):
        selected = second[0]
    else:
        selected = None

    if cast_to_int:
        return int(selected)
    else:
        return selected


def _prepare_data_for_sql(user_params):
    """
    This function takes a dictionary-like object user_params as input.
    It extracts various parameters needed for SQL script preparation
    based on the provided user parameters and returns them.
    :param user_params: calculated parameters
    :return: user parameters needed for sql
    """
    offer_type = _lead_first(user_params["offer_type_leaded"], user_params["offer_type_viewed"])
    rooms = _lead_first(user_params["rooms_leaded"], user_params["rooms_viewed"])
    price_type = _lead_first(user_params["price_segment_leaded"], user_params["price_segment_views"])

    geopoints = user_params["geopoints"][0]

    min_area, max_area = 0, 500
    offer_type_name = "apartment" if int(offer_type) == 1 else "house"
    area = _lead_first(
        user_params["area_segment_{}s_leaded".format(offer_type_name)],
        user_params["area_segment_{}s_viewed".format(offer_type_name)],
        cast_to_int=False,
    )
    if area is not None:
        min_area, max_area = area[0], area[1]

    return offer_type, rooms, geopoints, min_area, max_area, price_type


def prepare_sql_script(user_params, offers_to_exclude):
    """
    This function takes a dictionary-like object user_params as input.
    It calls _prepare_data_for_sql to obtain the necessary parameters,
    and then constructs an SQL script based on those parameters.
    The constructed SQL script and a DataFrame with user data are returned.
    :param user_params: user parameter's value
    :param offers_to_exclude: list of offers to exclude from query
    :return: sql script, and user data needed for the model
    """
    (
        offer_type,
        rooms,
        geopoints,
        min_area,
        max_area,
        price_type,
    ) = _prepare_data_for_sql(user_params)
    user_data = pd.DataFrame([{"distance": 0, "price": price_type, "id": None}])
    offer_type_tab = "rds2.mdb_apartments_apartment" if offer_type == 1 else "rds2.mdb_houses_house"
    id_param_name = "moo." + offer_type_tab.split("_")[2] + "_id"
    distance = 3 if offer_type == 1 else 10

    geopoints_split = geopoints.split(",")
    geopoints_1 = geopoints_split[0].strip("'")

    sql_txt = None
    params = None

    if "," in geopoints:
        geopoints_2 = geopoints_split[1].strip("'")
        sql_txt = (
            FIND_GH_PROPERTIES_TWO_AREAS.replace("AND moo.id NOT IN (%(exclude)s)", "")
            if len(offers_to_exclude) == 0
            else FIND_GH_PROPERTIES_TWO_AREAS
        )

        params = {
            "offer_type": offer_type,
            "rooms": rooms,
            "min_area": min_area,
            "max_area": max_area,
            "distance": distance,
            "geo_point": geopoints_1,
            "geo_point_2": geopoints_2,
            "offer_type_tab": AsIs(offer_type_tab),
            "id_param": AsIs(id_param_name),
            "exclude": AsIs(offers_to_exclude),
        }

    elif geopoints:
        sql_txt = (
            FIND_GH_PROPERTIES_ONE_AREA.replace("AND moo.id NOT IN (%(exclude)s)", "")
            if len(offers_to_exclude) == 0
            else FIND_GH_PROPERTIES_ONE_AREA
        )

        params = {
            "offer_type": offer_type,
            "rooms": rooms,
            "min_area": min_area,
            "max_area": max_area,
            "distance": distance,
            "geo_point": geopoints_1,
            "offer_type_tab": AsIs(offer_type_tab),
            "id_param": AsIs(id_param_name),
            "exclude": AsIs(offers_to_exclude),
        }

    return sql_txt, params, user_data


def get_user_params(user_uuid):
    """
    This function takes a user_uuid as input and retrieves user parameters from a SQL query.
    It modifies and processes the retrieved data, including finding the latest location,
    extracting dictionary columns, replacing values, and mapping certain columns based on predefined dictionaries.
    The processed user parameters are returned.
    :param user_uuid: user session_id
    :return:
    """
    user_params = pd.DataFrame(
        query_fetch(GET_DATA_FROM_USER_VECTOR_GH_OFFERS, {"user_id": user_uuid}),
        columns=const.USER_PARAMS,
    )
    if len(user_params) > 0:
        user_params["geopoints"] = _get_latest_location(user_params["client_id"][0], user_params["user_id"][0])
        user_params["uuid"] = user_uuid
        dicts = _find_dict_columns(user_params)
        for dict_popular in dicts:
            user_params[dict_popular] = user_params[dict_popular].apply(
                lambda x: (
                    _cnt_descr_stats_from_key_value(x, "top_key").astype(float)
                    if x in const.AREA_SEGMENT + const.PRICE_SEGMENT
                    else _cnt_descr_stats_from_key_value(x, "top_key")
                )
            )

        # replace min_max
        for area_segment_tab_name in const.AREA_SEGMENT:
            if bool(user_params[area_segment_tab_name][0]):
                user_params[area_segment_tab_name] = user_params[area_segment_tab_name].map(
                    const.AREA_SEGMENT_APARTMENTS_DICT
                    if "apartment" in area_segment_tab_name
                    else const.AREA_SEGMENT_HOUSES_DICT
                )

        for price_segment_tab_name in const.PRICE_SEGMENT:
            if bool(user_params[price_segment_tab_name][0]):
                user_params[price_segment_tab_name] = user_params[price_segment_tab_name].map(const.PRICE_SEGMENT_DICT)

        return user_params


def normalize_data(df_to_fit, second_df, columns_to_normalize):
    """
    Function normalizes selected parameters based on data from the relevant column in the data frame.
    :param df_to_fit: data frame to fit and transform
    :param second_df: dataframe to transform by scaler from above fitting
    :param columns_to_normalize:list of column names from the data frame to be normalized
    :return: a data frame with normalized columns and a dictionary with normalized values and their names
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    for col in columns_to_normalize:
        if not df_to_fit[[col]].empty:
            fit_scaler = scaler.fit(df_to_fit[[col]].values)
            df_to_fit[col] = fit_scaler.transform(df_to_fit[[col]].values)
            second_df[col] = fit_scaler.transform(second_df[[col]].values)
    return df_to_fit, second_df
