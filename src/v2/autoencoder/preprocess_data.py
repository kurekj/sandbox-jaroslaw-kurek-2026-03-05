import asyncio
import os
from functools import partial
from typing import Any, Callable, Coroutine, Optional, cast

import numpy as np
import pandas as pd
import swifter  # type: ignore # noqa: F401
from loguru import logger

from src.v2.autoencoder.get_poi_data import AGG_POI, get_all_pois
from src.v2.autoencoder.load_data import OFFERS_DICTS_QUERY, OFFERS_QUERY, PROPERTIES_DICTS_QUERY, PROPERTIES_QUERY
from src.v2.db_utils import execute_query
from src.v2.utils.encode_to_mhot import encode_to_mhot

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

VISIBLE_PROPERTIES_QUERY = f"""
with visible_offers as (
{OFFERS_QUERY}
where
	oo.vendor_id not in (979, 11130, 4879)
	and not oc.is_invalid
    and oc.is_active
	and rr.country = %s
    and oc.display_type = 1
    and os.display_status = 1
),
properties as (
{PROPERTIES_QUERY}
wheref
	not pc.is_invalid
    and pp.for_sale
    and pc.display_type = 1
)
select
    vo.*,
    p.*
from
    properties p
inner join visible_offers vo on
    p.offer_id = vo.offer_id
"""

VISIBLE_FOR_SALE_PROPERTIES_QUERY = f"""
with visible_offers as (
{OFFERS_QUERY}
where
	oo.vendor_id not in (979, 11130, 4879)
	and not oc.is_invalid
    and oc.is_active
	and rr.country = %s
    and oc.display_type = 1
    and os.display_status = 1
    and not oc.limited_presentation
),
properties as (
{PROPERTIES_QUERY}
where
	not pc.is_invalid
    and pp.for_sale
)
select
    vo.*,
    p.*
from
    properties p
inner join visible_offers vo on
    p.offer_id = vo.offer_id
"""

SELECTED_PROPERTIES_QUERY = f"""
with offers as (
{OFFERS_QUERY}
),
properties as (
{PROPERTIES_QUERY}
where
    pp.id = any(%s)
)
select
    o.*,
    p.*
from
    properties p
inner join offers o on
    p.offer_id = o.offer_id
"""


async def load_current_properties_data(
    ids: Optional[list[int]] = None,
    country: int = 1,
    only_for_sale: bool = False,
) -> pd.DataFrame:
    if ids is not None:
        properties = pd.DataFrame(await execute_query(SELECTED_PROPERTIES_QUERY, (ids,)))
    elif only_for_sale:
        properties = pd.DataFrame(await execute_query(VISIBLE_FOR_SALE_PROPERTIES_QUERY, (country,)))
    else:
        properties = pd.DataFrame(await execute_query(VISIBLE_PROPERTIES_QUERY, (country,)))

    offer_dicts = pd.DataFrame(await execute_query(OFFERS_DICTS_QUERY))
    properties_dicts = pd.DataFrame(await execute_query(PROPERTIES_DICTS_QUERY))

    properties = properties.merge(offer_dicts, how="cross")
    properties = properties.merge(properties_dicts, how="cross")

    return properties


def _group_additional_area_type(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Grouping additional area type")
    df = (
        df.groupby("property_id")
        .agg(
            {
                **{
                    col: "first"
                    for col in df.columns
                    if col not in ["additional_area_type", "additional_area_area", "property_id"]
                },
                # Collect additional_area_type values per property
                "additional_area_type": lambda x: [int(i) for i in x if pd.notna(i)],
                "additional_area_area": "sum",
            }
        )
        .reset_index()
    )
    df["additional_area_area"] = df["additional_area_area"].map(lambda x: x if x != 0 else np.nan)
    return df


def _normalize_price_m2(row: pd.Series, price_index: pd.DataFrame) -> tuple[float, Any]:  # type: ignore
    # if city is in price_index calculate normalized price_m2
    t = row["create_date"]
    city = row["city_name"]
    normalized_price = row["price_m2"]
    update_type = None

    if pd.isna(row["price_m2"]):
        return normalized_price, np.nan

    price_index_row: pd.DataFrame = price_index[(price_index["period_start"] <= t) & (price_index["period_end"] >= t)]

    if len(price_index_row) > 1:
        raise ValueError("More than one price index row found")
    elif city in price_index_row.columns and len(price_index_row) == 1:
        normalized_price *= 100 / price_index_row[city].values[0]
        update_type = "city"
    elif len(price_index_row) == 1:
        normalized_price *= 100 / price_index_row["Average_HPI"].values[0]
        update_type = "average"

    return normalized_price, update_type


def _normalize_price(df: pd.DataFrame, skip: bool = True) -> pd.DataFrame:
    logger.info("Normalizing price")

    # for rows where there is price but no price_m2 calculate it by dividing price by area
    df["price_m2"] = df["price_m2"].combine_first(df["price"].astype(float) / df["area"].astype(float))

    if skip:
        # When skip is True, we assume that price_m2 and price are already normalized
        df["normalize_price_m2"] = df["price_m2"].copy()
        df["normalize_price"] = df["price"].copy()
    else:
        logger.warning(
            "The price normalization seems to not really work as expected. Please check results after running."
        )
        # raise exception if housing_price_index.jsonl does not exist
        if not os.path.exists(
            os.path.join(CURRENT_DIR, *[".."] * 3, "data", "autoencoder", "housing_price_index.jsonl")
        ):
            raise FileNotFoundError(
                "housing_price_index.jsonl not found. Please run the price_index.ipynb notebook to generate it "
                + "or add it to the docker image."
            )

        # calculate average price over each quarter
        df["quarter"] = df["create_date"].dt.tz_localize(None).dt.to_period("Q").dt.to_timestamp()

        # load data/autoencoder/housing_price_index.jsonl
        logger.debug("Loading housing price index data")
        price_index = pd.read_json(
            os.path.join(CURRENT_DIR, *[".."] * 3, "data", "autoencoder", "housing_price_index.jsonl"),
            orient="records",
            lines=True,
        )

        # change column name from Gdynia* to Gdynia
        price_index.rename(columns={"Gdynia*": "Gdynia"}, inplace=True)

        price_index["period_start"] = (
            pd.to_datetime(price_index["period_start"]).dt.tz_localize("UTC").dt.tz_convert("Europe/Warsaw")
        )
        price_index["period_end"] = (
            pd.to_datetime(price_index["period_end"]).dt.tz_localize("UTC").dt.tz_convert("Europe/Warsaw")
        )

        logger.debug("Normalizing price_m2")
        df[["normalize_price_m2", "normalize_price_m2_type"]] = df.swifter.apply(
            _normalize_price_m2,
            axis=1,
            result_type="expand",
            price_index=price_index,
        )

        # calculate normalize_price out of normalize_price_m2 and area
        df["normalize_price"] = df["normalize_price_m2"] * df["area"].astype(float)

    return df


def _encode_to_mhot(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding to mhot")
    df = encode_to_mhot(df, "type_id", "all_type_id")
    df = encode_to_mhot(df, "facilities", "all_facilities")
    df = encode_to_mhot(df, "natural_sites", "all_natural_sites")
    df = encode_to_mhot(df, "holiday_location", "all_holiday_location")
    df = encode_to_mhot(df, "quarters", "all_quarters")
    df = encode_to_mhot(df, "flat_type", "all_flat_type")
    df = encode_to_mhot(df, "kitchen_type", "all_kitchen_type")
    df = encode_to_mhot(df, "house_type", "all_house_type")
    df = encode_to_mhot(df, "additional_area_type", "all_additional_area_type")

    return df


async def _get_all_pois(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Getting all pois")
    # create a new data frame with columns offer_id, lat, lon
    coord_df = df[["offer_id", "lat", "lon"]].copy()
    # make unique
    coord_df.drop_duplicates(subset=["offer_id"], inplace=True)
    # sort by lat and lon
    coord_df.sort_values(by=["lat", "lon"], inplace=True)

    # rename offer_id to id (get_all_pois expects id column)
    coord_df.rename(columns={"offer_id": "id"}, inplace=True)

    # get all pois for each offer
    logger.debug("Fetching pois from numlabs")
    pois_df = await get_all_pois(coord_df, 200)

    # join offer_pois with dff
    df = pd.merge(df, pois_df, left_on="offer_id", right_on="id", how="inner")
    df.rename(columns={"pois": "raw_pois"}, inplace=True)

    for poi_type in set(AGG_POI.values()):
        df[f"{poi_type.lower()}_pois_count"] = df["raw_pois"].apply(lambda x: x[poi_type]["count"])
        df[f"{poi_type.lower()}_pois_dist"] = df["raw_pois"].apply(lambda x: x[poi_type]["dist"])

    return df


def _convert_decimal_to_float(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["area", "additional_area_area"]
    for column in columns:
        logger.debug(f"Converting {column} to float")
        df[column] = df[column].astype(float)
    return df


def _round_coords(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Rounding coordinates and filtering out records without lat/lon")
    filtered_df = df.dropna(subset=["lat", "lon"])
    if filtered_df.shape[0] < df.shape[0]:
        logger.warning(f"Filtered out {df.shape[0] - filtered_df.shape[0]} records without lat/lon.")
    del df
    filtered_df["lat"] = filtered_df["lat"].round(4)
    filtered_df["lon"] = filtered_df["lon"].round(4)
    return filtered_df


def _fill_known_nans(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        ("plot_area", 0.0),
    ]

    logger.info(f"Filling known NaNs for columns: {[column for column, _ in columns]}")
    for column, value in columns:
        if column in df.columns:
            df[column] = df[column].fillna(value)
        else:
            logger.warning(f"Column '{column}' not found in DataFrame. Adding with default value: {value}")
            df[column] = value
    return df


# create a type for list of callables or coroutines
Preprocessor = Callable[[pd.DataFrame], pd.DataFrame]
AsyncPreprocessor = Callable[[pd.DataFrame], Coroutine[Any, None, pd.DataFrame]]


async def preprocess_properties_data(df: pd.DataFrame, skip_normalization: bool = True) -> pd.DataFrame:
    """
    Preprocesses a pandas DataFrame containing property data using a sequence of synchronous and asynchronous
    preprocessing functions.

    The function applies a series of preprocessing steps, which may include filling missing values, rounding
    coordinates, converting decimal values to floats, grouping additional area types, normalizing prices
    (conditionally), encoding features, and extracting points of interest. Both synchronous and asynchronous
    preprocessors are supported.

    Args:
        df (pd.DataFrame): The input DataFrame containing property data to preprocess.
        skip_normalization (bool, optional): If True, skips the price normalization step. Defaults to True.

    Returns:
        pd.DataFrame: The preprocessed DataFrame after applying all preprocessing steps.

    """
    preprocessors: list[Preprocessor | AsyncPreprocessor] = [
        _fill_known_nans,
        _round_coords,
        _convert_decimal_to_float,
        _group_additional_area_type,
        partial(_normalize_price, skip=skip_normalization),
        _encode_to_mhot,
        _get_all_pois,
    ]

    for preprocessor in preprocessors:
        if asyncio.iscoroutinefunction(preprocessor):
            df = await preprocessor(df)
        else:
            df = cast(pd.DataFrame, preprocessor(df))

    return df
