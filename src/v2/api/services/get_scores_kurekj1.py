"""
Compute recommendation scores without using any cache layers.

This module provides functions analogous to those in ``get_scores.py``, but
it bypasses all Redis/Sentinel caches.  All leads and property data are
loaded directly from the underlying databases on every call.  This can be
useful for debugging or for environments where caching is disabled.
"""

from __future__ import annotations

import asyncio
import sys
import time
from functools import lru_cache
from typing import Optional, Tuple, Any

import numpy as np
import pandas as pd
from loguru import logger

from src.v2.api.services.calculate_scores import calculate_all_users_scores
from src.v2.api.services.embed_properties import embed_properties
from src.v2.api.services.load_leads_df import load_leads_data_db
from src.v2.autoencoder.preprocess_data import load_current_properties_data, preprocess_properties_data

from src.v2.api.services.business_reranking import RerankParams, rerank_dataframe

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def _load_data(ids: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Load and preprocess property data directly from the database.

    If ``ids`` is None, all visible properties are loaded.  Otherwise, only
    the specified property IDs are fetched.  No caching layer is used.

    Args:
        ids: Optional list of property IDs to fetch.

    Returns:
        A pandas DataFrame containing preprocessed property data with
        feature columns.
    """
    df = await load_current_properties_data(ids)
    df = await preprocess_properties_data(df)
    return df


async def get_scores_df_no_cache(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate recommendation scores for a DataFrame of (user_id, property_id)
    pairs without using any caching.

    This function loads leads and property data directly from the
    underlying databases for each invocation.  It embeds properties,
    computes similarity scores via ``calculate_all_users_scores`` and
    returns the resulting DataFrame with an added ``score`` column.

    Args:
        df: A DataFrame with columns ``user_id`` and ``property_id``.

    Returns:
        The input DataFrame with an additional ``score`` column containing
        the computed recommendation scores.  NaN scores are inserted
        where data was missing.
    """
    logger.info("Calculating scores for user_id <-> property_id pairs (no cache)")

    # Load leads directly from the database for the provided users
    user_ids = df["user_id"].unique().tolist()
    leads_df = await load_leads_data_db(user_ids=user_ids)
    logger.info(f"Loaded {len(leads_df)} leads for {len(user_ids)} users")

    # Early exit if no leads were found
    if leads_df.empty:
        logger.warning("Leads dataframe is empty! Returning original df with NaN scores.")
        df = df.copy()
        df["score"] = np.nan
        return df

    # Load and preprocess property data for leads and scoring properties
    leads_property_ids = leads_df["property_id"].dropna().unique().tolist()
    scoring_property_ids = df["property_id"].dropna().unique().tolist()

    leads_properties_df = await _load_data(ids=leads_property_ids)
    scoring_properties_df = await _load_data(ids=scoring_property_ids)
    logger.info(
        f"Loaded {len(leads_properties_df)} leads properties and {len(scoring_properties_df)} scoring properties"
    )

    # If any of the property sets are empty, return NaN scores
    if leads_properties_df.empty or scoring_properties_df.empty:
        logger.warning(
            "Leads properties or scoring properties are empty! Returning original df with NaN scores."
        )
        df = df.copy()
        df["score"] = np.nan
        return df

    # Embed properties to obtain embeddings for scoring
    leads_properties_df = embed_properties(leads_properties_df)
    scoring_properties_df = embed_properties(scoring_properties_df)
    logger.info("Embedded leads properties and scoring properties")

    # Compute similarity scores for each user-property pair
    all_scores, user_ids_out = calculate_all_users_scores(
        leads_df=leads_df,
        leads_properties_df=leads_properties_df,
        properties_embeddings=np.array(scoring_properties_df["embeddings"].tolist()),
    )
    logger.info("Calculated scores")

    # Build a DataFrame from the scores matrix
    property_ids = scoring_properties_df["property_id"].tolist()
    scores_df = pd.DataFrame(all_scores, index=user_ids_out, columns=property_ids)
    scores_df = (
        scores_df.reset_index()
        .melt(id_vars="index", var_name="property_id", value_name="score")
        .rename(columns={"index": "user_id"})
    )

    # Merge the scores with the original pairs
    df = df.merge(scores_df, on=["user_id", "property_id"], how="left")
    return df


if __name__ == "__main__":  # pragma: no cover
    # Example usage for manual testing
    example_df = pd.DataFrame(
        {
            "user_id": [
                "9448824d-bdf0-4d96-ad53-2586b82b03b5",
                "9448824d-bdf0-4d96-ad53-2586b82b03b5",
                "9448824d-bdf0-4d96-ad53-2586b82b03b5",
                "9448824d-bdf0-4d96-ad53-2586b82b03b5"
            ],
            "property_id": [1139682, 1156871, 1202552, 1159886],
        }
    )

    #####################
    start_scoring = time.time()
    scores_df = asyncio.run(get_scores_df_no_cache(example_df))
    scoring_time = time.time() - start_scoring
    print(f"Score calculation time: {scoring_time:.2f} s")

    # Add the required business attributes to the DataFrame:
    # r   = profitability [0,1]
    # g   = plan-gap [0,1]
    # m   = fairness boost (e.g. +0.3 for small developers, 0 for medium, –0.2 for large)
    # v   = VIP flag (1 for premium, 0 otherwise)
    # contract_type = 'flat' or 'cpl'
    # cap_ratio    = current cap utilisation [0,1]
    scores_df["r"] = 0.5  # example value; replace with real data
    scores_df["g"] = 0.4  # example value
    scores_df["m"] = 0.1  # example value
    scores_df["v"] = 0  # example value
    scores_df["contract_type"] = "flat"
    scores_df["cap_ratio"] = 0.3

    # Define re‑ranking parameters
    params = RerankParams(
        gamma=1.2,  # emphasise high base scores
        mu=0.5,  # weight of fairness
        nu=0.8,  # weight of VIP
        rho=0.0,  # pacing weight
        delta=1.0,  # cap penalty weight
        lambda_=0.5  # diversity penalty weight
    )

    # Perform business re‑rank and measure time
    start_rerank = time.time()
    reranked_df = rerank_dataframe(scores_df, params, k=None, p_h=0.0)
    rerank_time = time.time() - start_rerank
    print(f"Business re‑ranking time: {rerank_time:.2f} s")

    # Sort results descending by base score and business score
    sorted_by_score = scores_df.sort_values("score", ascending=False)
    sorted_by_business = reranked_df.sort_values("business_score", ascending=False)

    # Display the top 10 rows (columns: property_id, score, business_score)
    print("\nTop 10 by base score (property_id, score, business_score):")
    print(sorted_by_score[["property_id", "score"]].head(10))

    print("\nTop 10 by business score (property_id, score, business_score):")
    print(sorted_by_business[["property_id", "score", "business_score"]].head(10))

    # Write to Excel with two sheets
    output_path = "recommender_results.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        sorted_by_score.to_excel(writer, sheet_name="SortedByScore", index=False)
        sorted_by_business.to_excel(writer, sheet_name="SortedByBusinessScore", index=False)

    print(f"Results saved to {output_path}")