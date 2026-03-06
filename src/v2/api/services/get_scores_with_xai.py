"""
This module extends the base scoring logic to produce an explainable AI (XAI)
report alongside the raw recommendation scores.  After computing the scores for
user/property pairs, a simple surrogate model is trained to approximate the
relationship between the model inputs (property features) and the predicted
scores.  SHAP values are then calculated for each example to quantify the
contribution of every feature to the final score.  The resulting
explanations are returned as a list of dictionaries that can be easily
serialized to JSON and delivered back to the end user.

The surrogate model approach is used because the real recommender operates
on dense embeddings rather than the raw input features.  Since the
mapping between the original feature space and these embeddings is
non‑linear and not easily accessible, the code below fits a lightweight
RandomForestRegressor on the observed scores using the same input
features.  While this does not exactly replicate the latent model, it
provides an interpretable approximation that highlights which attributes
drive the recommendation outcome.

The list of feature names (`EXPECTED_FEATURE_COLUMNS`) is derived from
the accompanying LaTeX report included with this project.  If the set of
features used in training changes in the future, update this list
accordingly.
"""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.v2.api.services.get_scores import get_scores_df  # type: ignore
from src.v2.api.services.get_scores import _load_data_cached  # type: ignore
from src.v2.utils.cache_utils import get_or_create_cached_df  # type: ignore

# Third‑party imports used for the surrogate model and SHAP explanations
from sklearn.ensemble import RandomForestRegressor
import shap  # type: ignore

# ---------------------------------------------------------------------------
# Feature specification
#
# This list enumerates all input features used to train the underlying
# autoencoder and recommendation model.  It is reproduced here verbatim
# from the accompanying report.  Any future additions or removals of
# features should be reflected here as well.

EXPECTED_FEATURE_COLUMNS: List[str] = [
    # Categorical (multi‑hot) facilities
    *(f"facilities_{i}" for i in range(1, 36)),
    # Categorical (multi‑hot) natural sites
    *(f"natural_sites_{i}" for i in range(1, 10)),
    # Categorical (multi‑hot) kitchen type
    *(f"kitchen_type_{i}" for i in range(1, 3)),
    # Categorical (multi‑hot) additional area type
    *(f"additional_area_type_{i}" for i in range(1, 5)),
    # Categorical (multi‑hot) quarters
    *(f"quarters_{i}" for i in range(1, 5)),
    # Categorical (multi‑hot) type_id
    *(f"type_id_{i}" for i in range(1, 4)),
    # Categorical (multi‑hot) house type
    *(f"house_type_{i}" for i in range(1, 5)),
    # Categorical (multi‑hot) flat type
    *(f"flat_type_{i}" for i in range(1, 4)),
    # Numerical (base) features
    "buildings",
    "properties",
    "lon",
    "lat",
    "area",
    "normalize_price_m2",
    "normalize_price",
    "rooms",
    "floor",
    "bathrooms",
    "additional_area_area",
    # Numerical (POI) features
    "all_pois_count",
    "all_pois_dist",
    "shops_pois_count",
    "shops_pois_dist",
    "food_pois_count",
    "food_pois_dist",
    "education_pois_count",
    "education_pois_dist",
    "health_pois_count",
    "health_pois_dist",
    "entertainment_pois_count",
    "entertainment_pois_dist",
    "sport_pois_count",
    "sport_pois_dist",
    "transport_pois_count",
    "transport_pois_dist",
]


async def _get_property_features(property_ids: List[int]) -> pd.DataFrame:
    """Load and return the feature matrix for the given list of property IDs.

    Parameters
    ----------
    property_ids : list of int
        Identifiers of properties for which to fetch raw feature data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per property with all expected
        features.  Missing columns are added with NaN values.
    """
    # Load the raw data using the existing caching mechanism.  Skip
    # normalization here; it has already been applied during model training.
    raw_df = await _load_data_cached(ids=property_ids, overwrite=False)
    # Ensure that all expected feature columns are present.  Any missing
    # columns are filled with zeros (appropriate for one‑hot encoded
    # features) or NaN for continuous variables, depending on the data type.
    for col in EXPECTED_FEATURE_COLUMNS:
        if col not in raw_df.columns:
            logger.debug(f"Adding missing column '{col}' with default NaN values")
            raw_df[col] = np.nan
    # Return only the necessary columns plus property_id for merging
    feature_df = raw_df[["property_id", *EXPECTED_FEATURE_COLUMNS]].copy()
    return feature_df


async def get_scores_with_explanations(
    df: pd.DataFrame,
    n_estimators: int = 50,
    random_state: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """Compute recommendation scores and generate SHAP explanations.

    This function first calls the existing ``get_scores_df`` to obtain
    recommendation scores for each user/property pair in ``df``.  It then
    constructs a surrogate model (RandomForestRegressor) trained on the
    property feature matrix to predict the scores.  Finally, SHAP values
    are computed for each input row to describe how each feature
    contributes to the score.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with two columns: ``user_id`` and ``property_id``.  It
        specifies which user/property pairs should be scored and
        explained.

    n_estimators : int, optional
        Number of trees in the RandomForestRegressor used as the
        surrogate model.  Adjust this parameter to balance
        explanation fidelity against performance.  Defaults to 50.

    random_state : int or None, optional
        Random seed for the surrogate model.  Providing a seed
        guarantees reproducibility of the surrogate model and SHAP
        values.  Defaults to 42.

    Returns
    -------
    list of dict
        Each dictionary contains the ``user_id``, ``property_id``, raw
        ``score`` produced by the recommender, and a nested mapping
        ``feature_contributions`` that associates each feature name with
        its SHAP value.  These dictionaries are easily serializable to
        JSON.
    """
    # Compute the raw scores using the existing get_scores_df function
    scores_df = await get_scores_df(df)
    # Filter out any rows with missing scores (NaN) – these cannot be
    # explained reliably
    valid_scores_df = scores_df[scores_df["score"].notnull()].copy()
    if valid_scores_df.empty:
        logger.warning("No valid scores were computed; returning empty explanation list")
        return []

    # Load the feature matrix for all properties present in the request
    property_ids = valid_scores_df["property_id"].astype(int).unique().tolist()
    feature_df = await _get_property_features(property_ids)
    # Merge the features into the scores dataframe based on property_id
    merged_df = valid_scores_df.merge(feature_df, on="property_id", how="left")

    # Extract the design matrix X and target vector y for the surrogate model
    X = merged_df[EXPECTED_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = merged_df["score"].to_numpy(dtype=float)

    # Handle any remaining NaNs by imputing zeros.  Since many features are
    # multi‑hot encoded, zero is a reasonable default for missing data.
    X = np.nan_to_num(X, nan=0.0)

    # Train a simple RandomForestRegressor to approximate the score
    surrogate = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    surrogate.fit(X, y)

    # Create a SHAP explainer for the surrogate model.  The TreeExplainer
    # leverages the structure of the forest to compute exact or approximate
    # SHAP values efficiently.
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X)

    # Construct the explanation output.  Iterate over each row and map
    # feature names to the corresponding SHAP value.  Cast all floats to
    # native Python types to ensure JSON serialization works correctly.
    explanations: List[Dict[str, Any]] = []
    for idx, row in merged_df.iterrows():
        shap_row = shap_values[idx]
        contributions: Dict[str, float] = {
            feature: float(shap_row[col_idx]) for col_idx, feature in enumerate(EXPECTED_FEATURE_COLUMNS)
        }
        explanations.append(
            {
                "user_id": row["user_id"],
                "property_id": int(row["property_id"]),
                "score": float(row["score"]),
                "feature_contributions": contributions,
            }
        )
    return explanations


# Convenience function to dump explanations to JSON
async def get_scores_with_explanations_json(
    df: pd.DataFrame,
    n_estimators: int = 50,
    random_state: Optional[int] = 42,
) -> str:
    """Wrapper around ``get_scores_with_explanations`` that returns a JSON string.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of user/property pairs to score.
    n_estimators : int, optional
        Number of trees in the surrogate RandomForestRegressor.  Defaults
        to 50.
    random_state : int or None, optional
        Random seed for reproducibility.  Defaults to 42.

    Returns
    -------
    str
        A JSON‑formatted string containing the list of explanation dictionaries.
    """
    import json  # local import to avoid unused import in environments where JSON is not needed
    results = await get_scores_with_explanations(df, n_estimators=n_estimators, random_state=random_state)
    return json.dumps(results, ensure_ascii=False)