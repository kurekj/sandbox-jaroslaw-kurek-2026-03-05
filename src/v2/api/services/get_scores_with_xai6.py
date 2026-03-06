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
import json
import os
import sys
import uuid
from datetime import datetime
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

# ---------------------------------------------------------------------------
# Helper functions
#
# The functions below provide additional functionality to load a feature
# mapping from an external JSON file and to convert arbitrary user_id
# values into UUID objects.  These helpers are used throughout the script
# to enhance interpretability and ensure type correctness when querying
# the database.

def load_feature_mapping(path: str = "feature_mapping.json") -> Dict[str, Any]:
    """Load a mapping of raw feature names to human‑readable names.

    The mapping file should be a JSON object where keys correspond to
    feature names (e.g. ``facilities_1``) and values are human‑readable
    descriptions (e.g. ``"własny akwen"``).  If the file cannot be read
    or parsed, an empty dictionary is returned and a warning is logged.

    Parameters
    ----------
    path : str, optional
        Path to the JSON mapping file.  Defaults to ``"feature_mapping.json"``.

    Returns
    -------
    dict
        A dictionary mapping feature names to descriptive strings.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            logger.warning(f"Mapping file {path} does not contain a JSON object; ignoring")
            return {}
        return mapping
    except Exception as exc:
        logger.warning(f"Could not load feature mapping from {path}: {exc}")
        return {}


def coerce_uuid(value: Any) -> str:
    """Normalize user identifiers to UUID strings without altering valid UUIDs.

    The recommendation database stores user IDs as strings (UUID format),
    so passing literal ``uuid.UUID`` objects to the query can prevent
    matches.  This helper leaves valid UUID strings untouched, converts
    integers into deterministic UUID strings, and falls back to a
    namespace-based UUID string for any other input.  The return type
    is always a string to ensure compatibility with downstream SQL
    queries.

    Parameters
    ----------
    value : Any
        The raw ``user_id`` value provided by the caller.

    Returns
    -------
    str
        A string representation of a UUID.  If ``value`` was already a
        valid UUID string, it is returned unchanged (normalized to
        canonical UUID string format).  Integers are converted to
        deterministic UUIDs, and all other types are hashed into
        namespace UUIDs.
    """
    # If the input is already a uuid.UUID, return its canonical string
    if isinstance(value, uuid.UUID):
        return str(value)
    # If the input is a string, try to normalize it to a canonical UUID
    try:
        # uuid.UUID will validate the string and return a UUID object
        uid_obj = uuid.UUID(str(value))
        return str(uid_obj)
    except Exception:
        pass
    # If the input is an integer, deterministically convert it to a UUID
    try:
        uid_obj = uuid.UUID(int=int(value))
        return str(uid_obj)
    except Exception:
        pass
    # As a last resort, generate a namespace-based UUID from the string
    return str(uuid.uuid5(uuid.NAMESPACE_OID, str(value)))

# ---------------------------------------------------------------------------
# Feature description mapping
#
# This dictionary maps each feature name to a human‑readable description of
# what the feature represents.  It is used when generating JSON output so
# that end users can understand what each variable corresponds to.  If the
# feature set changes in the future, update the mapping accordingly.
FEATURE_MAPPING: Dict[str, str] = {}
for feat in EXPECTED_FEATURE_COLUMNS:
    if feat.startswith("facilities_"):
        FEATURE_MAPPING[feat] = "Categorical (multi‑hot): facilities"
    elif feat.startswith("natural_sites_"):
        FEATURE_MAPPING[feat] = "Categorical (multi‑hot): natural sites"
    elif feat.startswith("kitchen_type_"):
        FEATURE_MAPPING[feat] = "Categorical (multi‑hot): kitchen type"
    elif feat.startswith("additional_area_type_"):
        FEATURE_MAPPING[feat] = "Categorical (multi‑hot): additional area type"
    elif feat.startswith("quarters_"):
        FEATURE_MAPPING[feat] = "Categorical (multi‑hot): quarters"
    elif feat.startswith("type_id_"):
        FEATURE_MAPPING[feat] = "Categorical (multi‑hot): type ID"
    elif feat.startswith("house_type_"):
        FEATURE_MAPPING[feat] = "Categorical (multi‑hot): house type"
    elif feat.startswith("flat_type_"):
        FEATURE_MAPPING[feat] = "Categorical (multi‑hot): flat type"
    elif feat in {
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
    }:
        # Base numerical features
        FEATURE_MAPPING[feat] = "Numerical (base)"
    elif feat.endswith("_count") or feat.endswith("_dist"):
        # POI numerical features
        # Extract the category prefix before _count or _dist
        prefix = feat.rsplit("_", 1)[0]  # e.g., all_pois or shops_pois
        category_name = prefix.replace("_pois", " POIs")
        if feat.endswith("_count"):
            FEATURE_MAPPING[feat] = f"Numerical (POI): {category_name} – count"
        else:
            FEATURE_MAPPING[feat] = f"Numerical (POI): {category_name} – distance"
    else:
        # Fallback description if none of the above conditions match
        FEATURE_MAPPING[feat] = "Unknown feature type"

# After constructing the base mapping, attempt to override entries with
# values from an external mapping file.  This allows us to provide
# domain‑specific descriptions for each feature where available.
_external_mapping = load_feature_mapping()
if _external_mapping:
    FEATURE_MAPPING.update({k: _external_mapping.get(k, v) for k, v in FEATURE_MAPPING.items()})


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
    # Coerce the user identifiers to UUID objects to satisfy database queries.
    # Doing this at the start ensures that any integer or string IDs are
    # converted into a consistent format.  Without this step, passing
    # integers into the SQL layer would result in a type mismatch when
    # compared against ``uuid`` columns.
    df = df.copy()
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].apply(coerce_uuid)

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
        # Build a dictionary of SHAP contributions keyed by raw feature name
        contributions: Dict[str, float] = {
            feature: float(shap_row[col_idx])
            for col_idx, feature in enumerate(EXPECTED_FEATURE_COLUMNS)
        }
        # Create a sorted list of contributions with both raw and mapped names
        sorted_contrib = sorted(
            [
                {
                    "feature": feature,
                    "mapped_feature": FEATURE_MAPPING.get(feature, feature),
                    "contribution": contributions[feature],
                }
                for feature in EXPECTED_FEATURE_COLUMNS
            ],
            key=lambda x: abs(x["contribution"]),
            reverse=True,
        )
        explanations.append(
            {
                # Convert UUID back to string for JSON serialization
                "user_id": str(row["user_id"]),
                "property_id": int(row["property_id"]),
                "score": float(row["score"]),
                "feature_contributions": contributions,
                "sorted_contributions": sorted_contrib,
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
    return json.dumps(results, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Command‑line interface
#
# When this module is executed as a script (``python get_scores_with_xai.py``),
# it can load a CSV file containing ``user_id`` and ``property_id`` columns,
# compute recommendation scores and SHAP explanations, and save the
# resulting JSON to a specified file.  This makes the module convenient
# to test outside of the main API service.

if __name__ == "__main__":
    """
    When run directly, this script executes the end‑to‑end XAI pipeline on a
    predefined set of user/property pairs, logs each step, prints the
    results to the console, and saves a JSON report to disk.
    
    The input data is embedded directly below for demonstration purposes.
    Update the ``df`` definition to point at the desired user and property
    identifiers in your environment.  No command‑line arguments are used; all
    configuration lives inside this file.
    """
    import json
    import time
    import sys
    import asyncio

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


    # Configure logging: log to both console (default) and a file.  Create a
    # dedicated output directory based on the current date and time.  All
    # generated files will be placed inside this directory.
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"XAI_{current_timestamp}"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as exc:
        logger.warning(f"Could not create output directory '{output_dir}': {exc}")

    log_file_name = os.path.join(output_dir, "xai_report.log")
    try:
        # Add a log file sink
        logger.add(log_file_name)
    except Exception:
        # If adding the log file fails, continue without file logging
        logger.warning(f"Could not add log file sink to {log_file_name}")

    # Define a sample DataFrame of user_id and property_id pairs.  Replace
    # these values with your actual data as needed.
    df = pd.DataFrame(
        {
            "user_id": [
                "0f51390e-63aa-4791-845e-97ef6dd92bfe",
                "0f51390e-63aa-4791-845e-97ef6dd92bfe",
                "c0dcface-0489-45cc-8a45-bc2ef2d243de",
                "c0dcface-0489-45cc-8a45-bc2ef2d243de",
                "c0dcface-0489-45cc-8a45-bc2ef2d243de",
            ],
            "property_id": [
                671261,
                1114792,
                988320,
                1221875,
                17407,
            ],
        }
    )

    # Logging start of the process
    logger.info("Starting XAI scoring and explanation pipeline")
    # Compute explanations using default hyperparameters
    explanations = asyncio.run(
        get_scores_with_explanations(
            df,
            n_estimators=50,
            random_state=42,
        )
    )

    # Display and log the explanations
    for entry in explanations:
        logger.info(
            f"User {entry['user_id']} / Property {entry['property_id']} -> Score {entry['score']:.4f}"
        )
        # Print a succinct view of the top contributing features based on the
        # sorted contributions list.  Show both the raw and mapped feature names.
        top_features = entry.get("sorted_contributions", [])[:5]
        print(
            f"\nUser: {entry['user_id']}\nProperty: {entry['property_id']}\nScore: {entry['score']:.4f}"
        )
        print("Top contributing features (feature: mapped_feature: SHAP value):")
        for feat_obj in top_features:
            print(
                f"  {feat_obj['feature']}: {feat_obj['mapped_feature']} => {feat_obj['contribution']:.4f}"
            )
        print("---")

    # Print the feature mapping so users understand what each variable represents
    print("\nFeature mapping (variable: description):")
    for feat_name, feat_desc in FEATURE_MAPPING.items():
        print(f"  {feat_name}: {feat_desc}")
    print("---")

    # Save the complete explanations list along with feature mapping to a JSON file
    json_file_name = os.path.join(output_dir, "xai_explanations.json")
    # Construct the result object with mapping and explanations
    output_data = {
        "feature_mapping": FEATURE_MAPPING,
        "explanations": explanations,
    }
    # Display what will be saved and where
    logger.info(
        f"Preparing to save {len(explanations)} explanation entries and feature mapping to {json_file_name}"
    )
    print(
        f"\nSaving JSON report with {len(explanations)} entries and feature mapping to '{json_file_name}'"
    )
    # Write the result to the JSON file with pretty formatting
    try:
        with open(json_file_name, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Saved JSON report (including feature mapping) to {json_file_name}"
        )
        print(
            f"Full JSON report (including feature mapping) saved to {json_file_name}"
        )
    except Exception as exc:
        logger.error(f"Failed to save JSON report to {json_file_name}: {exc}")