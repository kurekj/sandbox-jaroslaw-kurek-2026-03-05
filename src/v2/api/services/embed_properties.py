from functools import lru_cache

import pandas as pd
from loguru import logger

from src.v2.autoencoder.feature_columns_const import EXPECTED_FEATURE_COLUMNS
from src.v2.autoencoder.feature_specs import get_feature_columns
from src.v2.autoencoder.properties_embedding_model import PropertiesEmbeddingModel
from src.v2.config import get_config


@lru_cache
def _load_model(model_path: str) -> PropertiesEmbeddingModel:
    """Load the properties embedding model from the config."""
    logger.debug(f"Loading model from {model_path}")
    return PropertiesEmbeddingModel.load_from_checkpoint(model_path)


def sanitize_feature_columns(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Sanitize feature columns by converting object dtypes to numeric where possible.

    Args:
        data: The DataFrame to sanitize
        feature_columns: List of feature columns to check

    Returns:
        A sanitized copy of the DataFrame with object columns converted to numeric
    """
    #  Check for object dtypes in the feature columns
    object_cols = [col for col in feature_columns if data[col].dtype == "object"]

    if not object_cols:
        return data.copy()

    logger.warning(f"Found {len(object_cols)} columns with object dtype: {object_cols}")

    # Make a copy to avoid modifying the original
    sanitized_data = data.copy()

    # Convert object columns to numeric where possible
    for col in object_cols:
        try:
            sanitized_data[col] = pd.to_numeric(sanitized_data[col], errors="coerce")
            logger.debug(f"Converted column {col} to numeric with possible NaN values")
        except Exception as e:
            logger.warning(f"Failed to convert column {col}: {e}! Filling with NaN.")
            # Fill with NaN if conversion fails
            sanitized_data[col] = pd.NA
            # set type to float64
            sanitized_data[col] = sanitized_data[col].astype("float64")

    return sanitized_data


def _align_feature_columns(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Align input features to expected model schema.

    - Drops extra columns not listed in EXPECTED_FEATURE_COLUMNS
    - Adds missing expected columns filled with zeros
    - Returns columns ordered exactly as in EXPECTED_FEATURE_COLUMNS
    - Logs all actions (extras dropped, missing added, final shape)
    """
    # Log full list of provided feature columns
    logger.info("Provided feature columns ({}): {}", len(feature_columns), ",".join(feature_columns))

    expected_order = EXPECTED_FEATURE_COLUMNS
    provided_set = set(feature_columns)
    expected_set = set(expected_order)

    # Identify extras and missing
    extras = [col for col in feature_columns if col not in expected_set]
    missing = [col for col in expected_order if col not in provided_set]

    if extras:
        logger.warning("Dropping extra columns not expected ({}): {}", len(extras), ",".join(extras))

    if missing:
        logger.warning("Adding missing expected columns with zeros ({}): {}", len(missing), ",".join(missing))

    # Ensure missing columns exist in data, filled with zeros (float)
    if missing:
        data = data.copy()
        for col in missing:
            data[col] = 0.0

    # Final ordered and capped list
    final_columns = expected_order

    logger.info("Aligned feature columns to {} (dropped: {}, added: {})", len(final_columns), len(extras), len(missing))

    return data, final_columns


def embed_properties(data: pd.DataFrame, persistent_workers: bool = True) -> pd.DataFrame:
    if data.empty:
        data["embeddings"] = pd.NA
        return data

    feature_columns = get_feature_columns(data)[0]
    # Align to EXPECTED_FEATURE_COLUMNS and log changes
    data, feature_columns = _align_feature_columns(data, feature_columns)

    # Sanitize the data before embedding
    embed_data = sanitize_feature_columns(data, feature_columns)

    model = _load_model(get_config().properties_embedding_model.mlflow_artifact_path)

    embeddings = model.get_embeddings(embed_data[feature_columns])

    # add embeddings as a column to df
    data["embeddings"] = embeddings.tolist()

    return data
