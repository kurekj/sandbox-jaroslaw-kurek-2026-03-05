from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from src.v2.config import get_config


def calculate_user_scores_vectorized(
    user_history_embeddings: npt.NDArray[Any],
    user_history_timestamps: npt.NDArray[Any],
    properties_embeddings: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """
    Calculate similarity scores between properties and a user's history using vectorized operations.

    This function computes similarity scores between property embeddings and a user's weighted history embeddings.
    The process consists of several steps:

    1. Compute a similarity matrix between all properties and all user history items using dot product
    2. Apply time-based weighting to prioritize more recent user interactions
    3. Apply softmax with temperature to convert similarities to attention weights
    4. Compute weighted user history embeddings by combining history items according to their weights
        - IMPORTANT: A unique weighted user embedding is created for EACH property, as the attention
          mechanism creates property-specific weights. This means each property "attends" differently
          to the user's history items based on their relevance to that specific property.
    5. Calculate final scores as dot products between each property and its corresponding weighted history

    If both input embeddings are normalized (unit vectors), and softmax weights sum to 1,
    the resulting scores will be bounded within the [-1, 1] range, where:
    - Values close to 1 indicate high similarity to the user's weighted history
    - Values close to 0 indicate orthogonality (unrelated) to the user's weighted history
    - Values close to -1 indicate opposition to the user's weighted history

    Args:
        user_history_embeddings: Numpy array of shape (n_history_items, embedding_dim) containing
            embeddings of properties the user has interacted with.
        user_history_timestamps: Numpy array of shape (n_history_items,) containing timestamps of
            when the user interacted with each property.
        properties_embeddings: Numpy array of shape (n_properties, embedding_dim) containing
            embeddings of properties to score.

    Returns:
        Numpy array of shape (n_properties,) containing similarity scores for each property
    """
    config = get_config().user_score
    # Calculate all similarities at once (properties x history_items)
    similarity_matrix = np.dot(properties_embeddings, user_history_embeddings.T)

    # Apply timestamp weighting
    if len(user_history_timestamps) > 0:
        max_time = np.max(user_history_timestamps)
        min_time = np.min(user_history_timestamps)
        if max_time > min_time:
            time_weights = (user_history_timestamps - min_time) / (max_time - min_time)
        else:
            time_weights = np.ones_like(user_history_timestamps)

        # Broadcasting time weights across all properties
        combined_scores = (1 - config.time_decay_factor) * similarity_matrix + config.time_decay_factor * time_weights
    else:
        combined_scores = similarity_matrix

    # Apply softmax with temperature to each property's scores
    weights = np.exp(combined_scores / config.temperature) / np.sum(
        np.exp(combined_scores / config.temperature),
        axis=1,
        keepdims=True,
    )

    # Compute weighted embeddings for each property
    weighted_embeddings = np.matmul(weights, user_history_embeddings)

    # Calculate final scores (dot product of each property with its corresponding weighted embedding)
    # NOTE: This performs element-wise multiplication followed by sum along axis=1, which is equivalent to
    # computing the dot product for each pair of corresponding rows in the arrays. We use this approach
    # instead of np.dot because we need row-wise dot products rather than matrix multiplication.
    scores = np.sum(weighted_embeddings * properties_embeddings, axis=1)

    return cast(npt.NDArray[Any], scores)


def get_user_history(
    user_id: int,
    leads_df: pd.DataFrame,
    leads_properties_df: pd.DataFrame,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], pd.DataFrame]:
    """
    Build the user's history embeddings and timestamps. If `leads_df` contains an
    ``event_weight`` column (e.g. from Tinder interactions), each embedding is
    multiplied by the corresponding weight. Otherwise, the embedding is taken
    as-is (weight = 1.0).

    Args:
        user_id: A user identifier (algolytics_uuid).
        leads_df: DataFrame with at least columns ``algolytics_uuid``, ``property_id``,
            ``create_date`` and optionally ``event_weight``.
        leads_properties_df: DataFrame containing property embeddings keyed by ``property_id``.

    Returns:
        A tuple: (user_history_embeddings, user_history_timestamps, user_history_df)
    """
    # assert embeddings are not null/nan
    assert leads_properties_df["embeddings"].notnull().all(), "Embeddings contain null values"

    properties_ids = leads_df[leads_df["algolytics_uuid"] == user_id]["property_id"].unique().tolist()
    history_df = leads_properties_df[leads_properties_df["property_id"].isin(properties_ids)]
    user_history_df = leads_df[leads_df["algolytics_uuid"] == user_id].merge(
        history_df[["property_id", "embeddings"]],
        on="property_id",
        how="left",
    )
    # drop rows where embedding is null
    user_history_df = user_history_df[user_history_df["embeddings"].notnull()]
    user_history_df = user_history_df.sort_values(by=["create_date"], ascending=True)
    user_history_df = user_history_df.reset_index(drop=True)

    # Apply event weight if present; otherwise weight = 1.0
    if "event_weight" in user_history_df.columns:
        weights = user_history_df["event_weight"].astype(float).to_numpy()
        # Convert list-of-lists to a 2D array and scale each row by its weight
        embeddings_matrix = np.vstack(user_history_df["embeddings"].to_numpy())
        user_history_embeddings = embeddings_matrix * weights[:, None]
    else:
        user_history_embeddings = np.vstack(user_history_df["embeddings"].to_numpy())

    # NOTE: convert to seconds (drop tz)
    user_history_timestamps = np.array(user_history_df["create_date"].astype(np.int64) // 10**9)
    return user_history_embeddings, user_history_timestamps, user_history_df


def calculate_all_users_scores(
    leads_df: pd.DataFrame,
    leads_properties_df: pd.DataFrame,
    properties_embeddings: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    For each user, compute similarity scores for all properties by attending over their
    weighted history. Returns a matrix of shape (n_users x n_properties) and a list
    of user identifiers (algolytics_uuid) in the same order.
    """
    all_scores = []
    user_ids = leads_df["algolytics_uuid"].unique().tolist()
    for user_id in tqdm(user_ids, desc="Calculating user scores"):
        user_history_embeddings, user_history_timestamps, _ = get_user_history(
            user_id=user_id,
            leads_df=leads_df,
            leads_properties_df=leads_properties_df,
        )
        if len(user_history_embeddings) == 0:
            # fill with nans
            scores = np.full(len(properties_embeddings), np.nan)
        else:
            scores = calculate_user_scores_vectorized(
                user_history_embeddings=user_history_embeddings,
                user_history_timestamps=user_history_timestamps,
                properties_embeddings=properties_embeddings,
            )
        all_scores.append(scores)
    return np.array(all_scores), user_ids
