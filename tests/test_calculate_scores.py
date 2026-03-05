import numpy as np
import pytest

from src.v2.api.services.calculate_scores import calculate_user_scores_vectorized
from src.v2.config import get_config


def test_calculate_user_scores_basic() -> None:
    """Test the basic functionality of calculate_user_scores_vectorized with simple inputs."""
    # Setup
    user_history_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # First history item points in x direction
            [0.0, 1.0, 0.0],  # Second history item points in y direction
        ]
    )
    user_history_timestamps = np.array([100, 200])  # Second interaction is more recent
    properties_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # Property 1 similar to first history item
            [0.0, 1.0, 0.0],  # Property 2 similar to second history item
            [0.5, 0.5, 0.0],  # Property 3 somewhat similar to both
        ]
    )

    # Execute
    scores = calculate_user_scores_vectorized(
        user_history_embeddings=user_history_embeddings,
        user_history_timestamps=user_history_timestamps,
        properties_embeddings=properties_embeddings,
    )

    # Assert
    assert scores.shape == (3,)
    # Property 2 should have higher score than property 1 due to recency weighting
    assert scores[1] > scores[0]
    # All scores should be in [-1, 1] range
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0)


def test_calculate_user_scores_without_timestamps() -> None:
    """Test that function works correctly when timestamps are not provided."""
    # Setup
    user_history_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    user_history_timestamps = np.array([])  # No timestamps provided
    properties_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    # Execute
    scores = calculate_user_scores_vectorized(
        user_history_embeddings=user_history_embeddings,
        user_history_timestamps=user_history_timestamps,
        properties_embeddings=properties_embeddings,
    )

    # Assert
    assert scores.shape == (2,)
    # Scores should be similar since no time decay is applied
    assert abs(scores[0] - scores[1]) < 1e-6


def test_calculate_user_scores_equal_timestamps() -> None:
    """Test that function works correctly when all timestamps are equal."""
    # Setup
    user_history_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    user_history_timestamps = np.array([100, 100])  # Equal timestamps
    properties_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    # Execute
    scores = calculate_user_scores_vectorized(
        user_history_embeddings=user_history_embeddings,
        user_history_timestamps=user_history_timestamps,
        properties_embeddings=properties_embeddings,
    )

    # Assert
    assert scores.shape == (2,)
    # Scores should be roughly equal despite different embeddings since time weights are equal
    # and both properties exactly match one of the history items
    assert abs(scores[0] - scores[1]) < 1e-6


def test_calculate_user_scores_with_config_values() -> None:
    """Test that function correctly uses configuration values."""
    # Setup - Create embeddings similar to previous tests
    user_history_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    user_history_timestamps = np.array([100, 200])
    properties_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    config = get_config()

    # Store original values to restore later
    original_time_decay = config.user_score.time_decay_factor
    original_temperature = config.user_score.temperature

    try:
        # Test with different time decay factor
        config.user_score.time_decay_factor = 0.0  # No time decay
        scores_no_decay = calculate_user_scores_vectorized(
            user_history_embeddings=user_history_embeddings,
            user_history_timestamps=user_history_timestamps,
            properties_embeddings=properties_embeddings,
        )

        # Both properties should have roughly equal scores now
        assert abs(scores_no_decay[0] - scores_no_decay[1]) < 1e-6

        # Test with higher time decay factor
        config.user_score.time_decay_factor = 0.9  # High time decay
        scores_high_decay = calculate_user_scores_vectorized(
            user_history_embeddings=user_history_embeddings,
            user_history_timestamps=user_history_timestamps,
            properties_embeddings=properties_embeddings,
        )

        # Property 2 should have much higher score due to recency
        assert scores_high_decay[1] > scores_high_decay[0]

        # Test with different temperature
        config.user_score.time_decay_factor = original_time_decay
        config.user_score.temperature = 0.1  # Lower temperature = sharper differences
        scores_low_temp = calculate_user_scores_vectorized(
            user_history_embeddings=user_history_embeddings,
            user_history_timestamps=user_history_timestamps,
            properties_embeddings=properties_embeddings,
        )

        config.user_score.temperature = 10.0  # Higher temperature = smoother differences
        scores_high_temp = calculate_user_scores_vectorized(
            user_history_embeddings=user_history_embeddings,
            user_history_timestamps=user_history_timestamps,
            properties_embeddings=properties_embeddings,
        )

        # Score difference should be more pronounced with lower temperature
        score_diff_low_temp = abs(scores_low_temp[1] - scores_low_temp[0])
        score_diff_high_temp = abs(scores_high_temp[1] - scores_high_temp[0])
        assert score_diff_low_temp > score_diff_high_temp

    finally:
        # Restore original config values
        config.user_score.time_decay_factor = original_time_decay
        config.user_score.temperature = original_temperature


def test_calculate_user_scores_orthogonal_embeddings() -> None:
    """Test with orthogonal embeddings (should give scores near zero)."""
    # Setup
    user_history_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # x direction
        ]
    )
    user_history_timestamps = np.array([100])
    properties_embeddings = np.array(
        [
            [0.0, 1.0, 0.0],  # y direction - orthogonal to history
        ]
    )

    # Execute
    scores = calculate_user_scores_vectorized(
        user_history_embeddings=user_history_embeddings,
        user_history_timestamps=user_history_timestamps,
        properties_embeddings=properties_embeddings,
    )

    # Assert
    assert scores.shape == (1,)
    # Score should be close to 0 for orthogonal vectors
    assert abs(scores[0]) < 1e-6


def test_calculate_user_scores_opposite_embeddings() -> None:
    """Test with opposite direction embeddings (should give negative scores)."""
    # Setup
    user_history_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # x direction
        ]
    )
    user_history_timestamps = np.array([100])
    properties_embeddings = np.array(
        [
            [-1.0, 0.0, 0.0],  # -x direction - opposite to history
        ]
    )

    # Execute
    scores = calculate_user_scores_vectorized(
        user_history_embeddings=user_history_embeddings,
        user_history_timestamps=user_history_timestamps,
        properties_embeddings=properties_embeddings,
    )

    # Assert
    assert scores.shape == (1,)
    # Score should be negative for opposite vectors
    assert scores[0] < 0


def test_calculate_user_scores_empty_history() -> None:
    """Test behavior when user history is empty."""
    # Setup
    user_history_embeddings = np.array([])  # Empty array
    if len(user_history_embeddings) > 0:  # This is to avoid reshape errors
        user_history_embeddings = user_history_embeddings.reshape(0, 3)
    user_history_timestamps = np.array([])
    properties_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    # We expect this to fail gracefully since dot product can't be computed
    # with empty arrays
    with pytest.raises(Exception):
        calculate_user_scores_vectorized(
            user_history_embeddings=user_history_embeddings,
            user_history_timestamps=user_history_timestamps,
            properties_embeddings=properties_embeddings,
        )
