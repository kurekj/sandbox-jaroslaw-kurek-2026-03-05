from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from src.v2.api.services.get_scores import get_scores_df


@pytest.fixture
def input_df() -> pd.DataFrame:
    """Test input dataframe with user_id and property_id pairs."""
    return pd.DataFrame(
        {
            "user_id": ["user1", "user1", "user2"],
            "property_id": [1001, 1002, 1003],
        }
    )


@pytest.fixture
def mock_leads_df() -> pd.DataFrame:
    """Mock leads dataframe with user history."""
    return pd.DataFrame(
        {
            "create_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "property_id": [2001, 2002, 2003],
            "algolytics_uuid": ["user1", "user1", "user2"],
        }
    )


@pytest.fixture
def mock_properties_df() -> pd.DataFrame:
    """Mock properties dataframe."""
    return pd.DataFrame(
        {
            "property_id": [1001, 1002, 1003, 2001, 2002, 2003],
            "city_name": ["City1", "City2", "City3", "City4", "City5", "City6"],
            "area": [50, 60, 70, 80, 90, 100],
            "price": [500000, 600000, 700000, 800000, 900000, 1000000],
        }
    )


@pytest.fixture
def mock_embedded_properties_df(mock_properties_df: pd.DataFrame) -> pd.DataFrame:
    """Mock properties dataframe with embeddings."""
    df = mock_properties_df.copy()
    # Add mock embeddings as a list of 16 floats for each property
    df["embeddings"] = [[0.1] * 16 for _ in range(len(df))]
    return df


@pytest.fixture
def mock_scores() -> npt.NDArray[np.float64]:
    """Mock scores for user-property pairs."""
    # Create scores for all properties in mock_properties_df (property IDs: 1001, 1002, 1003, 2001, 2002, 2003)
    return np.array(
        [
            [0.8, 0.6, 0.4, 0.7, 0.5, 0.3],  # scores for user1 and all properties
            [0.3, 0.7, 0.9, 0.2, 0.4, 0.6],  # scores for user2 and all properties
        ]
    )


@pytest.mark.asyncio
async def test_empty_leads_df(input_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_scores_df handles empty leads dataframe correctly."""
    # Mock load_leads_data_db_cached to return an empty DataFrame
    mock_load_leads = AsyncMock(return_value=pd.DataFrame())
    monkeypatch.setattr("src.v2.api.services.get_scores.load_leads_data_db_cached", mock_load_leads)

    # Call the function
    result = await get_scores_df(input_df)

    # Verify the function returns the original df with NaN scores
    assert "score" in result.columns
    assert result["score"].isna().all()
    assert mock_load_leads.called
    assert len(result) == len(input_df)
    assert list(result["user_id"]) == list(input_df["user_id"])
    assert list(result["property_id"]) == list(input_df["property_id"])


@pytest.mark.asyncio
async def test_empty_leads_properties_df(
    input_df: pd.DataFrame,
    mock_leads_df: pd.DataFrame,
    mock_properties_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that get_scores_df handles empty leads properties dataframe correctly."""
    # Mock load_leads_data_db_cached to return a valid DataFrame
    mock_load_leads = AsyncMock(return_value=mock_leads_df)
    monkeypatch.setattr("src.v2.api.services.get_scores.load_leads_data_db_cached", mock_load_leads)

    # Mock _load_data_cached to return an empty DataFrame for the first call (leads properties)
    mock_load_data_cached = AsyncMock()
    mock_load_data_cached.side_effect = [pd.DataFrame(), mock_properties_df]  # First call empty
    monkeypatch.setattr("src.v2.api.services.get_scores._load_data_cached", mock_load_data_cached)

    # Call the function
    result = await get_scores_df(input_df)

    # Verify the function returns the original df with NaN scores
    assert "score" in result.columns
    assert result["score"].isna().all()
    assert mock_load_leads.called
    assert mock_load_data_cached.called
    assert len(result) == len(input_df)


@pytest.mark.asyncio
async def test_empty_scoring_properties_df(
    input_df: pd.DataFrame,
    mock_leads_df: pd.DataFrame,
    mock_properties_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that get_scores_df handles empty scoring properties dataframe correctly."""
    # Mock load_leads_data_db_cached to return a valid DataFrame
    mock_load_leads = AsyncMock(return_value=mock_leads_df)
    monkeypatch.setattr("src.v2.api.services.get_scores.load_leads_data_db_cached", mock_load_leads)

    # Mock _load_data_cached to return valid DataFrame for leads properties but empty for scoring properties
    mock_load_data_cached = AsyncMock()
    mock_load_data_cached.side_effect = [mock_properties_df, pd.DataFrame()]
    monkeypatch.setattr("src.v2.api.services.get_scores._load_data_cached", mock_load_data_cached)

    # Call the function
    result = await get_scores_df(input_df)

    # Verify the function returns the original df with NaN scores
    assert "score" in result.columns
    assert result["score"].isna().all()
    assert mock_load_leads.called
    assert mock_load_data_cached.call_count == 2
    assert len(result) == len(input_df)


@pytest.mark.asyncio
async def test_successful_score_calculation(
    input_df: pd.DataFrame,
    mock_leads_df: pd.DataFrame,
    mock_properties_df: pd.DataFrame,
    mock_embedded_properties_df: pd.DataFrame,
    mock_scores: npt.NDArray[np.float64],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test successful score calculation with valid inputs."""
    # Mock load_leads_data_db_cached to return a valid DataFrame
    mock_load_leads = AsyncMock(return_value=mock_leads_df)
    monkeypatch.setattr("src.v2.api.services.get_scores.load_leads_data_db_cached", mock_load_leads)

    # Mock _load_data_cached to return valid DataFrames for both leads properties and scoring properties
    mock_load_data_cached = AsyncMock()
    mock_load_data_cached.side_effect = [mock_properties_df, mock_properties_df]
    monkeypatch.setattr("src.v2.api.services.get_scores._load_data_cached", mock_load_data_cached)

    # Mock embed_properties to return a DataFrame with embeddings
    mock_embed = MagicMock(return_value=mock_embedded_properties_df)
    monkeypatch.setattr("src.v2.api.services.get_scores.embed_properties", mock_embed)

    # Mock calculate_all_users_scores to return mock scores and user_ids
    mock_calculate = MagicMock(return_value=(mock_scores, ["user1", "user2"]))
    monkeypatch.setattr("src.v2.api.services.get_scores.calculate_all_users_scores", mock_calculate)

    # Call the function
    result = await get_scores_df(input_df)

    # Verify the scores are calculated correctly and merged with the original df
    assert "score" in result.columns
    assert not result["score"].isna().any()
    assert mock_load_leads.called
    assert mock_load_data_cached.call_count == 2
    assert mock_embed.call_count == 2
    assert mock_calculate.called
    assert len(result) == len(input_df)

    # Check the first user's scores
    user1_rows = result[result["user_id"] == "user1"]
    property_1001_score = user1_rows[user1_rows["property_id"] == 1001]["score"].values[0]
    property_1002_score = user1_rows[user1_rows["property_id"] == 1002]["score"].values[0]
    assert abs(property_1001_score - 0.8) < 1e-6
    assert abs(property_1002_score - 0.6) < 1e-6

    # Check the second user's score
    user2_rows = result[result["user_id"] == "user2"]
    property_1003_score = user2_rows[user2_rows["property_id"] == 1003]["score"].values[0]
    assert abs(property_1003_score - 0.9) < 1e-6


@pytest.mark.asyncio
async def test_cached_scores_df() -> None:
    """Test the cached version of get_scores_df."""
    # Import the cached version
    from src.v2.api.services.get_scores import get_scores_df_cached

    # This is a simple smoke test to ensure the function exists and can be called.
    # Actual testing of caching behavior would be more complex and is beyond this test.

    input_df = pd.DataFrame(
        {
            "user_id": ["user1", "user1", "user2"],
            "property_id": [1001, 1002, 1003],
        }
    )

    # Mock all the dependencies to avoid actual API calls
    with patch("src.v2.api.services.get_scores.get_scores_df") as mock_get_scores:
        mock_get_scores.return_value = pd.DataFrame(
            {
                "user_id": ["user1", "user1", "user2"],
                "property_id": [1001, 1002, 1003],
                "score": [0.8, 0.6, 0.9],
            }
        )

        with patch("src.v2.api.services.get_scores.batch_get_or_set_cache") as mock_cache:
            mock_cache.return_value = {
                "user1:1001": 0.8,
                "user1:1002": 0.6,
                "user2:1003": 0.9,
            }

            with patch("src.v2.api.services.get_scores.get_scores_cache") as mock_get_scores_cache:
                mock_get_scores_cache.return_value = MagicMock()

                # Call the function
                result = await get_scores_df_cached(input_df)

                # Basic assertions
                assert "score" in result.columns
                assert not result["score"].isna().any()
                assert len(result) == len(input_df)
                assert mock_cache.called
