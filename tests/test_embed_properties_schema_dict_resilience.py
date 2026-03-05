import numpy as np
import pandas as pd
import pytest

from src.v2.api.services.embed_properties import embed_properties
from src.v2.autoencoder.feature_columns_const import EXPECTED_FEATURE_COLUMNS


class _DummyModel:
    def __init__(self) -> None:
        self.last_input_columns: list[str] | None = None

    def get_embeddings(self, df: pd.DataFrame) -> np.ndarray:  # type: ignore
        # capture the exact columns order used for embedding
        self.last_input_columns = df.columns.tolist()
        # return a fixed-size embedding (e.g., 8 dims) for each row
        return np.zeros((len(df), 8), dtype=float)


def _build_minimal_feature_df() -> pd.DataFrame:
    """Create a one-row DataFrame with all expected feature columns initialized to zeros."""
    data = {col: [0.0] for col in EXPECTED_FEATURE_COLUMNS}
    return pd.DataFrame(data)


@pytest.mark.asyncio
async def test_embed_properties_ignores_extra_mhot_columns_from_schema_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a minimal DF with all expected features
    df = _build_minimal_feature_df()

    # Simulate schema dictionary gaining new values that produce extra mhot columns
    # Use extreme out-of-range ids to reflect late additions, e.g. facilities_999 / flat_type_999
    df["facilities_999"] = [1.0]
    df["flat_type_999"] = [1.0]

    dummy_model = _DummyModel()
    # Patch model loader to avoid real checkpoint
    monkeypatch.setattr("src.v2.api.services.embed_properties._load_model", lambda *_args, **_kwargs: dummy_model)

    # Run embedder (synchronous function)
    result = embed_properties(df)

    # Ensure embeddings column was added
    assert "embeddings" in result.columns
    assert len(result) == 1

    # Verify that model received exactly EXPECTED_FEATURE_COLUMNS (extras dropped), order preserved
    assert dummy_model.last_input_columns == EXPECTED_FEATURE_COLUMNS


@pytest.mark.asyncio
async def test_embed_properties_ignores_new_schema_dict_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    # Start with expected features DF
    df = _build_minimal_feature_df()

    # Simulate adding two entirely new dictionary groups that our pipeline doesn't know about
    # These should be ignored by get_feature_columns and alignment
    df["brand_new_dict_a_1"] = [1.0]
    df["brand_new_dict_a_2"] = [0.0]
    df["brand_new_dict_b_1"] = [1.0]

    dummy_model = _DummyModel()
    monkeypatch.setattr("src.v2.api.services.embed_properties._load_model", lambda *_args, **_kwargs: dummy_model)

    result = embed_properties(df)

    # Embeddings added and only expected features used for the model
    assert "embeddings" in result.columns
    assert len(result) == 1
    assert dummy_model.last_input_columns == EXPECTED_FEATURE_COLUMNS
