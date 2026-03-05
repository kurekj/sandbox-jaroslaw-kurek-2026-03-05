from typing import Any, Optional

from pydantic import BaseModel

from src.v2.config import PropertiesEmbeddingModelConfig


class ScoresRequest(BaseModel):
    data: list[dict[str, Any]]
    """
    A list of dictionaries containing user_id and property_id pairs.
    Each dictionary should have the following structure:
    {
        "user_id": int,
        "property_id": int
    }
    """


class ScoresMetadata(BaseModel):
    """Metadata about the recommendation system and execution context."""

    recommendation_model: Optional[PropertiesEmbeddingModelConfig] = None
    """The configuration of the recommendation model, including the MLflow artifact path."""
    app_version: Optional[str] = None
    """The version of the application."""


class ScoresResponse(BaseModel):
    scores: list[dict[str, Any]]
    """
    A list of dictionaries containing the scores for each user_id and property_id pair.
    Each dictionary should have the following structure:
    {
        "user_id": int,
        "property_id": int,
        "score": float
    }
    """

    metadata: ScoresMetadata
    """Metadata about the recommendation system and execution context."""
