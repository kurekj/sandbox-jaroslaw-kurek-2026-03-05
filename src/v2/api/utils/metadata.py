from functools import lru_cache

from src.v2.api.models.scores import ScoresMetadata
from src.v2.config import get_config


@lru_cache()
def get_scores_metadata() -> ScoresMetadata:
    """
    Get all metadata for the scores response in a single cached call.

    This function is cached to avoid repeated work for each request,
    since the metadata typically doesn't change during application runtime.

    Returns:
        ScoresMetadata: Complete metadata object with all required fields.
    """
    config = get_config()

    return ScoresMetadata(
        recommendation_model=config.properties_embedding_model,
        app_version=config.app.version,
    )
