from pydantic import BaseModel


class PrefillCacheRequest(BaseModel):
    """Request model for prefill cache operation.

    This model defines optional parameters to control the cache prefilling process,
    specifically the overwrite behavior for different types of cached data.
    """

    overwrite_visible_properties: bool = True
    """Whether to overwrite existing visible properties data (default: True)"""

    overwrite_pois: bool = False
    """Whether to overwrite existing POI (Points of Interest) data (default: False)"""
