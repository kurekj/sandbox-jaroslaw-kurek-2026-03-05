"""
Script to compute recommendation scores with XAI explanations.
It loads a feature mapping from feature_mapping.json, converts user_id values
to UUIDs, computes SHAP contributions with a surrogate model, and outputs
both raw and sorted contributions.
"""

import asyncio
import json
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
import shap  # type: ignore

# adjust event loop policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# import existing functions from your project
from src.v2.api.services.get_scores import get_scores_df  # type: ignore
from src.v2.api.services.get_scores import _load_data_cached  # type: ignore

# expected feature columns
EXPECTED_FEATURE_COLUMNS: List[str] = [
    *(f"facilities_{i}" for i in range(1, 36)),
    *(f"natural_sites_{i}" for i in range(1, 10)),
    *(f"kitchen_type_{i}" for i in range(1, 3)),
    *(f"additional_area_type_{i}" for i in range(1, 5)),
    *(f"quarters_{i}" for i in range(1, 5)),
    *(f"type_id_{i}" for i in range(1, 4)),
    *(f"house_type_{i}" for i in range(1, 5)),
    *(f"flat_type_{i}" for i in range(1, 4)),
    "buildings", "properties", "lon", "lat", "area",
    "normalize_price_m2", "normalize_price", "rooms", "floor",
    "bathrooms", "additional_area_area",
    "all_pois_count", "all_pois_dist",
    "shops_pois_count", "shops_pois_dist",
    "food_pois_count", "food_pois_dist",
    "education_pois_count", "education_pois_dist",
    "health_pois_count", "health_pois_dist",
    "entertainment_pois_count", "entertainment_pois_dist",
    "sport_pois_count", "sport_pois_dist",
    "transport_pois_count", "transport_pois_dist",
]

def load_feature_mapping(path: str = "feature_mapping.json") -> Dict[str, Any]:
    """Load mapping of raw feature names to human‑readable names."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load feature mapping: {e}")
        return {}

def coerce_uuid(value: Any) -> uuid.UUID:
    """Convert different types of user_id to uuid.UUID to satisfy DB queries."""
    if isinstance(value, uuid.UUID):
        return value
    # If value is already a valid UUID string
    try:
        return uuid.UUID(str(value))
    except Exception:
        pass
    # If value is an integer, create a deterministic UUID based on the integer
    try:
        return uuid.UUID(int=int(value))
    except Exception:
        # fallback: generate a namespace-based UUID from the string representation
        return uuid.uuid5(uuid.NAMESPACE_OID, str(value))

async def _get_property_features(property_ids: List[int]) -> pd.DataFrame:
    """Load features for specified property IDs."""
    raw_df = await _load_data_cached(ids=property_ids, overwrite=False)
    for col in EXPECTED_FEATURE_COLUMNS:
        if col not in raw_df.columns:
            raw_df[col] = np.nan
    return raw_df[["property_id", *EXPECTED_FEATURE_COLUMNS]].copy()

async def get_scores_with_explanations(
    df: pd.DataFrame,
    n_estimators: int = 50,
    random_state: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """Compute scores and SHAP explanations, including sorted contributions."""
    # convert user_id to uuid to match DB column type
    df = df.copy()
    df["user_id"] = df["user_id"].apply(coerce_uuid)

    feature_mapping = load_feature_mapping()

    scores_df = await get_scores_df(df)
    valid_df = scores_df[scores_df["score"].notnull()].copy()
    if valid_df.empty:
        logger.warning("No valid scores; returning empty list.")
        return []

    property_ids = valid_df["property_id"].astype(int).unique().tolist()
    feature_df = await _get_property_features(property_ids)
    merged = valid_df.merge(feature_df, on="property_id", how="left")

    X = merged[EXPECTED_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = merged["score"].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0)

    surrogate = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    surrogate.fit(X, y)
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X)

    results: List[Dict[str, Any]] = []
    for idx, row in merged.iterrows():
        shap_row = shap_values[idx]
        contributions = {
            feature: float(shap_row[i])
            for i, feature in enumerate(EXPECTED_FEATURE_COLUMNS)
        }
        sorted_contrib = sorted(
            [
                {
                    "feature": feature,
                    "mapped_feature": feature_mapping.get(feature, feature),
                    "contribution": contributions[feature],
                }
                for feature in EXPECTED_FEATURE_COLUMNS
            ],
            key=lambda x: abs(x["contribution"]),
            reverse=True,
        )
        results.append(
            {
                "user_id": str(row["user_id"]),  # convert UUID back to string for JSON
                "property_id": int(row["property_id"]),
                "score": float(row["score"]),
                "feature_contributions": contributions,
                "sorted_contributions": sorted_contrib,
            }
        )
    return results

# convenience wrapper to produce JSON string
async def get_scores_with_explanations_json(
    df: pd.DataFrame,
    n_estimators: int = 50,
    random_state: Optional[int] = 42,
) -> str:
    explanations = await get_scores_with_explanations(df, n_estimators, random_state)
    return json.dumps(explanations, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    logger.add("xai_report.log")
    logger.info("Starting XAI scoring script...")

    # Example: replace with real UUIDs and property IDs
    example_df = pd.DataFrame({
        "user_id": [
            "d3a63a17-8047-4e19-a8c6-27ed8acde846",
            "3b8612fd-3b9d-4d4f-b6a2-35e9189b5f42",
        ],
        "property_id": [101, 202],
    })

    explanations = asyncio.run(get_scores_with_explanations(example_df))
    with open("xai_explanations.json", "w", encoding="utf-8") as f:
        json.dump(explanations, f, ensure_ascii=False, indent=2)
    for entry in explanations:
        logger.info(
            f"User {entry['user_id']}, property {entry['property_id']}, "
            f"score={entry['score']:.4f}, top feature: "
            f"{entry['sorted_contributions'][0]['feature']} "
            f"({entry['sorted_contributions'][0]['mapped_feature']}) "
            f"with contribution={entry['sorted_contributions'][0]['contribution']:.4f}"
        )
    logger.info("XAI scoring completed; results saved to xai_explanations.json")