from math import inf
from typing import Any, Hashable

# def get_poi_count(row: pd.Series) -> dict[str, dict[str, int]]:  # type: ignore
#     pois = {
#         "ALL": {"count": 0, "dist": inf},
#         "SHOPS": {"count": 0, "dist": inf},
#         "FOOD": {"count": 0, "dist": inf},
#         "EDUCATION": {"count": 0, "dist": inf},
#         "HEALTH": {"count": 0, "dist": inf},
#         "ENTERTAINMENT": {"count": 0, "dist": inf},
#         "SPORT": {"count": 0, "dist": inf},
#         "TRANSPORT": {"count": 0, "dist": inf},
#     }

#     if row["poi"]:
#         for poi in row["poi"]:
#             if poi["agg_type"] in pois:
#                 pois[poi["agg_type"]]["count"] += 1
#                 pois[poi["agg_type"]]["dist"] = min(pois[poi["agg_type"]]["dist"], poi["dist"])

#     # replace inf with -1
#     for k, v in pois.items():
#         if v["dist"] == inf:
#             v["dist"] = -1

#     return pois  # type: ignore


def get_poi_count(raw_pois: list[dict[Hashable, Any]]) -> dict[str, dict[str, int]]:  # type: ignore
    pois = {
        "ALL": {"count": 0, "dist": inf},
        "SHOPS": {"count": 0, "dist": inf},
        "FOOD": {"count": 0, "dist": inf},
        "EDUCATION": {"count": 0, "dist": inf},
        "HEALTH": {"count": 0, "dist": inf},
        "ENTERTAINMENT": {"count": 0, "dist": inf},
        "SPORT": {"count": 0, "dist": inf},
        "TRANSPORT": {"count": 0, "dist": inf},
    }

    for poi in raw_pois:
        if poi["agg_type"] in pois:
            pois[poi["agg_type"]]["count"] += 1
            pois[poi["agg_type"]]["dist"] = min(pois[poi["agg_type"]]["dist"], poi["dist"])

    # replace inf with -1
    for k, v in pois.items():
        if v["dist"] == inf:
            v["dist"] = -1

    return pois  # type: ignore
