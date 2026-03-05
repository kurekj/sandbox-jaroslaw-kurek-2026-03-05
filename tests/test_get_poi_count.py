from typing import Any, Hashable

from src.v2.utils.get_poi_count import get_poi_count


def test_get_poi_count_no_pois() -> None:
    row: list[dict[Hashable, Any]] = []
    result = get_poi_count(row)
    expected = {
        "ALL": {"count": 0, "dist": -1},
        "SHOPS": {"count": 0, "dist": -1},
        "FOOD": {"count": 0, "dist": -1},
        "EDUCATION": {"count": 0, "dist": -1},
        "HEALTH": {"count": 0, "dist": -1},
        "ENTERTAINMENT": {"count": 0, "dist": -1},
        "SPORT": {"count": 0, "dist": -1},
        "TRANSPORT": {"count": 0, "dist": -1},
    }
    assert result == expected


def test_get_poi_count_within_radius() -> None:
    row: list[dict[Hashable, Any]] = [{"dist": 500, "agg_type": "SHOPS"}, {"dist": 800, "agg_type": "FOOD"}]
    result = get_poi_count(row)
    expected = {
        "ALL": {"count": 0, "dist": -1},
        "SHOPS": {"count": 1, "dist": 500},
        "FOOD": {"count": 1, "dist": 800},
        "EDUCATION": {"count": 0, "dist": -1},
        "HEALTH": {"count": 0, "dist": -1},
        "ENTERTAINMENT": {"count": 0, "dist": -1},
        "SPORT": {"count": 0, "dist": -1},
        "TRANSPORT": {"count": 0, "dist": -1},
    }
    assert result == expected
