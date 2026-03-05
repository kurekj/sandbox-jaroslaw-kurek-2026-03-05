import numpy as np
import pandas as pd
import pytest

from src.v2.utils.encode_to_mhot import _safe_convert, encode_to_mhot


def test_encode_to_mhot_single_value() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "categories": ["A", "B", "C"],
            "all_categories": [["A", "B", "C"]] * 3,
        }
    )
    result = encode_to_mhot(df, "categories", "all_categories")
    expected_columns = ["id", "categories", "all_categories", "categories_a", "categories_b", "categories_c"]
    assert all(col in result.columns for col in expected_columns)
    assert result["categories_a"].tolist() == [1, 0, 0]
    assert result["categories_b"].tolist() == [0, 1, 0]
    assert result["categories_c"].tolist() == [0, 0, 1]


def test_encode_to_mhot_list_values() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "categories": [["A", "B"], ["B", "C"], ["A", "C"]],
            "all_categories": [["A", "B", "C"]] * 3,
        }
    )
    result = encode_to_mhot(df, "categories", "all_categories")
    expected_columns = ["id", "categories", "all_categories", "categories_a", "categories_b", "categories_c"]
    assert all(col in result.columns for col in expected_columns)
    assert result["categories_a"].tolist() == [1, 0, 1]
    assert result["categories_b"].tolist() == [1, 1, 0]
    assert result["categories_c"].tolist() == [0, 1, 1]


def test_encode_to_mhot_nan_values() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "categories": [float("nan"), "B", "C"],
            "all_categories": [["A", "B", "C"]] * 3,
        }
    )
    result = encode_to_mhot(df, "categories", "all_categories")
    expected_columns = ["id", "categories", "all_categories", "categories_a", "categories_b", "categories_c"]
    assert all(col in result.columns for col in expected_columns)
    assert result["categories_a"].tolist() == [0, 0, 0]
    assert result["categories_b"].tolist() == [0, 1, 0]
    assert result["categories_c"].tolist() == [0, 0, 1]


def test_encode_to_mhot_existing_columns() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "categories": ["A", "B", "C"],
            "all_categories": [["A", "B", "C"]] * 3,
            "categories_a": [0, 0, 0],
            "categories_b": [0, 0, 0],
            "categories_c": [0, 0, 0],
        }
    )
    result = encode_to_mhot(df, "categories", "all_categories")
    expected_columns = ["id", "categories", "all_categories", "categories_a", "categories_b", "categories_c"]
    assert all(col in result.columns for col in expected_columns)
    assert result["categories_a"].tolist() == [1, 0, 0]
    assert result["categories_b"].tolist() == [0, 1, 0]
    assert result["categories_c"].tolist() == [0, 0, 1]


def test_safe_convert() -> None:
    assert _safe_convert(np.nan) == []
    assert _safe_convert(float("nan")) == []
    assert _safe_convert(1.0) == ["1"]
    assert _safe_convert("A") == ["A"]
    assert _safe_convert(["A", "B"]) == ["A", "B"]
    with pytest.raises(ValueError):
        _safe_convert(object())
