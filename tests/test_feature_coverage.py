from __future__ import annotations

import io
from contextlib import redirect_stdout

import pandas as pd
import pytest

from src.v2.autoencoder.feature_specs import (
    APARTMENTS_FEATURE_SPECS,
    FEATURE_SPECS_TYPE,
    FeatureType,
    get_feature_coverage,
    print_feature_coverage_report,
)


def create_sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing feature coverage functions."""
    return pd.DataFrame(
        {
            # Location features
            "lon": [20.0, 21.0, 19.5, None, 22.0],
            "lat": [52.0, 52.1, 51.9, 52.2, None],
            # Property features
            "area": [50.0, 75.0, 60.0, 45.0, 80.0],
            "normalize_price_m2": [8000, 9000, 7500, 8500, 9500],
            "normalize_price": [400000, 675000, 450000, 382500, 760000],
            "rooms": [2, 3, 2, 2, 4],
            "floor": [3, 5, 1, 2, 7],
            "bathrooms": [1, 2, 1, 1, 2],
            # Categorical features (facilities) - one-hot encoded
            "facilities_parking": [1, 0, 1, 0, 1],
            "facilities_balcony": [0, 1, 1, 0, 1],
            "facilities_elevator": [1, 1, 0, 0, 1],
            # Natural sites - some missing
            "natural_sites_park": [1, 0, 0, 0, 1],
            "natural_sites_forest": [0, 0, 0, 0, 0],  # No coverage
            # Kitchen types
            "kitchen_type_open": [1, 0, 1, 1, 0],
            "kitchen_type_closed": [0, 1, 0, 0, 1],
            # POI counts and distances
            "all_pois_count": [15, 12, 8, 20, 25],
            "all_pois_dist": [0.5, 0.8, 1.2, 0.3, 0.2],
            "shops_pois_count": [5, 3, 2, 8, 10],
            "shops_pois_dist": [0.2, 0.4, 0.8, 0.1, 0.1],
            "education_pois_count": [2, 1, 0, 3, 4],  # Includes zero as valid value
            "education_pois_dist": [0.8, 1.2, None, 0.5, 0.3],  # One missing
            # Additional features
            "additional_area_type_garage": [1, 0, 0, 1, 1],
            "additional_area_type_garden": [0, 1, 0, 0, 0],
            "additional_area_area": [15, 25, 0, 10, 20],  # Includes zero as valid value
            # Quarter information
            "quarters_downtown": [1, 0, 0, 1, 0],
            "quarters_suburbs": [0, 1, 1, 0, 1],
        }
    )


def create_empty_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame for testing edge cases."""
    return pd.DataFrame()


def create_minimal_feature_specs() -> list[FEATURE_SPECS_TYPE]:
    """Create a minimal feature specs list for testing."""
    return [
        (r"^area$", FeatureType.NUMERICAL, 1.0),
        (r"^facilities_.*$", FeatureType.CATEGORICAL, 0.5),
        (r"^non_existent_.*$", FeatureType.NUMERICAL, 1.0),
    ]


def test_get_feature_coverage_basic_functionality() -> None:
    """Test basic functionality of get_feature_coverage function."""
    df = create_sample_dataframe()
    feature_specs = create_minimal_feature_specs()

    coverage_stats = get_feature_coverage(df, feature_specs)

    # Check that all patterns from feature_specs are in the result
    assert len(coverage_stats) == 3
    assert "^area$" in coverage_stats
    assert "^facilities_.*$" in coverage_stats
    assert "^non_existent_.*$" in coverage_stats

    # Check structure of coverage stats
    area_stats = coverage_stats["^area$"]
    assert "coverage_percentage" in area_stats
    assert "total_rows" in area_stats
    assert "rows_with_data" in area_stats
    assert "matching_columns" in area_stats

    # Check data types
    assert isinstance(area_stats["coverage_percentage"], float)
    assert isinstance(area_stats["total_rows"], int)
    assert isinstance(area_stats["rows_with_data"], int)
    assert isinstance(area_stats["matching_columns"], list)


def test_get_feature_coverage_numerical_features() -> None:
    """Test coverage calculation for numerical features."""
    df = create_sample_dataframe()
    feature_specs = [(r"^area$", FeatureType.NUMERICAL, 1.0)]

    coverage_stats = get_feature_coverage(df, feature_specs)
    area_stats = coverage_stats["^area$"]

    # Area column has all non-null values
    assert area_stats["coverage_percentage"] == 100.0
    assert area_stats["total_rows"] == 5
    assert area_stats["rows_with_data"] == 5
    assert area_stats["matching_columns"] == ["area"]


def test_get_feature_coverage_numerical_with_nulls() -> None:
    """Test coverage calculation for numerical features with null values."""
    df = create_sample_dataframe()
    feature_specs = [(r"^lon$", FeatureType.NUMERICAL, 1.0)]

    coverage_stats = get_feature_coverage(df, feature_specs)
    lon_stats = coverage_stats["^lon$"]

    # Lon column has 4 out of 5 non-null values
    assert lon_stats["coverage_percentage"] == 80.0
    assert lon_stats["total_rows"] == 5
    assert lon_stats["rows_with_data"] == 4
    assert lon_stats["matching_columns"] == ["lon"]


def test_get_feature_coverage_numerical_with_zeros() -> None:
    """Test coverage calculation for numerical features with zero values."""
    df = create_sample_dataframe()
    feature_specs = [(r"^education_pois_count$", FeatureType.NUMERICAL, 1.0)]

    coverage_stats = get_feature_coverage(df, feature_specs)
    education_stats = coverage_stats["^education_pois_count$"]

    # Education count has one zero value, but zeros are now valid data - only nulls are missing
    assert education_stats["coverage_percentage"] == 100.0
    assert education_stats["total_rows"] == 5
    assert education_stats["rows_with_data"] == 5
    assert education_stats["matching_columns"] == ["education_pois_count"]


def test_get_feature_coverage_numerical_zeros_are_valid() -> None:
    """Test that zeros are treated as valid values for numerical features."""
    df = pd.DataFrame({
        "count_col": [0, 0, 0, 1, 2],  # 3 zeros, 2 non-zeros - all should be valid
        "mixed_col": [0, None, 1, None, 0],  # 3 valid values (including zeros), 2 nulls
    })
    feature_specs = [
        (r"^count_col$", FeatureType.NUMERICAL, 1.0),
        (r"^mixed_col$", FeatureType.NUMERICAL, 1.0),
    ]

    coverage_stats = get_feature_coverage(df, feature_specs)

    # All zeros should be considered valid data
    count_stats = coverage_stats["^count_col$"]
    assert count_stats["coverage_percentage"] == 100.0
    assert count_stats["rows_with_data"] == 5

    # Only non-null values should be considered valid
    mixed_stats = coverage_stats["^mixed_col$"]
    assert mixed_stats["coverage_percentage"] == 60.0  # 3/5 * 100
    assert mixed_stats["rows_with_data"] == 3


def test_get_feature_coverage_categorical_features() -> None:
    """Test coverage calculation for categorical features."""
    df = create_sample_dataframe()
    feature_specs = [(r"^facilities_.*$", FeatureType.CATEGORICAL, 0.5)]

    coverage_stats = get_feature_coverage(df, feature_specs)
    facilities_stats = coverage_stats["^facilities_.*$"]

    # Check that it found all facility columns
    expected_columns = ["facilities_parking", "facilities_balcony", "facilities_elevator"]
    assert set(facilities_stats["matching_columns"]) == set(expected_columns)

    # All rows have at least one facility (checking manually from sample data)
    assert facilities_stats["coverage_percentage"] == 80.0
    assert facilities_stats["total_rows"] == 5
    assert facilities_stats["rows_with_data"] == 4


def test_get_feature_coverage_categorical_no_coverage() -> None:
    """Test coverage calculation for categorical features with no coverage."""
    df = pd.DataFrame(
        {
            "natural_sites_forest": [0, 0, 0, 0, 0],  # All zeros
            "other_column": [1, 2, 3, 4, 5],
        }
    )
    feature_specs = [(r"^natural_sites_.*$", FeatureType.CATEGORICAL, 0.5)]

    coverage_stats = get_feature_coverage(df, feature_specs)
    natural_stats = coverage_stats["^natural_sites_.*$"]

    assert natural_stats["coverage_percentage"] == 0.0
    assert natural_stats["total_rows"] == 5
    assert natural_stats["rows_with_data"] == 0
    assert natural_stats["matching_columns"] == ["natural_sites_forest"]


def test_get_feature_coverage_no_matching_columns() -> None:
    """Test coverage calculation when no columns match the pattern."""
    df = create_sample_dataframe()
    feature_specs = [(r"^non_existent_.*$", FeatureType.NUMERICAL, 1.0)]

    coverage_stats = get_feature_coverage(df, feature_specs)
    non_existent_stats = coverage_stats["^non_existent_.*$"]

    assert non_existent_stats["coverage_percentage"] == 0.0
    assert non_existent_stats["total_rows"] == 5
    assert non_existent_stats["rows_with_data"] == 0
    assert non_existent_stats["matching_columns"] == []


def test_get_feature_coverage_empty_dataframe() -> None:
    """Test coverage calculation with empty DataFrame."""
    df = create_empty_dataframe()
    feature_specs = create_minimal_feature_specs()

    coverage_stats = get_feature_coverage(df, feature_specs)

    for pattern in ["^area$", "^facilities_.*$", "^non_existent_.*$"]:
        stats = coverage_stats[pattern]
        assert stats["coverage_percentage"] == 0.0
        assert stats["total_rows"] == 0
        assert stats["rows_with_data"] == 0
        assert stats["matching_columns"] == []


def test_get_feature_coverage_multiple_matching_columns() -> None:
    """Test coverage calculation when pattern matches multiple columns."""
    df = create_sample_dataframe()
    feature_specs = [(r"^quarters_.*$", FeatureType.CATEGORICAL, 0.8)]

    coverage_stats = get_feature_coverage(df, feature_specs)
    quarters_stats = coverage_stats["^quarters_.*$"]

    # Should match both quarters_downtown and quarters_suburbs
    expected_columns = ["quarters_downtown", "quarters_suburbs"]
    assert set(quarters_stats["matching_columns"]) == set(expected_columns)
    assert quarters_stats["coverage_percentage"] == 100.0  # All rows have at least one quarter


def test_get_feature_coverage_apartments_specs() -> None:
    """Test coverage calculation using APARTMENTS_FEATURE_SPECS."""
    df = create_sample_dataframe()

    coverage_stats = get_feature_coverage(df, APARTMENTS_FEATURE_SPECS)

    # Check that we have results for all patterns in APARTMENTS_FEATURE_SPECS
    assert len(coverage_stats) == len(APARTMENTS_FEATURE_SPECS)

    # Test some specific patterns we know exist in our sample data
    assert "^area$" in coverage_stats
    assert "^facilities_.*$" in coverage_stats
    assert "^quarters_.*$" in coverage_stats

    # Test some patterns that don't exist in our sample data
    assert "^buildings$" in coverage_stats
    assert coverage_stats["^buildings$"]["coverage_percentage"] == 0.0


def test_get_feature_coverage_default_feature_specs() -> None:
    """Test that function uses default feature specs when none provided."""
    df = create_sample_dataframe()

    # This should use the default feature specs from config
    # We can't easily test the exact behavior without mocking the config,
    # but we can test that it doesn't raise an error
    coverage_stats = get_feature_coverage(df)

    assert isinstance(coverage_stats, dict)
    assert len(coverage_stats) > 0


def test_print_feature_coverage_report_output() -> None:
    """Test that print_feature_coverage_report produces expected output."""
    df = create_sample_dataframe()
    feature_specs = create_minimal_feature_specs()

    # Capture stdout
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        print_feature_coverage_report(df, feature_specs)

    output = captured_output.getvalue()

    # Check that expected elements are in the output
    assert "Feature Coverage Report" in output
    assert "Feature Pattern" in output
    assert "Coverage %" in output
    assert "Rows with Data" in output
    assert "Columns" in output
    assert "Total dataset rows:" in output
    assert "^area$" in output
    assert "^facilities_.*$" in output
    assert "100.00%" in output  # Area should have 100% coverage


def test_print_feature_coverage_report_sorting() -> None:
    """Test that print_feature_coverage_report sorts correctly."""
    df = create_sample_dataframe()
    feature_specs = [
        (r"^area$", FeatureType.NUMERICAL, 1.0),  # Should have 100% coverage
        (r"^lon$", FeatureType.NUMERICAL, 1.0),  # Should have 80% coverage
        (r"^non_existent_.*$", FeatureType.NUMERICAL, 1.0),  # Should have 0% coverage
    ]

    # Test with sorting enabled (default)
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        print_feature_coverage_report(df, feature_specs, sort_by_coverage=True)

    output = captured_output.getvalue()
    lines = output.split("\n")

    # Find the data lines (skip headers)
    data_lines = [line for line in lines if line.startswith("^")]

    # Check that they are sorted by coverage percentage (descending)
    assert len(data_lines) >= 3
    assert "^area$" in data_lines[0]  # Should be first (100%)
    assert "^lon$" in data_lines[1]  # Should be second (80%)
    assert "^non_existent_.*$" in data_lines[2]  # Should be last (0%)


def test_print_feature_coverage_report_no_sorting() -> None:
    """Test that print_feature_coverage_report respects sort_by_coverage=False."""
    df = create_sample_dataframe()
    feature_specs = create_minimal_feature_specs()

    # Test with sorting disabled
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        print_feature_coverage_report(df, feature_specs, sort_by_coverage=False)

    output = captured_output.getvalue()

    # Should still contain the same content
    assert "Feature Coverage Report" in output
    assert "^area$" in output
    assert "^facilities_.*$" in output


def test_print_feature_coverage_report_empty_dataframe() -> None:
    """Test print_feature_coverage_report with empty DataFrame."""
    df = create_empty_dataframe()
    feature_specs = create_minimal_feature_specs()

    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        print_feature_coverage_report(df, feature_specs)

    output = captured_output.getvalue()

    assert "Feature Coverage Report" in output
    assert "Total dataset rows: 0" in output


def test_coverage_percentage_calculation() -> None:
    """Test exact coverage percentage calculations."""
    # Create a DataFrame where we know exact coverage
    df = pd.DataFrame(
        {
            "test_col": [1, 2, None, 4, 0],  # 4 out of 5 have non-null values (zero is valid)
        }
    )
    feature_specs = [(r"^test_col$", FeatureType.NUMERICAL, 1.0)]

    coverage_stats = get_feature_coverage(df, feature_specs)
    test_stats = coverage_stats["^test_col$"]

    assert test_stats["coverage_percentage"] == 80.0  # 4/5 * 100
    assert test_stats["rows_with_data"] == 4
    assert test_stats["total_rows"] == 5


def test_categorical_coverage_calculation() -> None:
    """Test exact coverage calculation for categorical features."""
    # Create a DataFrame where we know exact categorical coverage
    df = pd.DataFrame(
        {
            "cat_a": [1, 0, 0, 0, 0],  # Row 0 has data
            "cat_b": [0, 1, 0, 0, 0],  # Row 1 has data
            "cat_c": [0, 0, 0, 1, 0],  # Row 3 has data
            # Row 2 and 4 have no data (all zeros)
        }
    )
    feature_specs = [(r"^cat_.*$", FeatureType.CATEGORICAL, 1.0)]

    coverage_stats = get_feature_coverage(df, feature_specs)
    cat_stats = coverage_stats["^cat_.*$"]

    assert cat_stats["coverage_percentage"] == 60.0  # 3/5 * 100 (rows 0, 1, 3)
    assert cat_stats["rows_with_data"] == 3
    assert cat_stats["total_rows"] == 5
    assert set(cat_stats["matching_columns"]) == {"cat_a", "cat_b", "cat_c"}


@pytest.mark.parametrize(
    "total_rows,rows_with_data,expected_percentage",
    [
        (100, 50, 50.0),
        (1000, 750, 75.0),
        (10, 0, 0.0),
        (1, 1, 100.0),
        (3, 2, 66.66666666666667),
    ],
)
def test_coverage_percentage_edge_cases(total_rows: int, rows_with_data: int, expected_percentage: float) -> None:
    """Test coverage percentage calculation with various edge cases."""
    # Create a DataFrame with the specified characteristics
    # Use None for missing data and 1 for valid data to get the expected coverage
    data = {"test_col": [1 if i < rows_with_data else None for i in range(total_rows)]}
    df = pd.DataFrame(data)
    feature_specs = [(r"^test_col$", FeatureType.NUMERICAL, 1.0)]

    coverage_stats = get_feature_coverage(df, feature_specs)
    test_stats = coverage_stats["^test_col$"]

    assert abs(test_stats["coverage_percentage"] - expected_percentage) < 1e-10
    assert test_stats["rows_with_data"] == rows_with_data
    assert test_stats["total_rows"] == total_rows
