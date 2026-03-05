import re
from enum import Enum
from typing import Annotated, TypedDict

import pandas as pd

from src.v2.config import get_config


class FeatureType(Enum):
    CATEGORICAL = 1
    NUMERICAL = 2


FEATURE_SPECS_TYPE = Annotated[
    tuple[str, FeatureType, float], "Tuple of regular expression, feature type, and weight for the feature."
]


class CoverageStats(TypedDict):
    coverage_percentage: float
    total_rows: int
    rows_with_data: int
    matching_columns: list[str]


# NOTE: The order of the features in the list is important since later on it is passed to a neural network as an input
APARTMENTS_FEATURE_SPECS: list[FEATURE_SPECS_TYPE] = [
    (r"^buildings$", FeatureType.NUMERICAL, 0.5),
    (r"^properties$", FeatureType.NUMERICAL, 0.5),
    (r"^lon$", FeatureType.NUMERICAL, 2.0),
    (r"^lat$", FeatureType.NUMERICAL, 2.0),
    (r"^facilities_.*$", FeatureType.CATEGORICAL, 0.5),
    (r"^natural_sites_.*$", FeatureType.CATEGORICAL, 0.5),
    (r"^area$", FeatureType.NUMERICAL, 2.0),
    (r"^normalize_price_m2$", FeatureType.NUMERICAL, 2.0),
    (r"^normalize_price$", FeatureType.NUMERICAL, 1.2),
    (r"^rooms$", FeatureType.NUMERICAL, 1.0),
    (r"^floor$", FeatureType.NUMERICAL, 0.5),
    (r"^bathrooms$", FeatureType.NUMERICAL, 0.5),
    (r"^kitchen_type_.*$", FeatureType.CATEGORICAL, 0.5),
    (r"^additional_area_type_.*$", FeatureType.CATEGORICAL, 1.0),
    (r"^additional_area_area$", FeatureType.NUMERICAL, 1.0),
    (r"^quarters_.*$", FeatureType.CATEGORICAL, 0.8),
    (r"^all_pois_count$", FeatureType.NUMERICAL, 0.3),
    (r"^all_pois_dist$", FeatureType.NUMERICAL, 0.3),
    (r"^shops_pois_count$", FeatureType.NUMERICAL, 0.5),
    (r"^shops_pois_dist$", FeatureType.NUMERICAL, 0.5),
    (r"^food_pois_count$", FeatureType.NUMERICAL, 0.5),
    (r"^food_pois_dist$", FeatureType.NUMERICAL, 0.5),
    (r"^education_pois_count$", FeatureType.NUMERICAL, 1.0),
    (r"^education_pois_dist$", FeatureType.NUMERICAL, 1.0),
    (r"^health_pois_count$", FeatureType.NUMERICAL, 0.8),
    (r"^health_pois_dist$", FeatureType.NUMERICAL, 0.8),
    (r"^entertainment_pois_count$", FeatureType.NUMERICAL, 0.5),
    (r"^entertainment_pois_dist$", FeatureType.NUMERICAL, 0.5),
    (r"^sport_pois_count$", FeatureType.NUMERICAL, 0.5),
    (r"^sport_pois_dist$", FeatureType.NUMERICAL, 0.5),
    (r"^transport_pois_count$", FeatureType.NUMERICAL, 1.0),
    (r"^transport_pois_dist$", FeatureType.NUMERICAL, 1.0),
]

FULL_FEATURE_SPECS: list[FEATURE_SPECS_TYPE] = [
    *APARTMENTS_FEATURE_SPECS,
    (r"^type_id_.*$", FeatureType.CATEGORICAL, 1.5),
    (r"^house_type_.*$", FeatureType.CATEGORICAL, 0.8),
    (r"^flat_type_.*$", FeatureType.CATEGORICAL, 0.8),
]

_SPEC_MAP = {
    "apartments": APARTMENTS_FEATURE_SPECS,
    "full": FULL_FEATURE_SPECS,
}


def _get_feature_specs_from_config() -> list[FEATURE_SPECS_TYPE]:
    feature_spec_type = get_config().properties_embedding_model.feature_spec_type
    if feature_spec_type in _SPEC_MAP:
        return _SPEC_MAP[feature_spec_type]
    else:
        raise ValueError(f"Invalid feature spec type: {feature_spec_type}. Must be one of {list(_SPEC_MAP.keys())}.")


def get_feature_columns(
    df: pd.DataFrame,
    feature_specs: list[FEATURE_SPECS_TYPE] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    feature_specs = feature_specs or _get_feature_specs_from_config()

    all_columns = df.columns.to_list()

    # compile features_spec
    features_regexp: list[re.Pattern[str]] = [re.compile(raw_regexp) for raw_regexp, _, _ in feature_specs]

    categorical_columns = []
    numerical_columns = []
    for i, spec in enumerate(features_regexp):
        if feature_specs[i][1] == FeatureType.CATEGORICAL:
            categorical_columns += [col for col in all_columns if spec.match(col)]
        else:
            numerical_columns += [col for col in all_columns if spec.match(col)]

    return categorical_columns + numerical_columns, categorical_columns, numerical_columns


def get_feature_weights(
    df: pd.DataFrame,
    normalize_weights: bool = True,
    feature_specs: list[FEATURE_SPECS_TYPE] | None = None,
) -> dict[str, float]:
    feature_specs = feature_specs or _get_feature_specs_from_config()

    # return weights in a form of a dict where key is a column name and value is a weight
    all_columns = df.columns.to_list()

    # compile features_spec
    features_regexp: list[re.Pattern[str]] = [re.compile(raw_regexp) for raw_regexp, _, _ in feature_specs]

    feature_weights = {}
    for i, spec in enumerate(features_regexp):
        columns = [col for col in all_columns if spec.match(col)]
        for col in columns:
            feature_weights[col] = feature_specs[i][2]

    if normalize_weights:
        # convert weight to sum to 1
        total_weight = sum(feature_weights.values())
        for col in feature_weights:
            feature_weights[col] /= total_weight

    return feature_weights


def get_feature_coverage(
    df: pd.DataFrame,
    feature_specs: list[FEATURE_SPECS_TYPE] | None = None,
) -> dict[str, CoverageStats]:
    """
    Calculate feature coverage for each feature group defined in feature specs.

    Args:
        df: DataFrame to analyze
        feature_specs: List of feature specifications. If None, uses config default.

    Returns:
        Dictionary with feature pattern as key and coverage stats as value.
        Each coverage stat contains:
        - 'coverage_percentage': Percentage of rows with at least one non-empty value
        - 'total_rows': Total number of rows
        - 'rows_with_data': Number of rows with at least one non-empty value
        - 'matching_columns': List of columns that matched the pattern

    Note:
        For numerical features, zero values are considered valid data (not missing).
        For categorical features, only non-zero values are considered valid data.
    """
    feature_specs = feature_specs or _get_feature_specs_from_config()

    all_columns = df.columns.to_list()
    total_rows = len(df)

    coverage_stats = {}

    for raw_regexp, feature_type, weight in feature_specs:
        pattern = re.compile(raw_regexp)
        matching_columns = [col for col in all_columns if pattern.match(col)]

        if not matching_columns:
            coverage_stats[raw_regexp] = CoverageStats(
                coverage_percentage=0.0,
                total_rows=total_rows,
                rows_with_data=0,
                matching_columns=[],
            )
            continue

        # Create a subset with only matching columns
        subset = df[matching_columns]

        # For categorical features, check if any column has non-zero values
        # For numerical features, check if any column has non-null values (zeros are valid)
        if feature_type == FeatureType.CATEGORICAL:
            # For categorical features (typically one-hot encoded), check for any 1s
            rows_with_data = int((subset > 0).any(axis=1).sum())
        else:
            # For numerical features, check for non-null values (zeros are considered valid data)
            rows_with_data = int(subset.notna().any(axis=1).sum())

        coverage_percentage = (rows_with_data / total_rows) * 100 if total_rows > 0 else 0.0

        coverage_stats[raw_regexp] = CoverageStats(
            coverage_percentage=coverage_percentage,
            total_rows=total_rows,
            rows_with_data=rows_with_data,
            matching_columns=matching_columns,
        )

    return coverage_stats


def print_feature_coverage_report(
    df: pd.DataFrame,
    feature_specs: list[FEATURE_SPECS_TYPE] | None = None,
    sort_by_coverage: bool = True,
) -> None:
    """
    Print a formatted report of feature coverage statistics.

    Args:
        df: DataFrame to analyze
        feature_specs: List of feature specifications. If None, uses config default.
        sort_by_coverage: Whether to sort results by coverage percentage (descending)
    """
    coverage_stats = get_feature_coverage(df, feature_specs)

    # Convert to list of tuples for sorting
    stats_items = list(coverage_stats.items())

    if sort_by_coverage:
        stats_items.sort(key=lambda x: x[1]["coverage_percentage"], reverse=True)

    print("Feature Coverage Report")
    print("=" * 80)
    print(f"{'Feature Pattern':<30} {'Coverage %':<12} {'Rows with Data':<15} {'Columns':<8}")
    print("-" * 80)

    for pattern, stats in stats_items:
        coverage_pct = stats["coverage_percentage"]
        rows_with_data = stats["rows_with_data"]
        total_rows = stats["total_rows"]
        matching_columns = stats["matching_columns"]
        num_columns = len(matching_columns)

        print(f"{pattern:<30} {coverage_pct:>10.2f}% {rows_with_data:>6}/{total_rows:<6} {num_columns:>6}")

    print("-" * 80)
    total_rows = coverage_stats[list(coverage_stats.keys())[0]]["total_rows"] if coverage_stats else 0
    print(f"Total dataset rows: {total_rows}")
