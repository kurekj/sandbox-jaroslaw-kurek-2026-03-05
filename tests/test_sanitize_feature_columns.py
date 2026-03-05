import unittest

import numpy as np
import pandas as pd
import pytest

from src.v2.api.services.embed_properties import sanitize_feature_columns


class TestSanitizeFeatureColumns(unittest.TestCase):
    df: pd.DataFrame
    feature_columns: list[str]

    def setUp(self) -> None:
        # Create a dataframe with various column types for testing
        self.df = pd.DataFrame(
            {
                # Normal numeric columns
                "numeric_int": [1, 2, 3, 4, 5],
                "numeric_float": [1.1, 2.2, 3.3, 4.4, 5.5],
                # Mixed type columns
                "mixed_numbers": ["1", 2, "3", 4.0, 5],
                "mixed_with_text": ["1", "2", "text", "4", "5"],
                # All None columns
                "all_none_object": [None, None, None, None, None],
                # None with numeric values
                "some_none_numeric": [1.0, None, 3.0, None, 5.0],
                "some_none_string": ["1.0", None, "3.0", None, "5.0"],
                # Edge cases
                "empty_strings": ["", "", "", "", ""],
                "nan_values": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )

        # Explicitly set dtype for all_none_object to object
        self.df["all_none_object"] = self.df["all_none_object"].astype("object")

        # Feature columns for testing
        self.feature_columns = self.df.columns.tolist()

    def test_sanitize_numeric_columns_unchanged(self) -> None:
        """Test that numeric columns remain unchanged"""
        sanitized_df: pd.DataFrame = sanitize_feature_columns(self.df, ["numeric_int", "numeric_float"])

        # Check that values and dtypes are preserved
        pd.testing.assert_series_equal(sanitized_df["numeric_int"], self.df["numeric_int"])
        pd.testing.assert_series_equal(sanitized_df["numeric_float"], self.df["numeric_float"])

        # Ensure original df is not modified
        self.assertIsNot(sanitized_df, self.df)

    def test_sanitize_mixed_columns(self) -> None:
        """Test handling of columns with mixed types"""
        sanitized_df: pd.DataFrame = sanitize_feature_columns(self.df, ["mixed_numbers", "mixed_with_text"])

        # Check mixed_numbers is converted fully to numeric
        self.assertEqual(sanitized_df["mixed_numbers"].dtype.kind, "f")
        np.testing.assert_array_equal(sanitized_df["mixed_numbers"].to_numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])

        # Check mixed_with_text has non-numeric values as NaN
        self.assertEqual(sanitized_df["mixed_with_text"].dtype.kind, "f")
        expected = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0], name="mixed_with_text")
        pd.testing.assert_series_equal(sanitized_df["mixed_with_text"], expected, check_dtype=False)

    def test_sanitize_all_none_column(self) -> None:
        """Test handling of columns with all None values and object dtype"""
        # This was identified as a major bug case
        sanitized_df: pd.DataFrame = sanitize_feature_columns(self.df, ["all_none_object"])

        # Check that the dtype is converted from object to float64
        self.assertEqual(sanitized_df["all_none_object"].dtype, np.dtype("float64"))

        # All values should be NaN
        self.assertTrue(sanitized_df["all_none_object"].isna().all())

    def test_sanitize_some_none_columns(self) -> None:
        """Test handling of columns with some None values"""
        sanitized_df: pd.DataFrame = sanitize_feature_columns(self.df, ["some_none_numeric", "some_none_string"])

        # Check that both columns are converted to float64
        self.assertEqual(sanitized_df["some_none_numeric"].dtype, np.dtype("float64"))
        self.assertEqual(sanitized_df["some_none_string"].dtype, np.dtype("float64"))

        # Check values
        expected_numeric = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], name="some_none_numeric")
        expected_string = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], name="some_none_string")

        pd.testing.assert_series_equal(sanitized_df["some_none_numeric"], expected_numeric, check_dtype=False)
        pd.testing.assert_series_equal(sanitized_df["some_none_string"], expected_string, check_dtype=False)

    def test_sanitize_edge_cases(self) -> None:
        """Test handling of edge cases like empty strings and NaN values"""
        sanitized_df: pd.DataFrame = sanitize_feature_columns(self.df, ["empty_strings", "nan_values"])

        # Empty strings should be converted to NaN
        self.assertEqual(sanitized_df["empty_strings"].dtype, np.dtype("float64"))
        self.assertTrue(sanitized_df["empty_strings"].isna().all())

        # NaN values should remain NaN with float64 dtype
        self.assertEqual(sanitized_df["nan_values"].dtype, np.dtype("float64"))
        self.assertTrue(sanitized_df["nan_values"].isna().all())

    def test_empty_dataframe(self) -> None:
        """Test with an empty dataframe"""
        empty_df: pd.DataFrame = pd.DataFrame()
        result: pd.DataFrame = sanitize_feature_columns(empty_df, [])
        self.assertTrue(result.empty)
        self.assertIsNot(result, empty_df)  # Should return a copy

    def test_no_object_columns(self) -> None:
        """Test when no columns need sanitization"""
        numeric_df: pd.DataFrame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result: pd.DataFrame = sanitize_feature_columns(numeric_df, ["a", "b"])
        # Should return the same dataframe if no sanitization needed
        pd.testing.assert_frame_equal(result, numeric_df)
        self.assertIsNot(result, numeric_df)  # But should be a copy

    def test_columns_not_in_dataframe(self) -> None:
        """Test behavior when feature columns are not in the dataframe"""
        with pytest.raises(KeyError):
            sanitize_feature_columns(self.df, ["non_existent_column"])

    def test_single_value_per_type(self) -> None:
        """Test with single values of each type"""
        single_value_df: pd.DataFrame = pd.DataFrame(
            {
                "int_value": [42],
                "float_value": [3.14],
                "str_number": ["100"],
                "str_text": ["text"],
                "none_value": [None],
                "bool_value": [True],
            }
        )
        single_value_df["none_value"] = single_value_df["none_value"].astype("object")

        result: pd.DataFrame = sanitize_feature_columns(single_value_df, single_value_df.columns.tolist())

        # Check conversions
        self.assertEqual(result["int_value"].dtype.kind, "i")
        self.assertEqual(result["float_value"].dtype.kind, "f")
        self.assertEqual(result["str_number"].dtype.kind, "i")
        self.assertEqual(result["str_text"].dtype.kind, "f")  # Converted but with NaN
        self.assertEqual(result["none_value"].dtype.kind, "f")  # Should be float64 with NaN

        # Check values
        self.assertEqual(result["int_value"].iloc[0], 42)
        self.assertEqual(result["float_value"].iloc[0], 3.14)
        self.assertEqual(result["str_number"].iloc[0], 100)
        self.assertTrue(np.isnan(result["str_text"].iloc[0]))
        self.assertTrue(np.isnan(result["none_value"].iloc[0]))
