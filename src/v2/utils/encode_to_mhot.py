from typing import Any

import pandas as pd
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer  # type: ignore


def _safe_convert(x: Any) -> list[str]:
    if (isinstance(x, float) and pd.isna(x)) or x is None:  # Handle NaN values
        return []
    elif isinstance(x, float):  # Handle single floats
        return [str(int(x))]
    elif isinstance(x, int):  # Handle single floats
        return [str(x)]
    elif isinstance(x, str):  # Handle single strings
        return [x]
    elif hasattr(x, "__iter__"):  # Keep lists as they are
        return [str(i) for i in x]
    else:
        raise ValueError(f"Unexpected value: {x} ({type(x)})")


def encode_to_mhot(df: pd.DataFrame, column: str, classes_column: str) -> pd.DataFrame:
    try:
        # Initialize the MultiLabelBinarizer
        # NOTE: convert classes to strings as safe_convert will convert all values to strings
        mlb = MultiLabelBinarizer(classes=[str(c) for c in df[classes_column].iloc[0]])

        converted_column = df[column].apply(_safe_convert)

        # Fit and transform the column
        mhot = mlb.fit_transform(converted_column)

        # Create a DataFrame with the multi-hot encoded column
        mhot_df = pd.DataFrame(mhot, columns=[(f"{column}_{cls}").lower() for cls in mlb.classes_])

        # Ensure the index matches the original DataFrame
        mhot_df.index = df.index

        # Drop existing multi-hot encoded columns if they exist
        existing_columns = set(df.columns)
        new_columns = set(mhot_df.columns)
        columns_to_drop = existing_columns.intersection(new_columns)
        if columns_to_drop:
            df = df.drop(columns=list(columns_to_drop))

        # Concatenate the original DataFrame with the multi-hot encoded DataFrame
        df = pd.concat([df, mhot_df], axis=1)
        return df
    except Exception as e:
        logger.error(f"Error while encoding {column} to multi-hot: {e}")
        raise e
