from functools import lru_cache

import pandas as pd
import torch
from torch.utils.data import Dataset


class AutoencoderDataset(Dataset[torch.Tensor]):
    """
    Dataset for real estate investment data.
    """

    numeric_cols: list[str]
    categorical_cols: list[str]
    feature_names: list[str]
    X: torch.Tensor
    column_name_to_idx: dict[str, int]

    def __init__(
        self,
        data: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ):
        """
        Initialize the dataset.

        Args:
            data: Pandas DataFrame containing real estate data
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names
        """
        if data.empty:
            raise ValueError("DataFrame is for the AutoencoderDataset is empty!")

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.feature_names = data.columns.tolist()

        self.X = torch.from_numpy(data.to_numpy()).float()

        # Map column names to indices for normalization
        self.column_name_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}

    def __len__(self) -> int:
        """Return the number of samples in dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample by index."""
        return self.X[idx]

    def get_column_index(self, column_name: str) -> int:
        """
        Get the index of a column by name.

        Args:
            column_name: Name of the column

        Returns:
            Index of the column
        """
        return self.column_name_to_idx[column_name]

    @lru_cache()
    def get_numeric_indices(self) -> list[int]:
        """
        Get indices of columns to normalize.


        Returns:
            List of column indices
        """
        return [self.column_name_to_idx[col] for col in self.numeric_cols if col in self.column_name_to_idx]

    @lru_cache()
    def get_categorical_indicies(self) -> list[int]:
        """
        Get indices of categorical columns.

        Returns:
            List of column indices
        """
        return [self.column_name_to_idx[col] for col in self.categorical_cols if col in self.column_name_to_idx]
