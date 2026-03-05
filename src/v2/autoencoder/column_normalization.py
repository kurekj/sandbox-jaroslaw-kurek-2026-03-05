import torch
import torch.nn as nn


class ColumnNormalization(nn.Module):
    """
    Normalize the specified columns of a tensor.
    """

    normalized_columns_indices: torch.Tensor
    """Normalized columns indices."""
    means: torch.Tensor
    stds: torch.Tensor

    def __init__(
        self,
        normalized_columns_indices: list[int],
        means: list[float],
        stds: list[float],
    ):
        """
        normalized_columns_indices: List or tensor of indices to normalize.
        means: List of means corresponding to the norm_indices.
        stds: List of standard deviations corresponding to the norm_indices.
        """
        super().__init__()
        # assert norm_indices, means and stds have the same length
        assert len(normalized_columns_indices) == len(means) == len(stds), (
            "norm_indices, means and stds must have the same length."
        )

        self.register_buffer("normalized_columns_indices", torch.tensor(normalized_columns_indices))
        self.register_buffer("means", torch.tensor(means))
        # NOTE: add a small value to avoid division by zero
        self.register_buffer("stds", torch.tensor(stds).add(1e-5))

        self.training = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Make a copy to avoid in-place modifications
        x_norm = x.clone()
        # Normalize only the specified columns
        # x_norm[:, norm_indices] will have shape (batch, num_norm_cols)
        x_norm[:, self.normalized_columns_indices] = (
            x_norm[:, self.normalized_columns_indices] - self.means
        ) / self.stds
        return x_norm
