from typing import Optional, cast

import lightning as L
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.v2.autoencoder.column_normalization import ColumnNormalization

EmbeddingType = npt.NDArray[np.float64]


class AutoencoderModule(L.LightningModule):
    """
    Autoencoder for real estate investment data with a custom normalization layer.
    """

    def __init__(
        self,
        input_dim: int,
        means: list[float],
        stds: list[float],
        hidden_dims: list[int] = [64, 32],
        latent_dim: int = 16,
        numerical_columns_indices: Optional[list[int]] = None,
        feature_weights: Optional[list[float]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize the autoencoder.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden dimensions for the encoder
            latent_dim: Dimension of the latent space
            numerical_columns_indices: Indices of columns to normalize
            means: List of means for normalization
            stds: List of standard deviations for normalization
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        super().__init__()

        # Store hyperparameters
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.feature_weights = feature_weights
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.normalize_embeddings = normalize_embeddings
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.save_hyperparameters()

        # Create a tensor with feature weights
        self.feature_weights_tensor = torch.tensor(feature_weights)

        # Store for validation embeddings
        self.validation_embeddings: list[torch.Tensor] = []

        # Initialize normalizer if columns are specified
        if numerical_columns_indices is None:
            numerical_columns_indices = []

        self.normalizer = ColumnNormalization(numerical_columns_indices, means, stds)

        # Build encoder
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.LeakyReLU(self.leaky_relu_slope))
            encoder_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.LeakyReLU(self.leaky_relu_slope))
            decoder_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder with special handling for categorical features.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstructed_x, latent_representation)
        """
        # Normalize selected columns
        x_normalized = self.normalizer(x)

        # Encode
        z = self.encoder(x_normalized)

        # Normalize embeddings
        if self.normalize_embeddings:
            z = F.normalize(z, p=2, dim=1)

        # Decode - raw output
        decoded_raw = cast(torch.Tensor, self.decoder(z))

        # Create the final reconstruction with proper activations
        x_reconstructed = decoded_raw.clone()

        # Apply sigmoid activation only to categorical features
        categorical_indices = self._get_categorical_indices()
        if categorical_indices:
            x_reconstructed[:, categorical_indices] = torch.sigmoid(decoded_raw[:, categorical_indices])

        return x_reconstructed, z

    def _calculate_loss(self, batch: torch.Tensor, stage: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate reconstruction loss with different loss functions for
        numerical and categorical features, with per-feature weighting.

        Args:
            batch: Input batch
            stage: Training stage (train/val)

        Returns:
            Tuple of (combined_loss, embeddings)
        """
        x = batch

        # Create mask for NaN values
        mask = ~torch.isnan(x)
        self.log(f"{stage}/nan_count", (~mask).sum().float(), prog_bar=True)

        # fill NaN values with 0
        x = torch.where(mask, x, torch.tensor(0.0, device=x.device))

        x_reconstructed, z = self(x)
        x = cast(torch.Tensor, self.normalizer(x))

        # Move feature weights to the same device as x
        feature_weights = self.feature_weights_tensor.to(x.device)

        # Use normalized_columns_indices to identify numerical features
        numerical_indices = self.normalizer.normalized_columns_indices
        categorical_indices = self._get_categorical_indices()

        # Calculate loss for numerical features with weights
        if numerical_indices.shape[0] > 0:
            x_num = x[:, numerical_indices]
            x_reconstructed_num = x_reconstructed[:, numerical_indices]
            numerical_mask = mask[:, numerical_indices]
            numerical_weights = feature_weights[numerical_indices]

            # Apply weights to each feature's loss
            numerical_losses = F.mse_loss(x_reconstructed_num, x_num, reduction="none")
            weighted_numerical_losses = numerical_losses * numerical_weights.view(1, -1)
            masked_weighted_losses = weighted_numerical_losses * numerical_mask.float()

            # Normalize by mask and weights
            numerical_loss = (
                masked_weighted_losses.sum() / (numerical_mask.float() * numerical_weights.view(1, -1)).sum()
            )
        else:
            numerical_loss = torch.tensor(0.0, device=x.device)

        # Calculate loss for categorical features with weights
        if categorical_indices:
            x_cat = x[:, categorical_indices]
            x_reconstructed_cat = x_reconstructed[:, categorical_indices]
            categorical_mask = mask[:, categorical_indices]
            categorical_weights = feature_weights[categorical_indices]

            # Apply weights to each feature's loss
            categorical_losses = F.binary_cross_entropy(x_reconstructed_cat, x_cat, reduction="none")
            weighted_categorical_losses = categorical_losses * categorical_weights.view(1, -1)
            masked_weighted_losses = weighted_categorical_losses * categorical_mask.float()

            # Normalize by mask and weights
            categorical_loss = (
                masked_weighted_losses.sum() / (categorical_mask.float() * categorical_weights.view(1, -1)).sum()
            )
        else:
            categorical_loss = torch.tensor(0.0, device=x.device)

        combined_loss = numerical_loss + categorical_loss

        # Log individual losses
        self.log(f"{stage}/num_loss", numerical_loss, prog_bar=True)
        self.log(f"{stage}/cat_loss", categorical_loss, prog_bar=True)
        self.log(f"{stage}/loss", combined_loss, prog_bar=True)

        return combined_loss, z

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step logic.

        Args:
            batch: Input batch
            batch_idx: Index of the current batch

        Returns:
            Dictionary with loss and logs
        """
        loss, _ = self._calculate_loss(batch, "train")
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Validation step logic.

        Args:
            batch: Input batch
            batch_idx: Index of the current batch
        """
        _, z = self._calculate_loss(batch, "val")

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure optimizer.

        Returns:
            Optimizer
        """
        return optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor

        Returns:
            Latent representation
        """
        # Create mask for NaN values
        mask = ~torch.isnan(x)

        # fill NaN values with 0
        x = torch.where(mask, x, torch.tensor(0.0, device=x.device))

        x_normalized = self.normalizer(x)
        with torch.no_grad():
            z: torch.Tensor = self.encoder(x_normalized)
            if self.normalize_embeddings:
                z = F.normalize(z, p=2, dim=1)
        return z

    def _get_categorical_indices(self) -> list[int]:
        """
        Get indices of categorical features (those not being normalized).

        Returns:
            List of categorical feature indices
        """
        numerical_indices = self.normalizer.normalized_columns_indices.tolist()
        all_indices = set(range(self.input_dim))
        numerical_indices_set = set(numerical_indices)
        categorical_indices = list(all_indices - numerical_indices_set)
        return categorical_indices
