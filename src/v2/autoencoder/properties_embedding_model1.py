import random
from typing import Optional

import lightning as L
import mlflow.artifacts
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split

from src.v2.autoencoder.autoencoder_dataset import AutoencoderDataset
from src.v2.autoencoder.autoencoder_module import AutoencoderModule, EmbeddingType
from src.v2.autoencoder.feature_specs import get_feature_columns, get_feature_weights
from src.v2.config import get_config


class AutoencoderConfig(BaseModel):
    """
    Configuration for the autoencoder model architecture.

    This class defines the structure and hyperparameters for the neural network
    architecture used in the autoencoder model.
    """

    hidden_dims: list[int] = [64, 32]
    """Dimensions of hidden layers in the encoder and decoder networks."""
    latent_dim: int = 16
    """Dimension of the latent space (embedding size)."""
    use_batch_norm: bool = True
    """Whether to use batch normalization in the network layers."""
    leaky_relu_slope: float = 0.2
    """Negative slope coefficient for LeakyReLU activation functions."""
    dropout: float = 0.2
    """Dropout rate applied after activation functions for regularization."""


class OptimizerConfig(BaseModel):
    """
    Configuration for the training optimizer.

    This class defines the hyperparameters for the optimizer used during model training.
    """

    learning_rate: float = 1e-3
    """Learning rate for the optimizer."""
    weight_decay: float = 1e-5
    """L2 regularization factor (weight decay) for the optimizer."""


class TrainingConfig(BaseModel):
    """
    Configuration for the training process.

    This class defines parameters that control the training process, including
    early stopping criteria, data splitting, and hardware utilization.
    """

    max_epochs: int = 100
    """Maximum number of training epochs."""
    early_stop_patience: int = 10
    """Number of epochs with no improvement after which training will be stopped."""
    min_delta: float = 1e-5
    """Minimum change in monitored quantity to qualify as improvement for early stopping."""
    strata_column: Optional[str] = "normalize_price_m2"
    """Column name to use for stratified sampling, or None to disable stratification."""
    strata_bins: int = 10
    """Number of bins to use for stratified sampling."""
    model_prefix: str = "properties_autoencoder"
    """Prefix for saved model checkpoint filenames."""
    seed: int = 42
    """Random seed for reproducibility."""
    batch_size: int = 64
    """Number of samples per batch during training and validation."""
    num_workers: int = 0
    """Number of subprocesses to use for data loading."""
    persistent_workers: bool = False
    """Whether to maintain worker processes between data loading iterations."""
    val_split: float = 0.2
    """Fraction of data to use for validation."""


class PropertiesEmbeddingModel:
    """
    Wrapper class for the real estate embedding model.

    This class handles training, loading, and using an autoencoder-based embedding
    model specifically designed for real estate property data. It provides methods
    for training the model, generating embeddings for properties, and loading
    pre-trained models from checkpoints.
    """

    model: Optional[AutoencoderModule] = None
    """The underlying autoencoder model, or None if not initialized."""

    @classmethod
    def load_from_checkpoint(cls, artifact_path: str) -> "PropertiesEmbeddingModel":
        """
        Load a pre-trained model from a checkpoint.

        Args:
            artifact_path: Path to the MLflow artifact containing the model checkpoint.

        Returns:
            A PropertiesEmbeddingModel instance with the loaded model
        """
        instance = cls.__new__(cls)

        if artifact_path.startswith("mlflow-artifacts:"):
            # Download checkpoint
            checkpoint_path = mlflow.artifacts.download_artifacts(artifact_path)
            logger.debug(f"Downloaded model checkpoint from {artifact_path}")
        else:
            logger.debug(f"Using local model checkpoint at {artifact_path}")
            checkpoint_path = artifact_path
        instance.model = AutoencoderModule.load_from_checkpoint(checkpoint_path)

        return instance

    def train(
        self,
        data: pd.DataFrame,
        logger: MLFlowLogger,
        model_config: AutoencoderConfig,
        optimizer_config: OptimizerConfig,
        training_config: TrainingConfig,
        additional_callbacks: list[L.Callback] = [],
        model: Optional[AutoencoderModule] = None,
    ) -> L.Trainer:
        """
        Train an autoencoder model for property embeddings.

        This method sets up and trains an autoencoder neural network to create embeddings for property data.
        The training process includes automatic splitting into train and validation sets, optional stratified sampling,
        normalization of numerical features, and early stopping to prevent overfitting.

        Notes
            The method automatically:
            - Extracts categorical and numerical features from the data
            - Calculates feature weights
            - Normalizes numerical features
            - Sets up stratified sampling if a strata column is provided
            - Implements early stopping and model checkpointing
            - Loads the best model based on validation loss

        Args:
            data (pd.DataFrame): The dataset containing property features to train on.
            logger (MLFlowLogger): Lightning logger for tracking metrics and model parameters.
            model_config (AutoencoderConfig): Configuration for the autoencoder model.
            optimizer_config (OptimizerConfig): Configuration for the optimizer.
            training_config (TrainingConfig): Configuration for training.
            additional_callbacks (list[L.Callback], optional): Additional Lightning callbacks to use during training,
                by default [].
            model (Optional[AutoencoderModule], optional): Pre-initialized autoencoder model to use instead of creating
                a new one, by default None.

        Returns:
            L.Trainer: The Lightning trainer instance after completion of training.
        """
        # Set hyperparameters not passed directly to a model
        logger.log_hyperparams(
            {
                "seed": training_config.seed,
                "batch_size": training_config.batch_size,
                "patience": training_config.early_stop_patience,
                "min_delta": training_config.min_delta,
                "max_epochs": training_config.max_epochs,
                "val_split": training_config.val_split,
                "strata_column": training_config.strata_column,
                "strata_bins": training_config.strata_bins,
                "dataset_length": len(data),
            }
        )

        # Set seed for reproducibility
        self._set_seed(training_config.seed)

        # Get feature columns and weights if not explicitly provided
        _, categorical_cols, numeric_cols = get_feature_columns(data)

        feature_weights = get_feature_weights(data)

        # Create dataset
        dataset = AutoencoderDataset(
            data=data,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        # Convert feature weights dict to a list of weights with the same order as the dataset columns
        feature_weights_sorted = self._setup_feature_weights_list(feature_weights=feature_weights, dataset=dataset)

        # Calculate means and std for normalization
        means = data[numeric_cols].mean().tolist()
        stds = data[numeric_cols].std().tolist()

        # Create model
        self.model = model or AutoencoderModule(
            input_dim=len(dataset.feature_names),
            hidden_dims=model_config.hidden_dims,
            latent_dim=model_config.latent_dim,
            numerical_columns_indices=dataset.get_numeric_indices(),
            means=means,
            stds=stds,
            learning_rate=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            feature_weights=feature_weights_sorted,
            use_batch_norm=model_config.use_batch_norm,
            leaky_relu_slope=model_config.leaky_relu_slope,
            dropout=model_config.dropout,
        )

        # Split data into train and validation
        dataset_size = len(dataset)
        val_size = int(dataset_size * training_config.val_split)
        train_size = dataset_size - val_size

        generator = torch.Generator().manual_seed(training_config.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
            persistent_workers=training_config.persistent_workers,
            sampler=self._create_stratified_sampler(
                subset=train_dataset,
                dataset=dataset,
                strata_column=training_config.strata_column,
                strata_bins=training_config.strata_bins,
            )
            if training_config.strata_column
            else None,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
            persistent_workers=training_config.persistent_workers,
        )

        # Configure training
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            min_delta=training_config.min_delta,
            patience=training_config.early_stop_patience,
            verbose=True,
            mode="min",
            strict=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            filename="epoch={epoch:02d}-val_loss={val/loss:.4f}",
            save_top_k=1,
            mode="min",
            auto_insert_metric_name=False,
        )

        # Create trainer
        trainer = L.Trainer(
            max_epochs=training_config.max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback, *additional_callbacks],
            logger=logger,
            #accelerator="auto",  # Use GPU if available
            accelerator="gpu",  # wymusza użycie GPU
            #devices="auto",
            devices=1,  # liczba kart, np. 2
            #strategy="ddp",  # zalecane przy treningu na wielu GPU
            #strategy=None
        )

        # Train model
        trainer.fit(self.model, train_loader, val_loader)

        # Load best model
        self.model = AutoencoderModule.load_from_checkpoint(checkpoint_callback.best_model_path)

        # Set model to evaluation mode
        self.model.eval()

        return trainer

    @staticmethod
    def _create_stratified_sampler(
        subset: Subset[torch.Tensor],
        dataset: AutoencoderDataset,
        strata_column: str,
        strata_bins: int,
    ) -> WeightedRandomSampler:
        assert strata_column is not None, "Strata column must be provided for stratified sampling."
        strata_values = np.array([i[dataset.column_name_to_idx[strata_column]].item() for i in subset])  # type: ignore

        # Check for and handle NaN values
        nan_mask = np.isnan(strata_values)
        if nan_mask.any():
            # Create a separate bin for NaN values
            non_nan_values = strata_values[~nan_mask]

            # Only calculate bins for non-NaN values
            if len(non_nan_values) > 0:
                # Create bins (edges) covering the full range of non-NaN strata_values
                bins = np.linspace(non_nan_values.min(), non_nan_values.max(), strata_bins)

                # Determine bin index for each non-NaN sample
                bin_indices = np.digitize(non_nan_values, bins) - 1

                # Map NaN values to a separate bin index (last bin)
                full_bin_indices = np.zeros_like(strata_values, dtype=int)
                full_bin_indices[~nan_mask] = bin_indices
                full_bin_indices[nan_mask] = strata_bins - 1  # Assign NaNs to the last bin

                # Count how many samples fall into each bin (including the NaN bin)
                bin_counts = np.bincount(full_bin_indices, minlength=strata_bins)
            else:
                # All values are NaN, create a single bin
                full_bin_indices = np.zeros_like(strata_values, dtype=int)
                bin_counts = np.array([len(strata_values)])
        else:
            # No NaN values, proceed as normal
            bins = np.linspace(strata_values.min(), strata_values.max(), strata_bins + 1)
            full_bin_indices = np.digitize(strata_values, bins) - 1
            bin_counts = np.bincount(full_bin_indices, minlength=strata_bins)

        # Calculate weight for each sample: inverse of the bin frequency
        weights = 1.0 / bin_counts[full_bin_indices]

        # Convert weights to a tensor
        weights = torch.DoubleTensor(weights)

        # Create the sampler. Setting replacement=True allows oversampling the underrepresented samples.
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return sampler

    @staticmethod
    def _setup_feature_weights_list(feature_weights: dict[str, float], dataset: AutoencoderDataset) -> list[float]:
        # Get indices of the columns in the dataset
        feature_weights_idx = [(dataset.get_column_index(col), feature_weights[col]) for col in feature_weights.keys()]

        # Sort the weights by the column index
        feature_weights_idx.sort(key=lambda x: x[0])

        sorted_weights = [weight for _, weight in feature_weights_idx]
        return sorted_weights

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input data into a latent representation.

        Args:
            x (torch.Tensor): Input data to encode.

        Returns:
            torch.Tensor: Latent representation of the input data.
        """
        assert self.model is not None, (
            "Model must be set before calling encode. Train the model or load from checkpoint."
        )
        self.model.eval()
        with torch.inference_mode():
            return self.model.encode(x)

    def get_embeddings(
        self,
        data: pd.DataFrame,
    ) -> EmbeddingType:
        assert self.model is not None, (
            "Model must be set before calling get_embeddings. Train the model or load from checkpoint."
        )
        config = get_config().properties_embedding_model
        _, categorical_cols, numeric_cols = get_feature_columns(data)
        # Create dataset for new data
        new_dataset = AutoencoderDataset(
            data,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        # Create dataloader for batch processing
        loader = DataLoader(
            new_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
        )

        # Generate embeddings
        embeddings = []

        for batch in loader:
            # Pass through encoder to get latent representation
            batch_emb = self.encode(batch)
            embeddings.append(batch_emb.numpy())

        return np.vstack(embeddings)

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
