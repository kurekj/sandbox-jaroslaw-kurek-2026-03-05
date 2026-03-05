import random
from typing import Optional

import lightning as L
import mlflow.artifacts
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, random_split

from src.v2.autoencoder.autoencoder_dataset import AutoencoderDataset
from src.v2.autoencoder.autoencoder_module import AutoencoderModule, EmbeddingType


class OffersEmbeddingModel:
    """
    Wrapper class for the real estate embedding model.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        logger: MLFlowLogger,
        numeric_cols: list[str],
        categorical_cols: list[str],
        feature_weights: dict[str, float],
        hidden_dims: list[int] = [64, 32],
        latent_dim: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 100,
        early_stop_patience: int = 10,
        min_delta: float = 1e-5,
        num_workers: int = 0,
        persistent_workers: bool = False,
        load_model: bool = False,
        artifact_path: Optional[str] = None,
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the embedding model.

        Args:
            data: Pandas DataFrame containing real estate data
            columns_to_normalize: List of column names to normalize
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names
            hidden_dims: List of hidden dimensions for the encoder
            latent_dim: Dimension of the latent space
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            early_stop_patience: Number of epochs to wait for improvement before stopping
        """
        # Set seed for reproducibility
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Create dataset
        self.dataset = AutoencoderDataset(data, numeric_cols, categorical_cols)

        # Convert feature weights dict to a list of weights with the same order as the dataset columns
        feature_weights_sorted = self.setup_feature_weights_list(feature_weights)


        if load_model:
            # TODO: test if it even works
            assert artifact_path is not None, "Artifact path must be provided when loading model."
            checkpoint_path = mlflow.artifacts.download_artifacts(artifact_path)
            self.model = AutoencoderModule.load_from_checkpoint(checkpoint_path)
        else:
            # Calculate means and std for normalization
            means = data[numeric_cols].mean().tolist()
            stds = data[numeric_cols].std().tolist()

            # Create model
            self.model = AutoencoderModule(
                input_dim=len(self.dataset.feature_names),
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                numerical_columns_indices=self.dataset.get_numeric_indices(),
                means=means,
                stds=stds,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                feature_weights=feature_weights_sorted,
                use_batch_norm=use_batch_norm,
                leaky_relu_slope=leaky_relu_slope,
                dropout=dropout,
            )

        # Store training parameters
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.mlflow_logger = logger
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.min_delta = min_delta

    def setup_feature_weights_list(self, feature_weights: dict[str, float]) -> list[float]:
        # get indices of the columns in the dataset
        # Convert feature weights dict to a list of weights with the same order as the dataset columns
        feature_weights_idx = [
            (self.dataset.get_column_index(col), feature_weights[col]) for col in feature_weights.keys()
        ]

        # sort the weights by the column index
        feature_weights_idx.sort(key=lambda x: x[0])

        sorted_weights = [weight for _, weight in feature_weights_idx]
        return sorted_weights

    def train(self, val_split: float = 0.2) -> L.Trainer:
        """
        Train the autoencoder.

        Args:
            val_split: Fraction of data to use for validation
        """
        # Split data into train and validation
        dataset_size = len(self.dataset)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

        # Configure training
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            min_delta=self.min_delta,
            patience=self.early_stop_patience,
            verbose=True,
            mode="min",
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            filename="offers_autoencoder-{epoch:02d}-{val/loss:.4f}",
            save_top_k=1,
            mode="min",
        )

        # Create trainer
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=self.mlflow_logger,
            accelerator="auto",  # Use GPU if available
            devices="auto",
        )

        # Train model
        trainer.fit(self.model, train_loader, val_loader)

        # Load best model
        self.model = AutoencoderModule.load_from_checkpoint(checkpoint_callback.best_model_path)

        # Set model to evaluation mode
        self.model.eval()

        return trainer

    def get_embeddings(self, data: pd.DataFrame) -> EmbeddingType:
        """
        Get embeddings for new data.

        Args:
            data: Pandas DataFrame containing real estate data

        Returns:
            Numpy array of embeddings
        """
        # Create dataset for new data
        new_dataset = AutoencoderDataset(
            data,
            numeric_cols=self.dataset.numeric_cols,
            categorical_cols=self.dataset.categorical_cols,
        )

        # Create dataloader for batch processing
        loader = DataLoader(
            new_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

        # Generate embeddings
        embeddings = []

        for batch in loader:
            with torch.no_grad():
                # Pass through encoder to get latent representation
                batch_emb = self.model.encode(batch)
                embeddings.append(batch_emb.numpy())

        return np.vstack(embeddings)
