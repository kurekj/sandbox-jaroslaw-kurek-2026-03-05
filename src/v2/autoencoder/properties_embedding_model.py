import os
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
    hidden_dims: list[int] = [64, 32]
    latent_dim: int = 16
    use_batch_norm: bool = True
    leaky_relu_slope: float = 0.2
    dropout: float = 0.2


class OptimizerConfig(BaseModel):
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class TrainingConfig(BaseModel):
    max_epochs: int = 100
    early_stop_patience: int = 10
    min_delta: float = 1e-5
    strata_column: Optional[str] = "normalize_price_m2"
    strata_bins: int = 10
    model_prefix: str = "properties_autoencoder"
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 0
    persistent_workers: bool = False
    val_split: float = 0.2
    devices: int | list[int] = 1
    """GPU device(s) to use. Set to a single int or a list of indices."""


class PropertiesEmbeddingModel:
    model: Optional[AutoencoderModule] = None

    @classmethod
    def load_from_checkpoint(cls, artifact_path: str) -> "PropertiesEmbeddingModel":
        instance = cls.__new__(cls)
        if artifact_path.startswith("mlflow-artifacts:"):
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

        self._set_seed(training_config.seed)
        _, categorical_cols, numeric_cols = get_feature_columns(data)
        feature_weights = get_feature_weights(data)

        dataset = AutoencoderDataset(
            data=data,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        feature_weights_sorted = self._setup_feature_weights_list(
            feature_weights=feature_weights, dataset=dataset
        )

        means = data[numeric_cols].mean().tolist()
        stds = data[numeric_cols].std().tolist()

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

        dataset_size = len(dataset)
        val_size = int(dataset_size * training_config.val_split)
        train_size = dataset_size - val_size

        generator = torch.Generator().manual_seed(training_config.seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )

        # Zapisywanie zbiorów treningowego i walidacyjnego do plików
        # Pobierz indeksy oryginalnych wierszy danych
        train_indices = getattr(train_dataset, "indices", None)
        val_indices = getattr(val_dataset, "indices", None)
        # Upewnij się, że te indeksy istnieją
        if train_indices is not None and val_indices is not None:
            train_df = data.iloc[train_indices].copy()
            val_df = data.iloc[val_indices].copy()
            # Katalog docelowy dla plików
            #save_dir = "logs"
            save_dir = getattr(logger, "log_dir", "logs")
            print(f"Dane treningowe i walidacyjne zostaną zapisane w katalogu: {save_dir}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Zapisz w formacie pickle
            train_df.to_pickle(os.path.join(save_dir, "train_data.pkl"))
            val_df.to_pickle(os.path.join(save_dir, "val_data.pkl"))

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

        # Single-GPU trainer for one trial
        trainer = L.Trainer(
            max_epochs=training_config.max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback, *additional_callbacks],
            logger=logger,
            accelerator="gpu",
            #devices=1,  # jedna karta na proces/trial
            devices=training_config.devices,  # jedno GPU lub lista indeksów
            log_every_n_steps=1,
            deterministic=True,  # stabilniejsze zachowanie
            num_sanity_val_steps=0,  # skraca start i eliminuje zbędny walidacyjny pass
        )

        trainer.fit(self.model, train_loader, val_loader)

        # Load best model
        self.model = AutoencoderModule.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
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
        strata_values = np.array(
            [i[dataset.column_name_to_idx[strata_column]].item() for i in subset]
        )  # type: ignore

        nan_mask = np.isnan(strata_values)
        if nan_mask.any():
            non_nan_values = strata_values[~nan_mask]
            if len(non_nan_values) > 0:
                bins = np.linspace(non_nan_values.min(), non_nan_values.max(), strata_bins)
                bin_indices = np.digitize(non_nan_values, bins) - 1
                full_bin_indices = np.zeros_like(strata_values, dtype=int)
                full_bin_indices[~nan_mask] = bin_indices
                full_bin_indices[nan_mask] = strata_bins - 1
                bin_counts = np.bincount(full_bin_indices, minlength=strata_bins)
            else:
                full_bin_indices = np.zeros_like(strata_values, dtype=int)
                bin_counts = np.array([len(strata_values)])
        else:
            bins = np.linspace(strata_values.min(), strata_values.max(), strata_bins + 1)
            full_bin_indices = np.digitize(strata_values, bins) - 1
            bin_counts = np.bincount(full_bin_indices, minlength=strata_bins)

        weights = 1.0 / bin_counts[full_bin_indices]
        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return sampler

    @staticmethod
    def _setup_feature_weights_list(
        feature_weights: dict[str, float], dataset: AutoencoderDataset
    ) -> list[float]:
        feature_weights_idx = [
            (dataset.get_column_index(col), feature_weights[col]) for col in feature_weights.keys()
        ]
        feature_weights_idx.sort(key=lambda x: x[0])
        sorted_weights = [weight for _, weight in feature_weights_idx]
        return sorted_weights

    def encode(self, x: torch.Tensor) -> torch.Tensor:
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

        new_dataset = AutoencoderDataset(
            data,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        loader = DataLoader(
            new_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
        )

        embeddings = []
        for batch in loader:
            batch_emb = self.encode(batch)
            embeddings.append(batch_emb.numpy())
        return np.vstack(embeddings)

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
