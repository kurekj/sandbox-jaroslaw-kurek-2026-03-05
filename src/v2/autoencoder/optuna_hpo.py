"""
Hyperparameter Optimization for Properties Autoencoder using Optuna

This script performs hyperparameter optimization for the properties autoencoder model
using Optuna, a hyperparameter optimization framework. The optimization process aims
to find the best combination of hyperparameters to minimize validation loss.

Key Features:
- Uses Optuna for systematic hyperparameter search
- Integrates with MLFlow for experiment tracking and artifact storage
- Supports parallel optimization with multiple trials
- Implements early stopping with PyTorch Lightning pruning callbacks
- Persists study results in SQLite database for resumable optimization

Usage:
    python optuna_hpo.py

The script will:
1. Load the visible properties dataset from data/autoencoder/visible_properties_dataset.jsonl
2. Create or load an existing Optuna study from SQLite database
3. Run hyperparameter optimization trials (default: 10 trials, 2 parallel jobs)
4. Log all experiments and model artifacts to MLFlow server
5. Print the best trial results upon completion

Hyperparameters being optimized:
- learning_rate: Learning rate for the optimizer (1e-4 to 1e-2, log scale)
- weight_decay: L2 regularization strength (1e-8 to 1e-3, log scale)
- batch_size: Training batch size (512, 1024, 2048, 4096)
- dropout: Dropout rate for regularization (0.0 to 0.3)
- leaky_relu_slope: Negative slope for LeakyReLU activation (0.0 to 0.2)

Fixed parameters (commented out after initial experiments showed optimal values):
- hidden_dims: Architecture layers [128, 64] (option 0 was best)
- latent_dim: Latent space dimension (32 was optimal)
- use_batch_norm: Batch normalization usage (True was optimal)

Artifacts:
- All training artifacts, models, and metrics are automatically logged to MLFlow
- Study progress is saved in logs/visible_properties_autoencoder.db
- Best model checkpoints are preserved for later use

Note:
- Ensure MLFlow server is running and accessible via the configured URI
    before starting the optimization process.
- Adjust the study name, database path, number of trials, and other parameters as needed for your environment.
"""

import os
from functools import partial

import mlflow
import optuna
import pandas as pd
from lightning.pytorch.loggers import MLFlowLogger
from optuna.integration import PyTorchLightningPruningCallback

from src.v2.autoencoder.feature_specs import get_feature_columns
from src.v2.autoencoder.properties_embedding_model import (
    AutoencoderConfig,
    OptimizerConfig,
    PropertiesEmbeddingModel,
    TrainingConfig,
)
from src.v2.config import get_config

mlflow.enable_system_metrics_logging()  # type: ignore


def objective_partial(
    trial: optuna.Trial,
    df: pd.DataFrame,
    max_epochs: int = 50,
    early_stop_patience: int = 5,
    min_delta: float = 5e-4,
    num_workers: int = 2,
    model_prefix: str = "properties_autoencoder_apartments",
) -> float:
    # Define hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

    hidden_dims_mapping = {
        0: [128, 64],
        1: [128, 64, 32],
        2: [64, 32],
    }
    # hidden_dims_option = trial.suggest_categorical("hidden_dims_option", list(hidden_dims_mapping.keys()))
    hidden_dims = hidden_dims_mapping[0]
    # latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32])
    latent_dim = 32
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    leaky_relu_slope = trial.suggest_float("leaky_relu_slope", 0.00, 0.2)
    # use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    use_batch_norm = True

    logger = MLFlowLogger(
        experiment_name="visible_properties_autoencoder",
        tracking_uri=str(get_config().mlflow.uri),
        log_model="all",
        tags={
            "hpo": "optuna",
            "optuna_study_name": trial.study.study_name,
            "optuna_trial_number": str(trial.number),
            "feature_specs": get_config().properties_embedding_model.feature_spec_type,
        },
    )

    autoencoder_config = AutoencoderConfig(
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        leaky_relu_slope=leaky_relu_slope,
    )

    optimizer_config = OptimizerConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    training_config = TrainingConfig(
        max_epochs=max_epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        min_delta=min_delta,
        num_workers=num_workers,
        model_prefix=model_prefix,
    )

    embedding_model = PropertiesEmbeddingModel()
    trainer = embedding_model.train(
        data=df,
        model_config=autoencoder_config,
        optimizer_config=optimizer_config,
        training_config=training_config,
        logger=logger,
        additional_callbacks=[PyTorchLightningPruningCallback(trial, monitor="val/loss")],
    )

    # Get the best validation loss
    val_loss = trainer.callback_metrics["val/loss"].item()

    return val_loss


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    print("Loading data...")

    # NOTE: Adjust the path to your dataset as needed
    df = pd.read_json(
        os.path.join(CURRENT_DIR, *[".."] * 3, "data", "autoencoder", "visible_properties_dataset.jsonl"),
        lines=True,
    )

    print("Data loaded.")

    feature_columns = get_feature_columns(df)[0]

    objective = partial(
        objective_partial,
        df=df[feature_columns],
        max_epochs=75,
        early_stop_patience=3,
        min_delta=5e-4,
        num_workers=2,
        model_prefix="visible_properties_autoencoder",
    )

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create Optuna study and optimize
    study = optuna.create_study(
        study_name="visible_properties_autoencoder_all_features_new_weights",
        storage="sqlite:///logs/visible_properties_autoencoder.db",  # NOTE: where to store the study
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
    )

    # NOTE: adjust n_trials and n_jobs as needed
    study.optimize(objective, n_trials=50, n_jobs=3)

    # Print best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
