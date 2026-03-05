"""
Hyperparameter Optimization for Properties Autoencoder using Optuna
... (reszta docstring bez zmian) ...
"""

import os

# --- POPRAWKA 1 (z poprzedniego błędu CUDA): Debugowanie ---
# To spowolni trening, ale da precyzyjny traceback
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from functools import partial
import datetime
import matplotlib.pyplot as plt

import time
import mlflow
import optuna
import pandas as pd
# from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.loggers import CSVLogger
from optuna.integration import PyTorchLightningPruningCallback

from src.v2.autoencoder.feature_specs import get_feature_columns
from src.v2.autoencoder.properties_embedding_model import (
    AutoencoderConfig,
    OptimizerConfig,
    PropertiesEmbeddingModel,
    TrainingConfig,
)
from src.v2.config import get_config

import torch

torch.set_float32_matmul_precision("high")  # lub "medium"s

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

    # liczba dostępnych kart
    num_gpus = torch.cuda.device_count()
    # wybierz GPU na podstawie numeru próby
    gpu_idx = trial.number % num_gpus
    # ustaw widoczne tylko to GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    #torch.cuda.set_device(gpu_idx)

    # logger = MLFlowLogger(
    #    experiment_name="visible_properties_autoencoder",
    #    tracking_uri=str(get_config().mlflow.uri),
    #    log_model="all",
    #    tags={
    #        "hpo": "optuna",
    #        "optuna_study_name": trial.study.study_name,
    #        "optuna_trial_number": str(trial.number),
    #        "feature_specs": get_config().properties_embedding_model.feature_spec_type,
    #    },
    # )

    # Logger Optuny:
    logger = CSVLogger(
        save_dir=logs_base_dir,  # użyj katalogu dla danego uruchomienia
        name=f"optuna_trial_{trial.number}"
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
        devices=[gpu_idx],  # konkretna karta GPU dla tego procesu
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

    # ===  CZYŚCIMY GPU I PAMIĘĆ ===
    try:
        # Zakończ loggera (np. CSVLogger)
        if hasattr(trainer.logger, "finalize"):
            trainer.logger.finalize("success")
    except Exception:
        pass

    # Usuń ciężkie obiekty Lightninga i Pythona
    # del model, trainer
    del embedding_model, trainer
    import gc
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # === KONIEC CZYSZCZENIA ===

    return val_loss


if __name__ == "__main__":
    import multiprocessing as mp

    # --- POPRAWKA 1 (z błędu SIGSEGV): Ustaw metodę startową na 'spawn' ZANIM cokolwiek innego się stanie ---
    mp.set_start_method("spawn", force=True)

    import faulthandler

    faulthandler.dump_traceback_later(timeout=300, repeat=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # −−−−−−−−−−−−−−−− logowanie do pliku i konsoli −−−−−−−−−−−−−−−−
    import sys
    import datetime
    import logging  # Import modułu logging

    # --- Ustaw katalog na logi z timestampem  ---
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logs_base_dir = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(logs_base_dir, exist_ok=True)

    # katalog i plik logu
    LOG_DIR = os.path.join(CURRENT_DIR, "optuna_train_logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(
        LOG_DIR,
        f"optuna_train_logs_{timestamp}.log"
    )

    # --- POPRAWKA 2 (z błędu SIGSEGV): Konfiguracja modułu logging zamiast tee ---
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logging.getLogger("optuna").setLevel(logging.INFO)
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    logging.info(f"Logi będą zapisywane do: {LOG_FILE}")
    logging.info(f"Katalog logów: {LOG_DIR}")
    logging.info(f"Metoda startowa multiprocessing: {mp.get_start_method()}")
    # --- Koniec Poprawki 2 ---

    logging.info("Loading data...")

    # NOTE: Adjust the path to your dataset as needed
    df = pd.read_json(
        os.path.join(CURRENT_DIR, *[".."] * 3, "data", "autoencoder", "visible_properties_dataset.jsonl"),
        lines=True,
    )

    logging.info("Data loaded.")

    feature_columns = get_feature_columns(df)[0]

    objective = partial(
        objective_partial,
        df=df[feature_columns],
        max_epochs=75,
        early_stop_patience=3,
        min_delta=5e-4,
        num_workers=2,
        #num_workers=0,
        model_prefix="visible_properties_autoencoder",
    )

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    study = optuna.create_study(
        study_name="visible_properties_autoencoder_all_features_new_weights",
        storage=f"sqlite:///{os.path.join(logs_base_dir, 'visible_properties_autoencoder.db')}",
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
    )

    # rozpoczęcie pomiaru czasu ---
    start_time = time.time()

    # NOTE: adjust n_trials and n_jobs as needed
    study.optimize(
        objective,
        n_trials=200,
        n_jobs=2,
        show_progress_bar=True)

    # --- nowy kod: koniec pomiaru czasu i zapisanie wyników ---
    elapsed = time.time() - start_time
    logging.info(f"Total optimization time: {elapsed / 60:.2f} minutes")

    # zapis wyników Optuny do pliku Excel
    results_df = study.trials_dataframe()

    # Konwersja trwania na sekundy
    results_df["duration_seconds"] = results_df["duration"] * 24 * 60 * 60

    # Oblicz łączny czas (suma trwania wszystkich prób)
    # total_training_seconds = results_df["duration_seconds"].sum()

    total_script_seconds = time.time() - start_time  # koniec pomiaru
    logging.info(f"Całkowity czas skryptu: {total_script_seconds / 60:.2f} minut")

    # Utwórz dodatkowy arkusz z podsumowaniem
    summary_df = pd.DataFrame({
        "metric": ["total_training_seconds"],
        "value": [total_script_seconds]
    })

    # Określ najlepszą próbę i jej katalog
    best_trial = study.best_trial
    trial_dir = os.path.join(logs_base_dir, f"optuna_trial_{best_trial.number}", "version_0")

    # Utwórz DataFrame z informacją o najlepszym trialu
    best_trial_info = pd.DataFrame({
        "metric": ["best_trial_number", "best_trial_dir"],
        "value": [best_trial.number, trial_dir]
    })

    # Zapisz wszystko do Excela w katalogu logs_base_dir
    results_path = os.path.join(logs_base_dir, "optuna_study_results.xlsx")
    with pd.ExcelWriter(results_path) as writer:
        results_df.to_excel(writer, sheet_name="trials", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        best_trial_info.to_excel(writer, sheet_name="best_trial_info", index=False)

    logging.info(f"Zapisano wyniki i podsumowanie do {results_path}")

    # Print best trial
    logging.info("Best trial:")
    trial = study.best_trial
    logging.info(f"  Value: {trial.value}")
    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")

    # identyfikacja najlepszego trialu
    best_trial = study.best_trial
    trial_dir = os.path.join(logs_base_dir, f"optuna_trial_{best_trial.number}", "version_0")
    metrics_file = os.path.join(trial_dir, "metrics.csv")

    # Wypisz na ekran ścieżkę i numer najlepszego trialu
    logging.info(f"Najlepszy trial ma numer: {best_trial.number}")
    logging.info(f"Ścieżka do katalogu najlepszego trialu: {trial_dir}")

    # Wczytanie pliku z metrykami
    metrics_df = pd.read_csv(metrics_file)

    # wybierz wiersze, w których wartości strat nie są NaN
    train_loss = metrics_df[['step', 'train/loss']].dropna()
    val_loss = metrics_df[['step', 'val/loss']].dropna()

    # ujednolicenie nazwy kolumny z wartością strat
    train_loss.rename(columns={'train/loss': 'value'}, inplace=True)
    val_loss.rename(columns={'val/loss': 'value'}, inplace=True)

    # rysowanie wykresu
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss['step'], train_loss['value'], label='Train loss', marker='o')
    plt.plot(val_loss['step'], val_loss['value'], label='Validation loss', marker='s')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Learning curves for the best trial')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # zapis wykresu
    plot_path = os.path.join(logs_base_dir, "best_trial_learning_curves.png")
    plt.savefig(plot_path, dpi=300)

    logging.info(f"Zapisano krzywe uczenia do: {plot_path}")