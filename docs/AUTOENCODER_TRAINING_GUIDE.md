# Autoencoder Training Guide

This guide covers the complete process of training the properties autoencoder model, from dataset preparation to hyperparameter optimization and deployment considerations.

## Overview

The properties autoencoder is a neural network model designed to learn compressed representations (embeddings) of property features. These embeddings are used in the recommendation system to find similar properties and calculate recommendation scores.

## Prerequisites

Before starting the training process, ensure you have:

1. **MLFlow Server**: A running MLFlow server for experiment tracking and model artifact storage
2. **Database Access**: Connection to the properties database for data extraction
3. **Python Environment**: All required dependencies installed (see `pyproject.toml`)
4. **Computational Resources**: Sufficient memory and GPU (optional but recommended) for training

## Training Pipeline

### Step 1: Dataset Preparation

The first step is to create a training dataset from the current properties data.

#### Using the Notebook (Recommended for Exploration)

Use the interactive notebook `notebooks/offer_autoencoder/visible_properties_dataset.ipynb` to:

1. **Load Current Properties Data**: Extract visible properties from the database
2. **Data Filtering**: Apply date filters and quality checks
3. **Preprocessing**: Clean and transform the data for training
4. **Feature Engineering**: Create feature columns according to specifications
5. **Dataset Export**: Save the processed dataset as JSONL format

**Key Steps in the Notebook:**

```python
# Load and filter data
df = await load_current_properties_data()
df = df[df["create_date"] > "2022-01-01"]  # Adjust date filter as needed

# Preprocess the data
df = await preprocess_properties_data(df)

# Export to training dataset
df.to_json("data/autoencoder/visible_properties_dataset.jsonl", orient="records", lines=True)
```

#### Converting to Production Script

For automated/production training, convert the notebook logic into a standalone Python script:

1. Extract the data loading and preprocessing logic from the notebook
2. Create a script in `scripts/` directory (e.g., `create_training_dataset.py`)
3. Add command-line arguments for date ranges, filters, and output paths
4. Include logging and error handling for production use

### Step 2: Hyperparameter Optimization

Once the dataset is ready, use the Optuna-based hyperparameter optimization to find the best model configuration.

#### Running Hyperparameter Optimization

```bash
PYTHONPATH=$(pwd) python src/v2/autoencoder/optuna_hpo.py
```

#### Configuration

The optimization script (`src/v2/autoencoder/optuna_hpo.py`) automatically:

1. **Loads Dataset**: Reads from `data/autoencoder/visible_properties_dataset.jsonl`
2. **Creates Study**: Initializes or loads existing Optuna study
3. **Runs Trials**: Executes hyperparameter search (default: 50 trials, 3 parallel jobs)
4. **Logs Results**: Saves all experiments to MLFlow and study progress to SQLite

#### Hyperparameters Being Optimized

- **learning_rate**: Optimizer learning rate (1e-4 to 1e-2, log scale)
- **weight_decay**: L2 regularization strength (1e-8 to 1e-3, log scale)
- **batch_size**: Training batch size (512, 1024, 2048, 4096)
- **dropout**: Dropout rate for regularization (0.0 to 0.3)
- **leaky_relu_slope**: Negative slope for LeakyReLU activation (0.0 to 0.2)

#### Fixed Parameters (Optimized in Initial Experiments)

- **hidden_dims**: Architecture layers `[128, 64]` (found to be optimal)
- **latent_dim**: Latent space dimension `32` (found to be optimal)
- **use_batch_norm**: Batch normalization `True` (found to be optimal)

#### Customizing the Optimization

Modify the following parameters in `optuna_hpo.py` as needed:

```python
# Study configuration
study = optuna.create_study(
    study_name="your_study_name",           # Change study name
    storage="sqlite:///logs/your_study.db", # Change database path
    n_trials=50,                            # Increase number of trials
    n_jobs=4,                               # Adjust parallel jobs
)

# Training configuration
objective = partial(
    objective_partial,
    df=df[feature_columns],
    max_epochs=100,                         # Increase training epochs
    early_stop_patience=5,                  # Adjust early stopping
    num_workers=4,                          # Adjust data loading workers
)
```

### Step 3: Model Deployment

After optimization, the best model artifacts are automatically stored in MLFlow. To deploy the model:

1. **Retrieve Best Model**: Use MLFlow to find URL for the best performing model
2. **Model Integration**: Update the production configuration to use the new model URL (it will be downloaded automatically)
3. **Validation**: Test the model embeddings quality before full deployment
4. **Rollback Plan**: Keep the previous model version as backup

## Production Considerations

### Cyclical Re-training

For optimal performance in production, implement cyclical re-training:

#### Recommended Schedule

- **Monthly Re-training**: For high-volume, rapidly changing property markets
- **Quarterly Re-training**: For stable markets with moderate property turnover
- **Triggered Re-training**: When data distribution significantly changes

#### Re-training Pipeline

1. **Data Monitoring**: Track data drift and model performance metrics
2. **Automated Dataset Creation**: Schedule dataset preparation jobs
3. **Hyperparameter Optimization**: Run abbreviated optimization (fewer trials)
4. **Model Validation**: Compare new model performance against current production model (if dataset exists)
5. **Gradual Deployment**: A/B test the new model before full rollout

### Performance Monitoring

Track the following metrics for production models:

1. **Reconstruction Loss**: Monitor validation loss trends
2. **Embedding Quality**: Measure similarity preservation and clustering quality
3. **Recommendation Performance**: Track downstream recommendation metrics
4. **Training Time**: Monitor computational efficiency
5. **Resource Usage**: Track memory and GPU utilization

### Data Quality Considerations

Ensure training data quality by:

1. **Completeness Checks**: Verify all required features are present
2. **Outlier Detection**: Remove or handle extreme values appropriately
3. **Feature Distribution**: Monitor for significant distribution shifts
4. **Temporal Consistency**: Ensure data represents current market conditions
