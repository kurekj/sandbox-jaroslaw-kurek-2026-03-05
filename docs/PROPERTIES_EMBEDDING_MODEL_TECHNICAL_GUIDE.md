# Properties Embedding Model (Autoencoder) - Technical Documentation

## Overview

The Properties Embedding Model is a neural network-based autoencoder specifically designed to create meaningful numerical representations (embeddings) of real estate properties. This model compresses high-dimensional property data into a lower-dimensional latent space while preserving the most important characteristics that define property similarity and relationships.

## Architecture Overview

The autoencoder follows a classic encoder-decoder architecture with several specialized components designed for real estate data:

```text
Input Features (N dimensions) → Encoder → Latent Space (32 dimensions) → Decoder → Reconstructed Features (N dimensions)
```

### Core Components

1. **Input Normalization Layer**: Selective normalization for numerical features
2. **Encoder Network**: Compresses input to latent representation
3. **Latent Space**: 32-dimensional embedding space (the size is parameterized and optimized)
4. **Decoder Network**: Reconstructs original features from embeddings
5. **Mixed Loss Functions**: Different loss functions for categorical vs numerical features

## Model Architecture Details

### 1. Input Processing and Feature Handling

#### Feature Types

The model handles two types of features:

- **Numerical Features**: Continuous values (area, price, coordinates, POI counts/distances)
- **Categorical Features**: Binary encoded features (kitchen type, facilities, quarters)

#### Feature Specifications

Features are defined with three key attributes:

- **Regular Expression Pattern**: Identifies which columns belong to this feature
- **Feature Type**: CATEGORICAL or NUMERICAL
- **Weight**: Importance factor for loss calculation (range: 0.3-2.0)

#### Key Feature Categories

```python
# High-importance features (weight 2.0)
- Location: lon, lat
- Size: area
- Pricing: normalize_price_m2

# Medium-importance features (weight 1.0-1.2)
- Property details: rooms, additional_area_type, normalize_price

# Lower-importance features (weight 0.3-0.8)
- POI distances/counts, facilities, building characteristics
```

### 2. Normalization Layer

The model uses a specialized `ColumnNormalization` layer that:

- **Selective Normalization**: Only normalizes specified numerical columns
- **Z-score Normalization**: `(value - mean) / std` for numerical features
- **Preservation**: Leaves categorical features unchanged

### 3. Encoder Architecture

The encoder compresses input data through multiple layers:

**Default Architecture:**

```text
Input (N features) 
    ↓
Linear Layer (N → 128)
    ↓
BatchNorm1d (optional, enabled by default)
    ↓
LeakyReLU (negative_slope=0.2)
    ↓
Dropout (rate=0.2)
    ↓
Linear Layer (128 → 64)
    ↓
BatchNorm1d (optional)
    ↓
LeakyReLU (negative_slope=0.2)
    ↓
Dropout (rate=0.2)
    ↓
Linear Layer (64 → 32) [Latent Space]
    ↓
L2 Normalization (optional, enabled by default)
```

**Key Design Choices:**

- **LeakyReLU Activation**: Prevents dying ReLU problem, allows small negative gradients
- **Batch Normalization**: Stabilizes training and accelerates convergence
- **Dropout**: Prevents overfitting by randomly setting neurons to zero during training
- **L2 Normalization**: Ensures embeddings are unit vectors, bounded similarity scores

### 4. Latent Space (Embedding Space)

**Dimensionality**: 32 dimensions (determined through hyperparameter optimization - HPO experiments tested dimensions of 8, 16, and 32, with 32 showing optimal performance)

**Properties**:

- **Normalized**: Embeddings are L2-normalized to unit vectors
- **Dense Representation**: Each dimension captures multiple property aspects
- **Similarity Preserving**: Similar properties have similar embeddings
- **Bounded**: Due to normalization, cosine similarity is bounded [-1, 1]

### 5. Decoder Architecture

The decoder mirrors the encoder in reverse:

```text
Latent Space (32 dimensions)
    ↓
Linear Layer (32 → 64)
    ↓
BatchNorm1d (optional)
    ↓
LeakyReLU (negative_slope=0.2)
    ↓
Dropout (rate=0.2)
    ↓
Linear Layer (64 → 128)
    ↓
BatchNorm1d (optional)
    ↓
LeakyReLU (negative_slope=0.2)
    ↓
Dropout (rate=0.2)
    ↓
Linear Layer (128 → N) [Raw Output]
    ↓
Selective Sigmoid (only for categorical features)
```

## Loss Function Design

The model uses a sophisticated loss function that accounts for different feature types and importance:

### Mixed Loss Function

```python
total_loss = weighted_numerical_loss + weighted_categorical_loss
```

#### Numerical Features Loss

- **Loss Function**: Mean Squared Error (MSE)
- **Application**: For continuous features like area, price, coordinates
- **Rationale**: MSE penalizes large errors more than small ones, appropriate for continuous values

#### Categorical Features Loss

- **Loss Function**: Binary Cross-Entropy (BCE)
- **Application**: For binary-encoded categorical features
- **Activation**: Sigmoid applied to decoder output for categorical features
- **Rationale**: BCE is optimal for binary classification problems

#### Feature Weighting System

Each feature has an assigned weight that reflects its importance:

```python
# Examples of feature weights:
location_weight = 2.0      # High importance - location is crucial
area_weight = 2.0          # High importance - size matters
price_weight = 2.0         # High importance - pricing information
facilities_weight = 0.5    # Lower importance - nice-to-have features
poi_distance_weight = 0.3  # Lower importance - convenience factors
```

**Weighted Loss Calculation**:

```python
feature_loss = base_loss * feature_weight
total_weighted_loss = sum(feature_loss for all features) / sum(all_weights)
```

## Training Process

### 1. Data Preparation

#### Stratified Sampling

The model implements stratified sampling based on property prices to ensure balanced representation:

- **Strata Column**: `normalize_price_m2` (normalized price per square meter)
- **Strata Bins**: 10 bins (configurable)
- **NaN Handling**: Properties with missing price data are assigned to a separate bin
- **Sampling Strategy**: Weighted random sampling with replacement to oversample underrepresented price ranges

#### Train/Validation Split

- **Default Split**: 80% training, 20% validation
- **Stratification**: Maintains price distribution in both sets
- **Random Seed**: Ensures reproducible splits

### 2. Optimization

#### Optimizer Configuration

- **Algorithm**: Adam optimizer
- **Learning Rate**: Optimized through hyperparameter search (typically 1e-4 to 1e-2)
- **Weight Decay**: L2 regularization (typically 1e-8 to 1e-3)

#### Training Configuration

- **Batch Size**: Configurable (512, 1024, 2048, 4096)
- **Max Epochs**: Configurable (typically 50-100)
- **Early Stopping**: Patience-based on validation loss
- **Model Checkpointing**: Saves best model based on validation loss

### 3. Regularization Techniques

#### Dropout

- **Rate**: 0.2 (configurable 0.0-0.3)
- **Application**: Applied after each hidden layer
- **Purpose**: Prevents overfitting by encouraging model robustness

#### Batch Normalization

- **Application**: After linear layers, before activation
- **Benefits**: Stabilizes gradients, accelerates training, acts as regularization

#### Weight Decay

- **L2 Regularization**: Penalizes large weights
- **Purpose**: Prevents overfitting, encourages simpler solutions

## Embedding Quality and Properties

### What the Embeddings Capture

The 32-dimensional embeddings encode multiple aspects of properties:

1. **Geographic Clustering**: Properties in similar locations have similar embeddings
2. **Price Similarity**: Properties with similar price ranges cluster together
3. **Size Relationships**: Properties with similar areas group together
4. **Feature Combinations**: Complex relationships between multiple features

**Note**: The 32-dimensional space was selected through systematic hyperparameter optimization, where dimensions of 8, 16, and 32 were evaluated. The 32-dimensional space provided the best balance between representation capacity and computational efficiency.

### Embedding Normalization

**L2 Normalization**: All embeddings are normalized to unit vectors

```python
embedding_normalized = embedding / ||embedding||_2
```

**Benefits**:

- **Bounded Similarity**: Cosine similarity ∈ [-1, 1]
- **Scale Invariance**: Magnitude doesn't affect similarity
- **Geometric Interpretation**: Embeddings lie on unit hypersphere

### Similarity Computation

**Cosine Similarity**: Used for comparing properties

```python
similarity = dot_product(embedding_a, embedding_b)
# Since embeddings are normalized: similarity = cos(angle)
```

**Interpretation**:

- **1.0**: Identical properties
- **0.0**: Orthogonal (unrelated) properties  
- **-1.0**: Opposite properties

## Model Evaluation and Validation

### Training Metrics

1. **Reconstruction Loss**: How well the model reconstructs input features
2. **Validation Loss**: Performance on unseen data
3. **Feature-wise Loss**: Individual loss for each feature type

### Quality Assessment

1. **Reconstruction Quality**: Compare reconstructed vs original features
2. **Embedding Clustering**: Visualize embeddings in 2D/3D space
3. **Similarity Preservation**: Check if similar properties have similar embeddings
4. **Downstream Performance**: Evaluate recommendation system performance

## Configuration and Hyperparameters

### Model Architecture Config

```python
AutoencoderConfig(
    hidden_dims=[128, 64],      # Encoder/decoder layer sizes
    latent_dim=32,              # Embedding dimension
    use_batch_norm=True,        # Enable batch normalization
    leaky_relu_slope=0.2,       # LeakyReLU negative slope
    dropout=0.2                 # Dropout rate
)
```

### Training Config

```python
TrainingConfig(
    max_epochs=100,             # Maximum training epochs
    early_stop_patience=10,     # Early stopping patience
    batch_size=1024,            # Training batch size
    val_split=0.2,              # Validation split ratio
    strata_column="normalize_price_m2",  # Stratification column
    strata_bins=10              # Number of strata bins
)
```

### Optimizer Config

```python
OptimizerConfig(
    learning_rate=1e-3,         # Adam learning rate
    weight_decay=1e-5           # L2 regularization strength
)
```

## Usage Examples

### Training a New Model

```python
from src.v2.autoencoder.properties_embedding_model import PropertiesEmbeddingModel

# Load and prepare data
df = load_properties_data()

# Configure model
model_config = AutoencoderConfig(latent_dim=32, hidden_dims=[128, 64])
optimizer_config = OptimizerConfig(learning_rate=1e-3)
training_config = TrainingConfig(max_epochs=50, batch_size=1024)

# Train model
embedding_model = PropertiesEmbeddingModel()
trainer = embedding_model.train(
    data=df,
    model_config=model_config,
    optimizer_config=optimizer_config,
    training_config=training_config,
    logger=mlflow_logger
)
```

### Generating Embeddings

```python
# Load trained model
model = PropertiesEmbeddingModel.load_from_checkpoint("path/to/checkpoint")

# Generate embeddings for new properties
embeddings = model.get_embeddings(new_properties_df)

# Each property now has a 32-dimensional embedding
print(embeddings.shape)  # (n_properties, 32)
```

### Computing Property Similarity

```python
import numpy as np

# Get embeddings for two properties
property_a_embedding = embeddings[0]
property_b_embedding = embeddings[1]

# Compute cosine similarity (since embeddings are normalized)
similarity = np.dot(property_a_embedding, property_b_embedding)
print(f"Similarity: {similarity:.3f}")  # Value between -1 and 1
```

## Advanced Features

### Stratified Sampling Implementation

The model includes sophisticated stratified sampling to handle imbalanced data:

```python
def _create_stratified_sampler(subset, dataset, strata_column, strata_bins):
    # Extract stratification values
    strata_values = extract_column_values(subset, strata_column)
    
    # Handle NaN values by creating separate bin
    nan_mask = np.isnan(strata_values)
    
    # Create bins for non-NaN values
    if any(~nan_mask):
        bins = np.linspace(min(non_nan_values), max(non_nan_values), strata_bins)
        bin_indices = digitize(strata_values, bins)
    
    # Assign NaN values to separate bin
    bin_indices[nan_mask] = strata_bins - 1
    
    # Calculate inverse frequency weights
    bin_counts = count_samples_per_bin(bin_indices)
    weights = 1.0 / bin_counts[bin_indices]
    
    return WeightedRandomSampler(weights, replacement=True)
```

### Feature Weight Management

The model automatically handles feature weights based on configuration:

```python
def _setup_feature_weights_list(feature_weights, dataset):
    # Map feature names to dataset column indices
    feature_weights_indexed = [
        (dataset.get_column_index(col), weight) 
        for col, weight in feature_weights.items()
    ]
    
    # Sort by column index to match dataset order
    feature_weights_indexed.sort(key=lambda x: x[0])
    
    return [weight for _, weight in feature_weights_indexed]
```

## Performance Considerations

### Memory Efficiency

- **Batch Processing**: Large datasets processed in batches
- **Persistent Workers**: Reuse data loading workers between epochs
- **GPU Acceleration**: Automatic GPU utilization when available

### Training Speed

- **Vectorized Operations**: Efficient PyTorch tensor operations
- **Mixed Precision**: Optional for faster training on modern GPUs
- **Data Loading**: Multi-worker data loading for reduced I/O bottleneck

### Scalability

- **Configurable Architecture**: Easily adjust model size for different datasets
- **Feature Selection**: Include/exclude features based on data availability
- **Batch Size Tuning**: Optimize batch size for available memory

## Integration with Recommendation System

The embeddings produced by this model are directly used in the recommendation system:

1. **Property Encoding**: Each property gets a 32-dimensional embedding
2. **User Preference Modeling**: User history is represented through property embeddings
3. **Similarity Computation**: Dot product between normalized embeddings
4. **Attention Mechanism**: Weighted combination of user history based on similarity

This creates a seamless pipeline from raw property data to personalized recommendations, with the autoencoder serving as the crucial feature learning component that enables effective property-to-property comparisons.
