# Web Recommendation System v2 for RynekPierwotny.pl

## Overview

This is the next-generation recommendation system for RynekPierwotny.pl, built using modern machine learning techniques and designed for high performance and scalability. The system uses an autoencoder-based approach to generate property embeddings and compute similarity scores between users and properties.

**Version Status**: This repository contains two versions of the recommendation system:

- **v1** (Legacy): Flask-based API with different ML approaches - see `src/v1/` and `src/README_v1.md`
- **v2** (Current): FastAPI-based system with autoencoder embeddings - see `src/v2/` and this README

## Architecture

The system consists of several key components:

1. **Autoencoder Model**: Generates 32-dimensional embeddings for properties
2. **Scoring Engine**: Computes similarity scores between users and properties
3. **Caching Layer**: Multi-tier Redis caching for optimal performance
4. **REST API**: FastAPI endpoints for score calculation and cache management
5. **Background Workers**: Celery workers for async cache warming

## Project Structure

```plaintext
├── src/v2/                          # Main application code
│   ├── api/                         # FastAPI application
│   │   ├── app.py                   # Main API application
│   │   ├── models/                  # Pydantic request/response models
│   │   ├── services/                # Business logic services
│   │   └── tasks.py                 # Celery background tasks
│   ├── model/                       # ML model components
│   │   ├── properties_embedding_model.py  # Autoencoder implementation
│   │   └── optuna_hpo.py            # Hyperparameter optimization
│   ├── utils/                       # Utility modules
│   │   ├── cache_utils.py           # Caching utilities
│   │   ├── sentinel_cache.py        # Redis/Sentinel cache implementation
│   │   └── prefill_cache.py         # Cache warming utilities
│   └── config.py                    # Application configuration
├── docs/                            # Documentation
├── notebooks/                       # Jupyter notebooks for analysis
├── data/                           # Training and evaluation data
├── containers/                     # Docker configuration
│   ├── docker-compose.yaml        # Local development setup
│   └── ae_api.Dockerfile          # API container image
└── tests/                         # Test suite
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Redis (for caching)
- PostgreSQL (for data storage)

### Local Development

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd web_recommendation_rp
   ```

2. **Set up environment**:

   ```bash
   # Copy and configure environment variables
   cp config/.env.example config/.env
   # Edit config/.env with your database and Redis credentials
   ```

3. **Run with Docker Compose**:

   ```bash
   docker compose -f containers/docker-compose.yaml -p ae_api up --build
   ```

   This will start:
   - Redis server (port 6379)
   - FastAPI application (port 8001)
   - Celery worker for background tasks

### Production Deployment

For production environments, the system is deployed on **Kubernetes (k8s)**. Check [Helm Charts repository](https://git.rynekpierwotny.com/property-group-data-science/helm-charts).

## API Usage

The v2 API provides several endpoints for recommendation scoring and cache management:

### Calculate Recommendation Scores

```bash
# Calculate scores for user-property pairs
curl -X POST "http://localhost:8001/calculate_scores" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"user_id": "user123", "property_id": 456},
      {"user_id": "user123", "property_id": 789}
    ]
  }'
```

**Response**:

```json
{
  "scores": [
    {"user_id": "user123", "property_id": 456, "score": 0.85},
    {"user_id": "user123", "property_id": 789, "score": 0.72}
  ]
}
```

### Cache Management

```bash
# Start cache prefill (background task)
curl -X POST "http://localhost:8001/prefill_cache/start" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"overwrite_visible_properties": true}'

# Check task status
curl -X GET "http://localhost:8001/prefill_cache/state?task_id=TASK_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Autoencoder Training Guide](docs/AUTOENCODER_TRAINING_GUIDE.md)**: Complete guide for training the embedding model
- **[Technical Architecture Guide](docs/PROPERTIES_EMBEDDING_MODEL_TECHNICAL_GUIDE.md)**: Deep dive into the autoencoder implementation
- **[Caching Mechanism Guide](docs/CACHING_MECHANISM_GUIDE.md)**: Detailed explanation of the caching system
- **[How Recommendation Scores Work](docs/HOW_RECOMMENDATION_SCORES_WORK.md)**: User-friendly explanation of the scoring algorithm
- **[Redis Sentinel Guide](docs/REDIS_SENTINEL_GUIDE.md)**: High availability Redis setup and implementation

## Model Training

The system uses an autoencoder neural network to generate property embeddings:

1. **Hyperparameter Optimization**: Use `optuna_hpo.py` to find optimal model parameters
2. **Model Training**: Train the autoencoder using PyTorch Lightning
3. **Evaluation**: Assess model performance using various metrics
4. **Deployment**: Update production models with new embeddings

For detailed training instructions, see the [Autoencoder Training Guide](docs/AUTOENCODER_TRAINING_GUIDE.md).

### Model Loading Options

The system supports multiple approaches for loading trained models:

#### MLflow Server (Recommended)

By default, the system loads models from an MLflow server using artifact paths like `mlflow-artifact:/path/to/model`.

#### Local Checkpoint Fallback

If the MLflow server fails or is unavailable, you can specify a local checkpoint path directly. This is demonstrated in the Docker Compose setup:

```yaml
volumes:
  - ../models/epoch=34-val_loss=0.0858.ckpt:/app/models/epoch=34-val_loss=0.0858.ckpt:ro
```

The `PropertiesEmbeddingModel.load_from_checkpoint()` method automatically detects whether the path is an MLflow artifact or a local file path.

#### Kubernetes Deployment Options

For Kubernetes deployments, you have several options for model storage:

1. **Volume Mounting** (Recommended):

   ```yaml
   volumes:
   - name: model-storage
     persistentVolumeClaim:
       claimName: models-pvc
   ```

2. **Image Embedding** (Not Recommended):
   - Rebuild the container image with model weights included
   - **Note**: This approach is not recommended as model weights typically change more frequently than code and should remain independent for better maintainability and deployment flexibility

The volume mounting approach provides better separation of concerns and allows for easier model updates without code changes.

## Performance & Caching

The system implements sophisticated caching strategies:

- **Multi-layered caching**: Property data (7 days TTL), user data (24 hours TTL), scores (24 hours TTL)
- **Intelligent prefill**: Background cache warming for frequently accessed data
- **Bulk operations**: Efficient batch processing to minimize database queries
- **Redis Sentinel support**: High availability caching with automatic failover

For detailed caching information, see the [Caching Mechanism Guide](docs/CACHING_MECHANISM_GUIDE.md).

## Configuration

The system uses Pydantic-based configuration with environment variable support:

```python
# Example configuration in config/.env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=recommendation_db
REDIS_HOST=localhost
REDIS_PORT=6379
API_KEYS=["your-secret-api-key"]
CACHE_DEFAULT_TTL=43200
```

## Development

### Setting up Development Environment

```bash
# Install dependencies
poetry install --with dev,training,plots

# Run tests
pytest

# Format code
ruff format

# Type checking
mypy src/
```

### Running Jupyter Notebooks

The project includes Jupyter notebooks for data analysis and model experimentation:

```bash
# Start Jupyter server
jupyter lab

# Navigate to notebooks/ directory
```

## Migration from v1

If you're migrating from the legacy v1 system:

1. **API Changes**: Update client code to use new FastAPI endpoints
2. **Authentication**: Implement API key-based authentication
3. **Response Format**: Adapt to new JSON response structures
4. **Caching**: Leverage new caching capabilities for better performance

## Contributing

1. Follow the existing code style (Ruff formatting)
2. Add tests for new functionality
3. Update documentation when adding features
4. Ensure all CI checks pass
