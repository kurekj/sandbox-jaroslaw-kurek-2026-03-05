# mypy: ignore-errors
import asyncio

import pandas as pd
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from src.v2.autoencoder.feature_specs import get_feature_columns
from src.v2.autoencoder.preprocess_data import load_current_apartment_data, preprocess_properties_data
from src.v2.autoencoder.properties_embedding_model import PropertiesEmbeddingModel
from src.v2.config import get_config

COLLECTION_NAME = "apartments_embeddings_raw"


async def load_data() -> pd.DataFrame:
    df = await load_current_apartment_data()
    df = await preprocess_properties_data(df)

    return df


async def main() -> None:
    config = get_config().properties_embedding_model
    assert config.mlflow_artifact_path is not None, "MLflow artifact path must be set"

    model = PropertiesEmbeddingModel.load_from_checkpoint(config.mlflow_artifact_path)

    df = await load_data()
    feature_columns = get_feature_columns(df)[0]

    logger.info(f"Getting embeddings for {len(df)} properties")
    embeddings = model.get_embeddings(df[feature_columns], batch_size=1024)

    # # add normalized lat and lon to the embeddings
    # scaler = StandardScaler()
    # lon_df = scaler.fit_transform(df[["lon"]])
    # lat_df = scaler.fit_transform(df[["lat"]])

    # # decrease the weight of lat and lon
    # coord_weight = 0.5
    # lon_df = lon_df * coord_weight
    # lat_df = lat_df * coord_weight

    # embeddings = np.concatenate([embeddings, lon_df, lat_df], axis=1)

    # add embeddings as a column to df
    df["embeddings"] = embeddings.tolist()

    client = QdrantClient(url="http://localhost:6333")
    logger.debug("Connected to Qdrant")

    if client.collection_exists(COLLECTION_NAME):
        logger.info(f"Collection {COLLECTION_NAME} exists, deleting")
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.EUCLID),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=10),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            ),
        ),
    )
    logger.info(f"Created collection {COLLECTION_NAME}")

    # convert lat and lon columns to a single column called location (it should be a dict)
    # it is needed for the qdrant to index geo data
    df["location"] = df.apply(lambda row: {"lat": row["lat"], "lon": row["lon"]}, axis=1)
    df.drop(columns=["lat", "lon"], inplace=True)

    batch_size = 1000
    total_points = len(df)

    for i in tqdm(range(0, total_points, batch_size), desc="Upserting data points"):
        batch_df = df.iloc[i : min(i + batch_size, total_points)]
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=[
                PointStruct(id=row["property_id"], vector=row["embeddings"], payload=row.to_dict())
                for _, row in batch_df.iterrows()
            ],
        )
    logger.info("All data points have been upserted successfully.")


if __name__ == "__main__":
    asyncio.run(main())
