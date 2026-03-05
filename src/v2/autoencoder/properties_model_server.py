from typing import Any

import litserve as ls  # type: ignore
import numpy as np
import numpy.typing as npt
import torch

from src.v2.autoencoder.properties_embedding_model import PropertiesEmbeddingModel
from src.v2.config import get_config


class PropertiesEmbeddingAPI(ls.LitAPI):  # type: ignore
    def __init__(self, artifact_path: str) -> None:
        super().__init__()
        self.artifact_path = artifact_path

    def setup(self, device: str) -> None:
        self.model = PropertiesEmbeddingModel.load_from_checkpoint(self.artifact_path)

    def decode_request(self, request: dict[str, Any]) -> Any:
        return request["data"]

    def predict(self, x: Any) -> npt.NDArray[np.float64]:
        return self.model.encode(torch.tensor(x)).numpy()  # type: ignore

    def encode_response(self, output: Any) -> dict[str, Any]:
        return {"embedding": output}


if __name__ == "__main__":
    config = get_config().properties_embedding_model
    assert config.mlflow_artifact_path is not None, "MLflow artifact path must be set"
    api = PropertiesEmbeddingAPI(config.mlflow_artifact_path)
    server = ls.LitServer(api)
    server.run()
