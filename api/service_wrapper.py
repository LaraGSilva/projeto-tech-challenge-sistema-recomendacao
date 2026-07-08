import torch
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from shared.utils.config import load_config
import os

class RecommendationService:

    def __init__(self,
                 model_name="Neural-NeuMF-MLP",
                 alias="production"):

        self.cfg = load_config()

        model_uri = f"models:/{model_name}@{alias}"

        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.to("cpu")
        self.model.eval()

        client = MlflowClient()

        model_version = client.get_model_version_by_alias(
            name=model_name,
            alias=alias
        )

        run_id = model_version.run_id

        full_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/mappings.npy"
        )

        data = np.load(full_path, allow_pickle=True).item()

        self.user_to_idx = data["user_to_idx"]
        self.item_to_idx = data["item_to_idx"]
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}

    def get_recommendations(self, visitorid: int, k: int) -> list:
        if visitorid not in self.user_to_idx:
            return []
        
        u_idx = self.user_to_idx[visitorid]
        item_indices = list(self.item_to_idx.values())
        
        with torch.no_grad():
            u_t = torch.full((len(item_indices),), u_idx, dtype=torch.long)
            i_t = torch.tensor(item_indices, dtype=torch.long)
            scores = self.model(u_t, i_t).numpy()
            
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[idx] for idx in top_indices]