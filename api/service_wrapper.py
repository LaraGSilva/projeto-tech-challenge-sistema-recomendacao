import torch
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from shared.utils.config import load_config
from typing import List, Dict, Any

class RecommendationService:
    """Serviço responsável por carregar modelos do MLflow e realizar inferências.

    Este serviço encapsula o carregamento do modelo PyTorch a partir do Model Registry
    do MLflow e gerencia os mapeamentos de ID para índices necessários para a 
    inferência em tempo real.
    """

    def __init__(self, model_name: str = "Neural-NeuMF-MLP", alias: str = "production") -> None:
        """Inicializa o serviço carregando o modelo e artefatos do MLflow."""
        self.cfg: Dict[str, Any] = load_config()

        # Carregamento do modelo
        model_uri: str = f"models:/{model_name}@{alias}"
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.to("cpu")
        self.model.eval()

        # Carregamento de artefatos (mappings)
        client: MlflowClient = MlflowClient()
        model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
        run_id: str = model_version.run_id

        full_path: str = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/mappings.pkl" # Ajustado para pkl conforme seu build_features
        )

        with open(full_path, "rb") as f:
            data = mlflow.pytorch.pickle.load(f) # Ou apenas pickle.load

        self.user_to_idx: Dict[int, int] = data["user_to_idx"]
        self.idx_to_item: Dict[int, int] = data["idx_to_item"]

    def _get_max_item_idx(self) -> int:
        """Determina dinamicamente o número máximo de itens baseado na estrutura do modelo."""
        # Se for NeuMF_Light, usa item_embed. Se for Retail, usa item_mf_embed.
        if hasattr(self.model, 'item_embed'):
            return self.model.item_embed.num_embeddings
        elif hasattr(self.model, 'item_mf_embed'):
            return self.model.item_mf_embed.num_embeddings
        return len(self.idx_to_item)

    def get_recommendations(self, visitorid: int, k: int) -> List[int]:
        """Gera recomendações de itens usando o método .recommend() unificado do Mixin."""
        
        # A nova estrutura de modelos permite usar o método .recommend() injetado pelo Mixin!
        # Isso simplifica drasticamente a lógica de inferência.
        
        # Injeção dinâmica necessária para o .recommend() (do RecommenderMixin) funcionar
        self.model.user_to_idx = self.user_to_idx
        self.model.idx_to_item = self.idx_to_item
        
        # Chama a interface unificada
        return self.model.recommend(visitorid, k)