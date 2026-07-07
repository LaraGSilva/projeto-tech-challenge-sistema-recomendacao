import torch
import numpy as np
import mlflow
import mlflow.pytorch
from shared.utils.config import load_config

class RecommendationService:
    def __init__(self, model_name: str = "Neural-NeuMF-MLP", stage: str = "Production"):
        """
        Serviço de recomendação que carrega modelo e artefatos diretamente do MLflow.
        """
        self.cfg = load_config()
        
        # 1. Carrega o modelo PyTorch registrado no MLflow
        # A URI aponta para o registro centralizado
        model_uri = f"models:/{model_name}@Production"
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.to("cpu")
        self.model.eval()
        
        # 2. Carrega os mapeamentos a partir do artefato registrado no MLflow
        # Usamos o mlflow para baixar o artefato 'mappings.npy' associado ao modelo
        artifacts_path = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/mappings.npy")
        data = np.load(artifacts_path, allow_pickle=True).item()
        
        self.user_to_idx = data['user_to_idx']
        self.item_to_idx = data['item_to_idx']
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}

    def get_recommendations(self, visitorid: int, k: int) -> list:
        if visitorid not in self.user_to_idx:
            return []
        
        u_idx = self.user_to_idx[visitorid]
        item_indices = list(self.item_to_idx.values())
        
        with torch.no_grad():
            u_t = torch.full((len(item_indices),), u_idx, dtype=torch.long)
            i_t = torch.tensor(item_indices, dtype=torch.long)
            
            # Executa a inferência com o modelo carregado
            scores = self.model(u_t, i_t).numpy()
            
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[idx] for idx in top_indices]