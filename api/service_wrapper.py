import os
from typing import Any, Dict, List

import mlflow
import numpy as np
import torch

from shared.ml.models import NeuMF_Light, NeuMFConfig


class RecommendationService:
    """Serviço para carregar modelos de recomendação e realizar inferências."""

    def __init__(self, model_name: str, alias: str) -> None:
        """Inicializa o serviço carregando o modelo e os mapeamentos."""

        # 1. Obter URI e baixar artefatos do MLflow
        model_uri: str = f"models:/{model_name}@{alias}"
        local_path: str = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

        # 2. Carregar o arquivo de mapeamentos (mappings.npy)
        mapping_path = os.path.join(local_path, "mappings.npy")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"Arquivo de mapeamentos não encontrado: {mapping_path}"
            )

        data = np.load(mapping_path, allow_pickle=True).item()
        self.mappings: Dict[str, Any] = data

        print(f"DEBUG: Chaves encontradas no mappings: {list(self.mappings.keys())}")

        # 3. Preparar configuração para instanciar o modelo manualmente
        # Nota: Ajuste os valores se a sua configuração real for diferente
        user_to_idx = self.mappings["user_to_idx"]
        item_to_idx = self.mappings["item_to_idx"]

        config = NeuMFConfig(
            n_users=len(user_to_idx),
            n_items=len(item_to_idx),
            n_categorias=1,  # Ajuste se o seu modelo usar múltiplas categorias
            item_to_cat_array=[0] * len(item_to_idx),
        )

        # 4. Instanciar o modelo explicitamente (evita erro de atributos ausentes)
        self.model = NeuMF_Light(config)

        # 5. Carregar os pesos (ajuste o caminho se necessário)
        # O MLflow geralmente salva o estado em 'model.pth' ou similar dentro do artefato
        state_dict_path = os.path.join(local_path, "model.pth")
        if os.path.exists(state_dict_path):
            self.model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

        self.model.eval()

        # 6. Injeção de dependência dos mapeamentos necessários
        self.model.user_to_idx = user_to_idx
        # Invertendo item_to_idx para idx_to_item para o modelo mapear o resultado
        self.model.idx_to_item = {v: k for k, v in item_to_idx.items()}

    def get_recommendations(self, user_id: int, k: int) -> List[int]:
        """Gera recomendações para um usuário específico."""
        return self.model.recommend(user_id, k)
