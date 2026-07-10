from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel
from torch import nn


class NeuMFConfig(BaseModel):
    """Configuração centralizada para o modelo NeuMF usando Pydantic."""

    n_users: int
    n_items: int
    n_categorias: int
    item_to_cat_array: List[int]
    mf_dim: int = 16
    mlp_dim: int = 32
    categoria_dim: int = 8
    hidden_dims: List[int] = [128, 64]
    dropout: float = 0.1


class BaseRecommender(nn.Module):
    """Interface unificada para modelos de recomendação."""

    def __init__(self) -> None:
        super().__init__()
        # Inicializamos como None para permitir a injeção dinâmica posterior
        self.user_to_idx: Optional[Dict[int, int]] = None
        self.idx_to_item: Optional[Dict[int, int]] = None

    def recommend(self, user_id: int, k: int, **kwargs: Any) -> List[int]:
        raise NotImplementedError("Subclasses devem implementar recommend()")


class NeuMF_Light(BaseRecommender):  # noqa: N801
    """Modelo Neural Matrix Factorization para o dataset RetailRocket."""

    def __init__(self, config: NeuMFConfig) -> None:
        super().__init__()

        self.user_mf_embed = nn.Embedding(config.n_users, config.mf_dim)
        self.item_mf_embed = nn.Embedding(config.n_items, config.mf_dim)
        self.user_mlp_embed = nn.Embedding(config.n_users, config.mlp_dim)
        self.item_mlp_embed = nn.Embedding(config.n_items, config.mlp_dim)
        self.cat_mlp_embed = nn.Embedding(config.n_categorias, config.categoria_dim)

        self.register_buffer(
            "item_categoria_idx",
            torch.tensor(config.item_to_cat_array, dtype=torch.long),
        )

        in_dim = config.mlp_dim * 2 + config.categoria_dim
        layers: List[nn.Module] = []
        for h_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.prediction_layer = nn.Linear(config.mf_dim + config.hidden_dims[-1], 1)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_mf = self.user_mf_embed(user_idx)
        item_mf = self.item_mf_embed(item_idx)
        gmf_vector = user_mf * item_mf

        user_mlp = self.user_mlp_embed(user_idx)
        item_mlp = self.item_mlp_embed(item_idx)
        cat_mlp = self.cat_mlp_embed(self.item_categoria_idx[item_idx])

        mlp_vector = self.mlp(torch.cat([user_mlp, item_mlp, cat_mlp], dim=-1))
        return self.prediction_layer(torch.cat([gmf_vector, mlp_vector], dim=-1)).view(
            -1
        )

    def recommend(self, user_id: int, k: int, **kwargs: Any) -> List[int]:
        """Gera recomendações ordenadas para um usuário."""
        # Acesso seguro aos atributos injetados
        if self.user_to_idx is None or self.idx_to_item is None:
            raise AttributeError("Mapeamentos não foram injetados no modelo.")

        if user_id not in self.user_to_idx:
            return []

        u_idx = self.user_to_idx[user_id]

        # Cria tensores para todos os itens disponíveis
        all_item_idxs = torch.arange(len(self.idx_to_item))
        user_t = torch.full((len(all_item_idxs),), u_idx)

        # Inferência
        with torch.no_grad():
            scores = self.forward(user_t, all_item_idxs).cpu().numpy()

        # Ordena os scores
        top_indices = np.argsort(scores)[::-1]

        # Filtra apenas os índices que existem no dicionário idx_to_item
        valid_recommendations = []
        for i in top_indices:
            idx = int(i)
            if idx in self.idx_to_item:
                valid_recommendations.append(self.idx_to_item[idx])

            if len(valid_recommendations) == k:
                break

        return valid_recommendations
