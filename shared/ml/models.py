from typing import Any, Dict, List

import numpy as np
import torch
from pydantic import BaseModel
from torch import nn


class NeuMFConfig(BaseModel):
    """Configuração centralizada para o modelo NeuMF usando Pydantic.

    Attributes:
        n_users (int): Número total de usuários únicos.
        n_items (int): Número total de itens únicos.
        n_categorias (int): Número total de categorias.
        item_to_cat_array (List[int]): Mapeamento estático de índice de item para categoria.
        mf_dim (int): Dimensão do embedding para o ramo de Fatoração de Matriz (GMF).
        mlp_dim (int): Dimensão do embedding para o ramo MLP.
        categoria_dim (int): Dimensão do embedding de categoria.
        hidden_dims (List[int]): Dimensões das camadas ocultas da MLP.
        dropout (float): Taxa de dropout para regularização.
    """

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

    # Adicionando atributos necessários para o método recommend funcionar
    user_to_idx: Dict[int, int]
    idx_to_item: Dict[int, int]

    def recommend(self, user_id: int, k: int, **kwargs: Any) -> List[int]:
        """Gera recomendações para um usuário dado.

        Args:
            user_id (int): ID único do usuário.
            k (int): Número de itens a recomendar.
            **kwargs (Any): Argumentos adicionais.

        Returns:
            List[int]: Lista de IDs dos itens recomendados.

        Raises:
            NotImplementedError: Se o método não for implementado pela subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar recommend()")


class NeumfRetailrocketCpu(BaseRecommender):
    """Modelo Neural Matrix Factorization para o dataset RetailRocket."""

    def __init__(self, config: NeuMFConfig) -> None:
        """Inicializa as camadas do modelo NeuMF.

        Args:
            config (NeuMFConfig): Objeto de configuração contendo os hiperparâmetros.
        """
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
        """Define o fluxo de dados (forward pass) do modelo.

        Args:
            user_idx (torch.Tensor): Tensores de índices dos usuários.
            item_idx (torch.Tensor): Tensores de índices dos itens.

        Returns:
            torch.Tensor: Scores de preferência previstos.
        """
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
        """Gera recomendações ordenadas para um usuário.

        Args:
            user_id (int): ID do usuário.
            k (int): Quantidade de itens a retornar.
            **kwargs (Any): Parâmetros opcionais.

        Returns:
            List[int]: Lista de IDs de itens recomendados.
        """
        if not hasattr(self, "user_to_idx") or user_id not in self.user_to_idx:
            return []

        u_idx = self.user_to_idx[user_id]
        all_item_idxs = torch.arange(len(self.idx_to_item))
        user_t = torch.full((len(all_item_idxs),), u_idx)

        with torch.no_grad():
            scores = self.forward(user_t, all_item_idxs).numpy()

        top_indices = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[i] for i in top_indices]
