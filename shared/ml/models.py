import torch
import torch.nn as nn
import numpy as np
from typing import List, Any, Dict

class BaseRecommender(nn.Module):
    """Classe base para garantir interface unificada entre modelos de recomendação."""
    
    def recommend(self, user_id: int, k: int, **kwargs: Any) -> List[int]:
        """Gera recomendações para um usuário dado.

        Args:
            user_id (int): ID do usuário.
            k (int): Número de itens a recomendar.
            **kwargs: Argumentos adicionais.

        Raises:
            NotImplementedError: Se o método não for implementado pela subclasse.
        """
        raise NotImplementedError

class NeuMF_RetailRocket_CPU(BaseRecommender):
    """Modelo Neural Matrix Factorization para o dataset RetailRocket."""

    def __init__(self, n_users: int, n_items: int, n_categorias: int, item_to_cat_array: List[int],
                 mf_dim: int = 16, mlp_dim: int = 32, categoria_dim: int = 8,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.1, **kwargs: Any):
        """Inicializa as camadas do modelo NeuMF.

        Args:
            n_users (int): Número total de usuários.
            n_items (int): Número total de itens.
            n_categorias (int): Número total de categorias.
            item_to_cat_array (List[int]): Mapeamento estático de item para categoria.
            mf_dim (int): Dimensão do embedding para Fatoração de Matriz.
            mlp_dim (int): Dimensão do embedding para o ramo MLP.
            categoria_dim (int): Dimensão do embedding de categoria.
            hidden_dims (List[int]): Lista com tamanhos das camadas ocultas da MLP.
            dropout (float): Taxa de dropout para regularização.
            **kwargs: Parâmetros adicionais.
        """
        super().__init__()
        self.user_mf_embed = nn.Embedding(n_users, mf_dim)
        self.item_mf_embed = nn.Embedding(n_items, mf_dim)
        self.user_mlp_embed = nn.Embedding(n_users, mlp_dim)
        self.item_mlp_embed = nn.Embedding(n_items, mlp_dim)
        self.cat_mlp_embed = nn.Embedding(n_categorias, categoria_dim)
        self.register_buffer("item_categoria_idx", torch.tensor(item_to_cat_array, dtype=torch.long))
        
        in_dim = mlp_dim * 2 + categoria_dim
        layers: List[nn.Module] = []
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        self.prediction_layer = nn.Linear(mf_dim + hidden_dims[-1], 1)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Define o fluxo de dados (forward pass) do modelo."""
        user_mf = self.user_mf_embed(user_idx)
        item_mf = self.item_mf_embed(item_idx)
        gmf_vector = user_mf * item_mf
        
        user_mlp = self.user_mlp_embed(user_idx)
        item_mlp = self.item_mlp_embed(item_idx)
        cat_mlp = self.cat_mlp_embed(self.item_categoria_idx[item_idx])
        
        mlp_vector = self.mlp(torch.cat([user_mlp, item_mlp, cat_mlp], dim=-1))
        return self.prediction_layer(torch.cat([gmf_vector, mlp_vector], dim=-1)).view(-1)

    def recommend(self, user_id: int, k: int, **kwargs: Any) -> List[int]:
        """Gera recomendações usando pontuação do modelo."""
        if not hasattr(self, 'user_to_idx') or user_id not in self.user_to_idx:
            return []
        
        u_idx = self.user_to_idx[user_id]
        all_item_idxs = torch.arange(len(self.idx_to_item))
        user_t = torch.full((len(all_item_idxs),), u_idx)
        
        with torch.no_grad():
            scores = self.forward(user_t, all_item_idxs).numpy()
        
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[i] for i in top_indices]

class NeuMF_Light(BaseRecommender):
    """Versão simplificada do modelo NeuMF."""

    def __init__(self, n_users: int, n_items: int, mf_dim: int = 16, **kwargs: Any):
        """Inicializa modelo leve."""
        super().__init__()
        self.user_embed = nn.Embedding(n_users, mf_dim)
        self.item_embed = nn.Embedding(n_items, mf_dim)
        self.prediction_layer = nn.Linear(mf_dim, 1)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Executa o forward pass simplificado."""
        vector = self.user_embed(u) * self.item_embed(i)
        return self.prediction_layer(vector).view(-1)

    def recommend(self, user_id: int, k: int, **kwargs: Any) -> List[int]:
        """Gera recomendações para o modelo light."""
        if not hasattr(self, 'user_to_idx') or user_id not in self.user_to_idx:
            return []
        
        u_idx = self.user_to_idx[user_id]
        all_item_idxs = torch.arange(len(self.idx_to_item))
        user_t = torch.full((len(all_item_idxs),), u_idx)
        
        with torch.no_grad():
            scores = self.forward(user_t, all_item_idxs).numpy()
            
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[i] for i in top_indices]