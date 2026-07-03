import torch
import torch.nn as nn
from typing import List

class NeuMF_RetailRocket_CPU(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_categorias: int, item_to_cat_array: list,
                 mf_dim: int = 16, mlp_dim: int = 32, categoria_dim: int = 8,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        
        # Embeddings
        self.user_mf_embed = nn.Embedding(n_users, mf_dim)
        self.item_mf_embed = nn.Embedding(n_items, mf_dim)
        self.user_mlp_embed = nn.Embedding(n_users, mlp_dim)
        self.item_mlp_embed = nn.Embedding(n_items, mlp_dim)
        self.cat_mlp_embed = nn.Embedding(n_categorias, categoria_dim)

        # Buffer para dados que não são parâmetros treináveis
        self.register_buffer("item_categoria_idx", torch.tensor(item_to_cat_array, dtype=torch.long))

        # MLP Layers
        in_dim = mlp_dim * 2 + categoria_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        
        # Camada final de predição (GMF + MLP)
        self.prediction_layer = nn.Linear(mf_dim + hidden_dims[-1], 1)

    def forward(self, user_idx, item_idx):
        # GMF Path
        user_mf = self.user_mf_embed(user_idx)
        item_mf = self.item_mf_embed(item_idx)
        gmf_vector = user_mf * item_mf
        
        # MLP Path
        user_mlp = self.user_mlp_embed(user_idx)
        item_mlp = self.item_mlp_embed(item_idx)
        cat_mlp = self.cat_mlp_embed(self.item_categoria_idx[item_idx])
        
        mlp_vector = self.mlp(torch.cat([user_mlp, item_mlp, cat_mlp], dim=-1))
        
        # Fusão
        combined = torch.cat([gmf_vector, mlp_vector], dim=-1)
        return self.prediction_layer(combined).view(-1)

class NeuMF_Light(nn.Module):
    """Modelo simples de Fatoração de Matrizes para baselines."""
    def __init__(self, n_users: int, n_items: int, mf_dim: int = 16):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, mf_dim)
        self.item_embed = nn.Embedding(n_items, mf_dim)
        self.prediction_layer = nn.Linear(mf_dim, 1)

    def forward(self, u, i):
        # Multiplicação elemento a elemento
        vector = self.user_embed(u) * self.item_embed(i)
        return self.prediction_layer(vector).view(-1)