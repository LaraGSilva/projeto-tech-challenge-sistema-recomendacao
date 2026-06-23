import pytest
import torch
import numpy as np
import pandas as pd
from src.recommender.models.train_neural import (
    RetailRocketBPRDatasetCPU,
    NeuMF_RetailRocket_CPU,
    WeightedBPRLoss
)

@pytest.fixture
def dummy_neural_data():
    df = pd.DataFrame({
        'user_idx': [0, 1, 0, 2],
        'item_idx': [0, 1, 2, 0],
        'weight': [1.0, 3.0, 5.0, 1.0]
    })
    item_pop = {0: 2, 1: 1, 2: 1}
    item_to_cat = np.array([0, 1, 0]) # 3 itens, mapeados para categorias 0 e 1
    return df, item_pop, item_to_cat

def test_dataset_initialization(dummy_neural_data):
    df, item_pop, _ = dummy_neural_data
    dataset = RetailRocketBPRDatasetCPU(df, item_pop, n_items=3)
    
    assert len(dataset) == 4
    u, p, n, w = dataset[0]
    assert isinstance(u, torch.Tensor)
    assert p != n # Positivo não pode ser igual ao negativo gerado

def test_neumf_forward_pass(dummy_neural_data):
    _, _, item_to_cat = dummy_neural_data
    
    model = NeuMF_RetailRocket_CPU(
        n_users=3, 
        n_items=3, 
        n_categorias=2, 
        item_to_cat_array=item_to_cat,
        mf_dim=8,
        mlp_dim=16,
        hidden_dims=[32, 16]
    )
    
    # Batch simulado de 2 interações
    users = torch.tensor([0, 1])
    pos_items = torch.tensor([2, 1])
    neg_items = torch.tensor([1, 0])
    
    # Teste de forward pass completo (treino)
    pos_scores, neg_scores = model(users, pos_items, neg_items)
    
    assert pos_scores.shape == torch.Size([2])
    assert neg_scores.shape == torch.Size([2])
    
    # Teste de loss function
    criterion = WeightedBPRLoss()
    weights = torch.tensor([1.0, 5.0])
    loss = criterion(pos_scores, neg_scores, weights)
    
    assert loss.item() > 0