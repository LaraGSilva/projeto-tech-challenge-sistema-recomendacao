import pytest
import torch
import numpy as np
from shared.ml.model_factory import ModelFactory

def test_model_recommendation_output_format():
    """Verifica se o método recommend retorna o formato de lista correto e tamanho K."""
    # Setup de dados falsos para teste
    n_users, n_items = 10, 50
    model = ModelFactory.create_model(
        "neumf_light", 
        config={"n_users": n_users, "n_items": n_items, "mf_dim": 8}
    )
    
    # Injeção de dependência dos mapeamentos (Mock)
    mappings = {
        "user_to_idx": {51: 0},
        "idx_to_item": {i: i * 1000 for i in range(n_items)}
    }
    model.user_to_idx = mappings["user_to_idx"]
    model.idx_to_item = mappings["idx_to_item"]
    
    # Execução
    k = 5
    recs = model.recommend(51, k)
    
    # Asserções
    assert isinstance(recs, list), "O retorno deve ser uma lista"
    assert len(recs) == k, f"Deveria retornar {k} recomendações"
    assert all(isinstance(i, int) for i in recs), "Todos os itens devem ser inteiros"

def test_model_cold_start_handling():
    """Verifica se o modelo retorna lista vazia para usuários desconhecidos."""
    model = ModelFactory.create_model(
        "neumf_light", 
        config={"n_users": 10, "n_items": 50, "mf_dim": 8}
    )
    model.user_to_idx = {51: 0}
    
    # Usuário que não está no mapeamento
    recs = model.recommend(999, 5)
    
    assert recs == [], "Usuário desconhecido deveria retornar lista vazia"