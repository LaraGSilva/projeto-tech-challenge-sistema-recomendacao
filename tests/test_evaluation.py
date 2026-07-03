import math
import pytest
from shared.ml.evaluate_metrics import (
    _calculate_precision_at_k,
    _calculate_recall_at_k,
    _calculate_ndcg_at_k,
    avaliar_sistema_recomendacao
)

def test_precision_at_k():
    recs = [1, 2, 3, 4, 5]
    relevantes = {2, 5, 8}
    
    # K = 5: Acertou 2 e 5. Precision = 2/5 = 0.4
    assert _calculate_precision_at_k(recs, relevantes, 5) == 0.4
    
    # K = 3: Acertou apenas o 2. Precision = 1/3
    assert math.isclose(_calculate_precision_at_k(recs, relevantes, 3), 0.3333, rel_tol=1e-3)
    
    # Nenhum acerto
    assert _calculate_precision_at_k([10, 11], relevantes, 2) == 0.0

def test_recall_at_k():
    recs = [1, 2, 3, 4, 5]
    relevantes = {2, 5, 8}
    
    # K = 5: Acertou 2 dos 3 relevantes. Recall = 2/3 = 0.666...
    assert math.isclose(_calculate_recall_at_k(recs, relevantes, 5), 0.6666, rel_tol=1e-3)

def test_ndcg_at_k():
    recs = [1, 2, 3]
    relevantes = {2} # Item 2 está no rank 1 (índice 1)
    
    # DCG = 1 / log2(rank + 2) -> rank 1 = 1 / log2(3) = 0.6309
    # IDCG = 1 / log2(0 + 2) -> 1 / 1 = 1.0
    # NDCG = 0.6309 / 1.0 = 0.6309
    ndcg_calc = _calculate_ndcg_at_k(recs, relevantes, 3)
    assert math.isclose(ndcg_calc, 0.6309, rel_tol=1e-3)

def test_avaliar_sistema_recomendacao():
    # Mock de função de recomendação
    def mock_recommend_fn(user_id, k, **kwargs):
        recs = {
            "u1": [10, 20, 30],
            "u2": [40, 50, 60]
        }
        return recs.get(user_id, [])

    gt_dict = {
        "u1": [20, 30], # Acertou 2
        "u2": [99]      # Errou tudo
    }
    
    # n_items_total = 100, K = 3
    # Coverage: Recomendou 6 itens únicos (10,20,30,40,50,60). Cov = 6/100 = 0.06
    metrics = avaliar_sistema_recomendacao(
        recommend_fn=mock_recommend_fn,
        test_users=["u1", "u2"],
        gt_dict=gt_dict,
        n_items_total=100,
        k=3
    )
    
    assert "Precision@3" in metrics
    assert "Recall@3" in metrics
    assert "NDCG@3" in metrics
    assert metrics["Coverage"] == 0.06