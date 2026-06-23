"""
Módulo de avaliação de sistemas de recomendação.
Contém implementações vetorizadas e otimizadas de métricas clássicas
como Precision@K, Recall@K, NDCG@K e Catalog Coverage.
"""

import math
from typing import Dict, List, Callable, Set, Any

def _calculate_precision_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """Calcula a métrica Precision@K para um único usuário."""
    if not recommended:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / k

def _calculate_recall_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """Calcula a métrica Recall@K para um único usuário."""
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / len(relevant)

def _calculate_ndcg_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """Calcula a métrica NDCG@K (Normalized Discounted Cumulative Gain) para um único usuário."""
    if not relevant or not recommended:
        return 0.0
    
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)  # +2 porque o índice começa em 0 e a fórmula usa log2(rank + 1)
            
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def avaliar_sistema_recomendacao(
    recommend_fn: Callable,
    test_users: List[Any],
    gt_dict: Dict[Any, List[Any]],
    n_items_total: int,
    k: int = 10,
    **kwargs
) -> Dict[str, float]:
    """
    Avalia um modelo de recomendação iterando sobre os usuários de teste
    e agregando múltiplas métricas de performance.

    Args:
        recommend_fn: Função que recebe (user_id, k, **kwargs) e retorna lista de itens recomendados.
        test_users: Lista de IDs de usuários para testar.
        gt_dict: Dicionário contendo o Ground Truth {user_id: [itens_relevantes]}.
        n_items_total: Número total de itens únicos no catálogo (para calcular cobertura).
        k: Número máximo de recomendações por usuário (Top-K).
        **kwargs: Parâmetros adicionais repassados para a recommend_fn (ex: modelo, dicionários de conversão).

    Returns:
        Um dicionário contendo as médias das métricas calculadas.
    """
    precisions = []
    recalls = []
    ndcgs = []
    
    # Conjunto para armazenar todos os itens únicos recomendados (usado no Coverage)
    todos_itens_recomendados = set()

    for user in test_users:
        if user not in gt_dict:
            continue
            
        itens_relevantes = set(gt_dict[user])
        if not itens_relevantes:
            continue

        # Chama a função wrapper do modelo para gerar as recomendações
        recomendacoes = recommend_fn(user, k=k, **kwargs)
        
        # Registra os itens para o cálculo de cobertura de catálogo
        todos_itens_recomendados.update(recomendacoes[:k])

        # Calcula as métricas individuais
        precisions.append(_calculate_precision_at_k(recomendacoes, itens_relevantes, k))
        recalls.append(_calculate_recall_at_k(recomendacoes, itens_relevantes, k))
        ndcgs.append(_calculate_ndcg_at_k(recomendacoes, itens_relevantes, k))

    # Calcula médias globais
    mean_precision = sum(precisions) / len(precisions) if precisions else 0.0
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    coverage = len(todos_itens_recomendados) / n_items_total if n_items_total > 0 else 0.0

    return {
        f"Precision@{k}": mean_precision,
        f"Recall@{k}": mean_recall,
        f"NDCG@{k}": mean_ndcg,
        "Coverage": coverage
    }