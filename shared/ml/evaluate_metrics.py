"""Módulo de avaliação de sistemas de recomendação.

Contém implementações vetorizadas e otimizadas de métricas clássicas
como Precision@K, Recall@K, NDCG@K e Catalog Coverage.
"""

import math
from typing import Dict, List, Callable, Set, Any

def _calculate_precision_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """Calcula a métrica Precision@K para um único usuário.

    Args:
        recommended: Lista de itens recomendados pelo sistema.
        relevant: Conjunto de itens relevantes (Ground Truth) para o usuário.
        k: O número de cortes (cutoff) para a avaliação.

    Returns:
        float: Valor da precisão (proporção de itens relevantes nos Top-K).
    """
    if not recommended:
        return 0.0
    recommended_k = recommended[:k]
    hits: int = sum(1 for item in recommended_k if item in relevant)
    return hits / k

def _calculate_recall_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """Calcula a métrica Recall@K para um único usuário.

    Args:
        recommended: Lista de itens recomendados pelo sistema.
        relevant: Conjunto de itens relevantes para o usuário.
        k: O número de cortes (cutoff) para a avaliação.

    Returns:
        float: Valor do recall (proporção de itens relevantes encontrados nos Top-K).
    """
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits: int = sum(1 for item in recommended_k if item in relevant)
    return hits / len(relevant)

def _calculate_ndcg_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """Calcula a métrica NDCG@K (Normalized Discounted Cumulative Gain) para um usuário.

    Args:
        recommended: Lista de itens recomendados.
        relevant: Conjunto de itens relevantes.
        k: O número de cortes (cutoff) para a avaliação.

    Returns:
        float: Valor normalizado entre 0 e 1, considerando a posição dos acertos.
    """
    if not relevant or not recommended:
        return 0.0
    
    dcg: float = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)
            
    idcg: float = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def avaliar_sistema_recomendacao(
    recommend_fn: Callable[..., List[Any]],
    test_users: List[Any],
    gt_dict: Dict[Any, List[Any]],
    n_items_total: int,
    k: int = 10,
    **kwargs: Any
) -> Dict[str, float]:
    """Avalia um modelo de recomendação iterando sobre usuários de teste.

    Realiza a predição para todos os usuários fornecidos, calcula as métricas 
    individuais e retorna a média agregada do sistema.

    Args:
        recommend_fn: Função injetada que recebe (user_id, k, **kwargs) e retorna itens.
        test_users: Lista de IDs de usuários a serem testados.
        gt_dict: Dicionário Ground Truth {user_id: [itens_relevantes]}.
        n_items_total: Total de itens únicos no catálogo (para cálculo de Coverage).
        k: Número máximo de recomendações consideradas no Top-K.
        **kwargs: Parâmetros extras repassados para a função de recomendação.

    Returns:
        Dict[str, float]: Dicionário contendo as médias de Precision, Recall, NDCG e Coverage.
    """
    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []
    
    # Conjunto para armazenar todos os itens únicos recomendados (Catalog Coverage)
    todos_itens_recomendados: Set[Any] = set()

    for user in test_users:
        if user not in gt_dict:
            continue
            
        itens_relevantes: Set[Any] = set(gt_dict[user])
        if not itens_relevantes:
            continue

        # Chama a função wrapper do modelo
        recomendacoes: List[Any] = recommend_fn(user, k, n_items_total, **kwargs)

        # Registra os itens para o cálculo de Coverage
        todos_itens_recomendados.update(recomendacoes[:k])

        # Calcula as métricas individuais
        precisions.append(_calculate_precision_at_k(recomendacoes, itens_relevantes, k))
        recalls.append(_calculate_recall_at_k(recomendacoes, itens_relevantes, k))
        ndcgs.append(_calculate_ndcg_at_k(recomendacoes, itens_relevantes, k))

    # Calcula médias globais
    mean_precision: float = sum(precisions) / len(precisions) if precisions else 0.0
    mean_recall: float = sum(recalls) / len(recalls) if recalls else 0.0
    mean_ndcg: float = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    coverage: float = len(todos_itens_recomendados) / n_items_total if n_items_total > 0 else 0.0

    return {
        f"Precision{k}": mean_precision,
        f"Recall{k}": mean_recall,
        f"NDCG{k}": mean_ndcg,
        "Coverage": coverage
    }