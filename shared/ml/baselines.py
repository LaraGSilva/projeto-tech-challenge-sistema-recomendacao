from typing import Any, Dict, List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


class PopularityModel:
    """Baseline baseado na popularidade global dos itens.

    Atributos:
        item_scores (np.ndarray): Soma das interações por item.
        idx_to_item (Dict[int, Any]): Mapeamento de índice interno para ID original.
        top_items_idx (np.ndarray): Índices dos itens ordenados por popularidade.
    """

    def __init__(
        self, matrix: csr_matrix, mappings: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Inicializa o modelo de popularidade.

        Args:
            matrix (csr_matrix): Matriz esparsa de interação usuário-item.
            mappings (Dict[str, Any]): Dicionário contendo o mapeamento 'idx_to_item'.
        """
        self.item_scores: np.ndarray = np.asarray(matrix.sum(axis=0)).flatten()
        self.idx_to_item: Dict[int, Any] = mappings["idx_to_item"]
        self.top_items_idx: np.ndarray = np.argsort(self.item_scores)[::-1]

    def recommend(self, user_id: int, k: int, *args: Any, **kwargs: Any) -> List[Any]:
        """Retorna os itens mais populares globalmente.

        Args:
            user_id (int): ID do usuário (não utilizado neste modelo).
            k (int): Número de recomendações a retornar.
            *args: Argumentos posicionais extras.
            **kwargs: Argumentos nomeados extras.

        Returns:
            List[Any]: Lista com os IDs originais dos itens recomendados.
        """
        return [self.idx_to_item[i] for i in self.top_items_idx[:k]]


class KNNModel:
    """Baseline baseado em KNN User-to-User."""

    def __init__(
        self, matrix: csr_matrix, mappings: Dict[str, Any], k: int = 20, **kwargs: Any
    ) -> None:
        """Inicializa e treina o modelo KNN.

        Args:
            matrix (csr_matrix): Matriz de interação para treino.
            mappings (Dict[str, Any]): Mapeamentos de ID e índices.
            k (int): Número de vizinhos a considerar.
        """
        self.matrix: csr_matrix = matrix
        self.user_to_idx: Dict[Any, int] = mappings["user_to_idx"]
        self.idx_to_item: Dict[int, Any] = mappings["idx_to_item"]

        self.knn = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=k + 1
        )
        self.knn.fit(matrix)

    def recommend(self, user_id: Any, k: int, *args: Any, **kwargs: Any) -> List[Any]:
        """Calcula recomendações baseadas na similaridade entre usuários.

        Args:
            user_id (Any): ID do usuário a receber recomendações.
            k (int): Quantidade de itens a recomendar.

        Returns:
            List[Any]: Lista de IDs de itens recomendados.
        """
        if user_id not in self.user_to_idx:
            return []

        user_idx: int = self.user_to_idx[user_id]
        distances, neighbors = self.knn.kneighbors(self.matrix[user_idx])

        neighbors = neighbors[0][1:]
        similarities = 1 - distances[0][1:]

        neighbor_matrix = self.matrix[neighbors].toarray()
        scores = similarities @ neighbor_matrix

        already_seen = self.matrix[user_idx].nonzero()[1]
        scores[already_seen] = -np.inf

        top_idx = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[i] for i in top_idx]


class SVDModel:
    """Baseline baseado em Fatoração de Matrizes (SVD)."""

    def __init__(
        self,
        matrix: csr_matrix,
        mappings: Dict[str, Any],
        n_components: int = 50,
        **kwargs: Any,
    ) -> None:
        """Inicializa e treina o modelo SVD.

        Args:
            matrix (csr_matrix): Matriz esparsa de interação.
            mappings (Dict[str, Any]): Mapeamentos de ID e índices.
            n_components (int): Número de dimensões latentes.
        """
        self.matrix: csr_matrix = matrix
        self.user_to_idx: Dict[Any, int] = mappings["user_to_idx"]
        self.idx_to_item: Dict[int, Any] = mappings["idx_to_item"]

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors: np.ndarray = self.svd.fit_transform(matrix)
        self.item_factors: np.ndarray = self.svd.components_.T

    def recommend(self, user_id: Any, k: int, *args: Any, **kwargs: Any) -> List[Any]:
        """Gera recomendações usando o produto escalar dos fatores latentes.

        Args:
            user_id (Any): ID do usuário para recomendação.
            k (int): Quantidade de itens a retornar.

        Returns:
            List[Any]: Lista com os IDs originais dos itens recomendados.
        """
        if user_id not in self.user_to_idx:
            return []

        user_idx: int = self.user_to_idx[user_id]
        scores: np.ndarray = self.user_factors[user_idx] @ self.item_factors.T

        already_seen = self.matrix[user_idx].nonzero()[1]
        scores[already_seen] = -np.inf

        top_idx = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[i] for i in top_idx]
