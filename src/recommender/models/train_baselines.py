# src/recommender/models/train_baselines.py
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# ── configuração ──────────────────────────────────────────────────────────────

MATRIX_PATH   = Path("data/features/user_item_matrix.npz")
MAPPINGS_PATH = Path("data/features/mappings.pkl")
MODELS_DIR    = Path("models/baselines")
N_RECS        = 10
SEED          = 42

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("retailrocket-recommender")


# ── helpers de avaliação ──────────────────────────────────────────────────────

def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    hits = len(set(recommended[:k]) & relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(
        1 / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def coverage(all_recommended: list[list[int]], n_items: int) -> float:
    unique = {item for recs in all_recommended for item in recs}
    return len(unique) / n_items


def evaluate(
    recommend_fn,       # callable: user_idx -> list[int]
    matrix: csr_matrix,
    test_users: list[int],
    n_items: int,
    k: int = N_RECS,
) -> dict[str, float]:
    """Roda as 4 métricas sobre uma amostra de usuários de teste."""
    precisions, recalls, ndcgs, all_recs = [], [], [], []

    for u in test_users:
        # itens vistos na segunda metade do histórico = ground truth
        relevant = set(matrix[u].nonzero()[1])
        if not relevant:
            continue

        recs = recommend_fn(u)
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        all_recs.append(recs)

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k":    float(np.mean(recalls)),
        "ndcg_at_k":      float(np.mean(ndcgs)),
        "coverage":       coverage(all_recs, n_items),
    }


# ── carregamento dos dados ────────────────────────────────────────────────────

def load_artifacts() -> tuple[csr_matrix, dict]:
    matrix = load_npz(MATRIX_PATH)

    with open(MAPPINGS_PATH, "rb") as f:
        mappings = pickle.load(f)

    return matrix, mappings


# ── baseline 1: popularity ────────────────────────────────────────────────────

def train_popularity(
    matrix: csr_matrix,
    test_users: list[int],
) -> None:
    with mlflow.start_run(run_name="popularity"):
        mlflow.log_param("model_type", "popularity")
        mlflow.log_param("n_recommendations", N_RECS)

        # score de cada item = soma de todos os pesos na coluna
        item_scores = np.asarray(matrix.sum(axis=0)).flatten()
        top_items   = np.argsort(item_scores)[::-1][:N_RECS].tolist()

        # a recomendação é idêntica para qualquer usuário
        recommend_fn = lambda u: top_items

        metrics = evaluate(recommend_fn, matrix, test_users, matrix.shape[1])
        mlflow.log_metrics(metrics)

        # salvar modelo (simples: só a lista ordenada)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "popularity_top_items.npy"
        np.save(model_path, top_items)
        mlflow.log_artifact(str(model_path))

        print(f"[popularity] {metrics}")


# ── baseline 2: knn user-based ────────────────────────────────────────────────

def train_knn(
    matrix: csr_matrix,
    test_users: list[int],
    k_neighbors: int = 20,
) -> None:
    with mlflow.start_run(run_name="knn_user_cf"):
        mlflow.log_param("model_type", "knn_user_cf")
        mlflow.log_param("k_neighbors", k_neighbors)
        mlflow.log_param("metric", "cosine")
        mlflow.log_param("n_recommendations", N_RECS)

        knn = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=k_neighbors + 1
        )
        knn.fit(matrix)

        def recommend_fn(user_idx: int) -> list[int]:
            distances, neighbors = knn.kneighbors(matrix[user_idx])
            neighbors   = neighbors[0][1:]          # remove o próprio usuário
            similarities = 1 - distances[0][1:]

            # soma ponderada das interações dos vizinhos
            neighbor_matrix = matrix[neighbors].toarray()
            scores = similarities @ neighbor_matrix

            already_seen = matrix[user_idx].nonzero()[1]
            scores[already_seen] = -np.inf
            return np.argsort(scores)[::-1][:N_RECS].tolist()

        metrics = evaluate(recommend_fn, matrix, test_users, matrix.shape[1])
        mlflow.log_metrics(metrics)

        # salvar modelo
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "knn_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(knn, f)
        mlflow.log_artifact(str(model_path))

        print(f"[knn] {metrics}")


# ── baseline 3: svd matrix factorization ─────────────────────────────────────

def train_svd(
    matrix: csr_matrix,
    test_users: list[int],
    n_components: int = 50,
) -> None:
    with mlflow.start_run(run_name="svd"):
        mlflow.log_param("model_type", "svd")
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("n_recommendations", N_RECS)
        mlflow.log_param("seed", SEED)

        svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        user_factors = svd.fit_transform(matrix)    # (n_users, n_components)
        item_factors = svd.components_.T            # (n_items, n_components)

        def recommend_fn(user_idx: int) -> list[int]:
            scores = user_factors[user_idx] @ item_factors.T
            already_seen = matrix[user_idx].nonzero()[1]
            scores[already_seen] = -np.inf
            return np.argsort(scores)[::-1][:N_RECS].tolist()

        metrics = evaluate(recommend_fn, matrix, test_users, matrix.shape[1])
        mlflow.log_metrics(metrics)

        # salvar fatores
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(MODELS_DIR / "svd_user_factors.npy", user_factors)
        np.save(MODELS_DIR / "svd_item_factors.npy", item_factors)
        mlflow.sklearn.log_model(svd, "svd_model")

        print(f"[svd] {metrics}")


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    matrix, mappings = load_artifacts()

    n_users = matrix.shape[0]

    # amostra de usuários para avaliação (evita rodar nos 235K)
    rng         = np.random.default_rng(SEED)
    test_users  = rng.choice(n_users, size=min(5_000, n_users), replace=False).tolist()

    train_popularity(matrix, test_users)
    train_knn(matrix, test_users)
    train_svd(matrix, test_users)


if __name__ == "__main__":
    main()