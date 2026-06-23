import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# ── Importando as métricas padronizadas do projeto ────────────────────────────
from src.recommender.evaluation.evaluate import avaliar_sistema_recomendacao

# ── configuração ──────────────────────────────────────────────────────────────

MATRIX_PATH = Path("data/features/user_item_matrix.npz")
MAPPINGS_PATH = Path("data/features/mappings.pkl")
MODELS_DIR = Path("models/baselines")
N_RECS = 10
SEED = 42

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("retailrocket-recommender")

# ── carregamento dos dados ────────────────────────────────────────────────────


def load_artifacts() -> tuple[csr_matrix, dict]:
    matrix = load_npz(MATRIX_PATH)
    with open(MAPPINGS_PATH, "rb") as f:
        mappings = pickle.load(f)
    return matrix, mappings

# ── preparação do ground truth para o avaliador ───────────────────────────────


def _build_ground_truth(matrix: csr_matrix, mappings: dict, test_users_idx: list[int]) -> dict:
    """Reconstrói o gt_dict (user_id -> list[item_id]) a partir da matriz esparsa."""
    gt_dict = {}
    idx_to_user = mappings["idx_to_user"]
    idx_to_item = mappings["idx_to_item"]

    for u_idx in test_users_idx:
        # Pega as colunas com valor > 0 na linha do usuário
        relevant_idx = matrix[u_idx].nonzero()[1]
        if len(relevant_idx) > 0:
            original_user_id = idx_to_user[u_idx]
            original_items = [idx_to_item[i] for i in relevant_idx]
            gt_dict[original_user_id] = original_items

    return gt_dict

# ── baseline 1: popularity ────────────────────────────────────────────────────


def train_popularity(
    matrix: csr_matrix,
    mappings: dict,
    test_users_original: list[int],
    gt_dict: dict,
    n_items_total: int
) -> None:
    with mlflow.start_run(run_name="popularity"):
        mlflow.log_param("model_type", "popularity")
        mlflow.log_param("n_recommendations", N_RECS)

        item_scores = np.asarray(matrix.sum(axis=0)).flatten()
        top_items_idx = np.argsort(item_scores)[::-1][:N_RECS].tolist()

        # Converte a recomendação de índices para IDs originais
        top_items_original = [mappings["idx_to_item"][i]
                              for i in top_items_idx]

        # Wrapper adaptado para o formato da função de avaliação
        def recommend_fn(user_id, k, **kwargs):
            return top_items_original[:k]

        metrics = avaliar_sistema_recomendacao(
            recommend_fn=recommend_fn,
            test_users=test_users_original,
            gt_dict=gt_dict,
            n_items_total=n_items_total,
            k=N_RECS
        )

        mlflow.log_metrics({
            "eval.precision_at_10": metrics[f"Precision@{N_RECS}"],
            "eval.recall_at_10": metrics[f"Recall@{N_RECS}"],
            "eval.ndcg_at_10": metrics[f"NDCG@{N_RECS}"],
            "eval.catalog_coverage": metrics["Coverage"]
        })

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "popularity_top_items.npy"
        np.save(model_path, top_items_idx)
        mlflow.log_artifact(str(model_path))

        print(f"[popularity] {metrics}")

# ── baseline 2: knn user-based ────────────────────────────────────────────────


def train_knn(
    matrix: csr_matrix,
    mappings: dict,
    test_users_original: list[int],
    gt_dict: dict,
    n_items_total: int,
    k_neighbors: int = 20,
) -> None:
    with mlflow.start_run(run_name="knn_user_cf"):
        mlflow.log_param("model_type", "knn_user_cf")
        mlflow.log_param("k_neighbors", k_neighbors)
        mlflow.log_param("metric", "cosine")
        mlflow.log_param("n_recommendations", N_RECS)

        knn = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=k_neighbors + 1)
        knn.fit(matrix)

        user_to_idx = mappings["user_to_idx"]
        idx_to_item = mappings["idx_to_item"]

        def recommend_fn(user_id, k, **kwargs):
            if user_id not in user_to_idx:
                return []

            user_idx = user_to_idx[user_id]
            distances, neighbors = knn.kneighbors(matrix[user_idx])
            neighbors = neighbors[0][1:]
            similarities = 1 - distances[0][1:]

            neighbor_matrix = matrix[neighbors].toarray()
            scores = similarities @ neighbor_matrix

            already_seen = matrix[user_idx].nonzero()[1]
            scores[already_seen] = -np.inf
            top_idx = np.argsort(scores)[::-1][:k].tolist()

            return [idx_to_item[i] for i in top_idx]

        metrics = avaliar_sistema_recomendacao(
            recommend_fn=recommend_fn,
            test_users=test_users_original,
            gt_dict=gt_dict,
            n_items_total=n_items_total,
            k=N_RECS
        )

        mlflow.log_metrics({
            "eval.precision_at_10": metrics[f"Precision@{N_RECS}"],
            "eval.recall_at_10": metrics[f"Recall@{N_RECS}"],
            "eval.ndcg_at_10": metrics[f"NDCG@{N_RECS}"],
            "eval.catalog_coverage": metrics["Coverage"]
        })

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "knn_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(knn, f)
        mlflow.log_artifact(str(model_path))

        print(f"[knn] {metrics}")

# ── baseline 3: svd matrix factorization ─────────────────────────────────────


def train_svd(
    matrix: csr_matrix,
    mappings: dict,
    test_users_original: list[int],
    gt_dict: dict,
    n_items_total: int,
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

        user_to_idx = mappings["user_to_idx"]
        idx_to_item = mappings["idx_to_item"]

        def recommend_fn(user_id, k, **kwargs):
            if user_id not in user_to_idx:
                return []

            user_idx = user_to_idx[user_id]
            scores = user_factors[user_idx] @ item_factors.T
            already_seen = matrix[user_idx].nonzero()[1]
            scores[already_seen] = -np.inf

            top_idx = np.argsort(scores)[::-1][:k].tolist()
            return [idx_to_item[i] for i in top_idx]

        metrics = avaliar_sistema_recomendacao(
            recommend_fn=recommend_fn,
            test_users=test_users_original,
            gt_dict=gt_dict,
            n_items_total=n_items_total,
            k=N_RECS
        )

        mlflow.log_metrics({
            "eval.precision_at_10": metrics[f"Precision@{N_RECS}"],
            "eval.recall_at_10": metrics[f"Recall@{N_RECS}"],
            "eval.ndcg_at_10": metrics[f"NDCG@{N_RECS}"],
            "eval.catalog_coverage": metrics["Coverage"]
        })

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(MODELS_DIR / "svd_user_factors.npy", user_factors)
        np.save(MODELS_DIR / "svd_item_factors.npy", item_factors)
        mlflow.sklearn.log_model(svd, "svd_model")

        print(f"[svd] {metrics}")

# ── entrypoint ────────────────────────────────────────────────────────────────


def main() -> None:
    matrix, mappings = load_artifacts()

    n_users = matrix.shape[0]
    n_items_total = matrix.shape[1]

    rng = np.random.default_rng(SEED)
    test_users_idx = rng.choice(n_users, size=min(
        5_000, n_users), replace=False).tolist()

    # Prepara dicionário padrão para a função avaliar_sistema_recomendacao
    gt_dict = _build_ground_truth(matrix, mappings, test_users_idx)
    test_users_original = list(gt_dict.keys())

    train_popularity(matrix, mappings, test_users_original,
                     gt_dict, n_items_total)
    train_knn(matrix, mappings, test_users_original, gt_dict, n_items_total)
    train_svd(matrix, mappings, test_users_original, gt_dict, n_items_total)


if __name__ == "__main__":
    main()
