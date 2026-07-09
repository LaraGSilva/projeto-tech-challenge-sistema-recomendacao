import pickle
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable

import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix

# Ajuste do path para importar módulos da arquitetura
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from shared.utils.config import load_config
from shared.ml.evaluate_metrics import avaliar_sistema_recomendacao
from shared.ml.model_factory import ModelFactory

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
mlflow.set_experiment("retailrocket-recommender")

# ── Configurações ─────────────────────────────────────────────────────────────
MATRIX_PATH: Path = Path("data/features/user_item_matrix.npz")
MAPPINGS_PATH: Path = Path("data/features/mappings.pkl")
SEED: int = 42

def load_artifacts() -> Tuple[csr_matrix, Dict[str, Any]]:
    """Carrega a matriz esparsa e os mapeamentos do projeto.

    Returns:
        Tuple[csr_matrix, Dict[str, Any]]: Uma tupla contendo a matriz de interações 
        (csr_matrix) e o dicionário de mapeamentos.
    """
    matrix: csr_matrix = load_npz(MATRIX_PATH)
    with open(MAPPINGS_PATH, "rb") as f:
        mappings: Dict[str, Any] = pickle.load(f)
    return matrix, mappings

def _build_ground_truth(
    matrix: csr_matrix, 
    mappings: Dict[str, Any], 
    test_users_idx: List[int]
) -> Dict[int, List[int]]:
    """Reconstrói o dicionário de ground truth para avaliação.

    Args:
        matrix (csr_matrix): Matriz de interação usuário-item.
        mappings (Dict[str, Any]): Dicionário contendo 'idx_to_user' e 'idx_to_item'.
        test_users_idx (List[int]): Lista de índices de usuários para teste.

    Returns:
        Dict[int, List[int]]: Dicionário onde chaves são IDs originais dos usuários 
        e valores são listas de IDs originais dos itens relevantes.
    """
    gt_dict: Dict[int, List[int]] = {}
    idx_to_user: Dict[int, int] = mappings["idx_to_user"]
    idx_to_item: Dict[int, int] = mappings["idx_to_item"]

    for u_idx in test_users_idx:
        relevant_idx: np.ndarray = matrix[u_idx].nonzero()[1]
        if len(relevant_idx) > 0:
            original_user_id: int = idx_to_user[u_idx]
            original_items: List[int] = [idx_to_item[i] for i in relevant_idx]
            gt_dict[original_user_id] = original_items
    return gt_dict

def run_baseline_training() -> None:
    """Orquestra o treinamento e avaliação dos modelos baselines.

    Carrega dados, configura modelos, realiza o treino/inferência, avalia o sistema
    e registra os resultados e parâmetros no MLflow.
    """
    print("Iniciando treinamento de baselines...")

    cfg: Dict[str, Any] = load_config()
    matrix, mappings = load_artifacts()
    n_users, n_items = matrix.shape
    
    # 1. Seleção de usuários para teste
    rng: np.random.Generator = np.random.default_rng(SEED)
    test_users_idx: List[int] = rng.choice(n_users, size=min(5_000, n_users), replace=False).tolist()
    gt_dict: Dict[int, List[int]] = _build_ground_truth(matrix, mappings, test_users_idx)
    test_users: List[int] = list(gt_dict.keys())

    # 2. Configurações dos modelos
    model_configs: Dict[str, Dict[str, Any]] = {
        "popularity": {"config": {"matrix": matrix, "mappings": mappings}},
        "knn": {
            "config": {
                "matrix": matrix, 
                "mappings": mappings, 
                "k": cfg['baselines']['knn']['k_neighbors']
            }
        },
        "svd": {
            "config": {
                "matrix": matrix, 
                "mappings": mappings, 
                "n_components": cfg['baselines']['svd']['n_components']
            }
        }
    }

    df_raw: pd.DataFrame = pd.read_csv("shared/data/data_csv/raw/events.csv")

    # 3. Loop de Treino e Avaliação via Factory
    for name, params in model_configs.items():
        with mlflow.start_run(run_name=f"baseline_{name}"):
            dataset_mlflow = mlflow.data.from_pandas(
                df_raw, name="dataset_RetailRocket_Events_and_Properties"
            )
            mlflow.log_input(dataset_mlflow, context="training")

            # Instancia via Factory
            model = ModelFactory.create_model(name, params["config"], mappings)
            
            # Avaliação padronizada
            metrics: Dict[str, float] = avaliar_sistema_recomendacao(
                recommend_fn=model.recommend,
                test_users=test_users,
                gt_dict=gt_dict,
                n_items_total=n_items,
                k=cfg['baselines'].get('n_recs', 10)
            )
            
            # Logging no MLflow
            if name != "popularity":
                mlflow.log_params(params["config"])
            mlflow.log_metrics({f"eval.{k.lower()}": v for k, v in metrics.items()})
            
            print(f"Baseline {name} concluído. Métricas: {metrics}")

if __name__ == "__main__":
    run_baseline_training()