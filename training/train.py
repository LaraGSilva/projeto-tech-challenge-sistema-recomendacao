import os
import logging
from typing import List, Dict, Any, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from mlflow.models import ModelSignature, infer_signature
from mlflow.types import ColSpec, Schema
from torch.utils.data import DataLoader, Dataset

from shared.data.preprocessing import DefaultEventPreprocessor
from shared.ml.evaluate_metrics import avaliar_sistema_recomendacao
from shared.ml.model_factory import ModelFactory
from shared.utils.config import load_config

# Configuração do Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Configuração de ambiente
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
mlflow.set_experiment("retailrocket-recommender")

def recommend_wrapper_mlp_fast(
    user_idx: int, k: int, n_items_total: int, model: nn.Module, **kwargs: Any
) -> List[int]:
    """Gera recomendações usando o modelo MLP/NeuMF.

    Args:
        user_idx (int): Índice do usuário.
        k (int): Número de itens a recomendar.
        n_items_total (int): Total de itens disponíveis.
        model (nn.Module): Modelo PyTorch carregado.
        **kwargs: Argumentos extras, espera-se 'idx_to_item'.

    Returns:
        List[int]: Lista de IDs dos itens recomendados.
    """
    model.eval()
    idx_to_item: Dict[int, int] = kwargs.get('idx_to_item')

    if idx_to_item is None:
        raise ValueError("O argumento 'idx_to_item' é obrigatório no kwargs.")

    with torch.no_grad():
        u_t = torch.full((n_items_total,), user_idx, dtype=torch.long)
        i_t = torch.arange(n_items_total, dtype=torch.long)
        scores = model(u_t, i_t).numpy()

    top_indices = np.argsort(scores)[::-1]
    recs = []
    for idx in top_indices:
        idx_int = int(idx)
        if idx_int in idx_to_item:
            recs.append(idx_to_item[idx_int])
        if len(recs) == k:
            break
    return recs

class RetailRocketDataset(Dataset):
    """Dataset para carregamento dos dados de interação RetailRocket."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Inicializa o dataset com tensores PyTorch."""
        self.users = torch.from_numpy(df['user_idx'].values).long()
        self.items = torch.from_numpy(df['item_idx'].values).long()
        self.weights = torch.from_numpy(df['weight'].values).float()

    def __len__(self) -> int:
        """Retorna o número de amostras."""
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna tupla (user, item, weight) para um dado índice."""
        return self.users[idx], self.items[idx], self.weights[idx]

def main() -> None:
    """Executa o pipeline de treino, avaliação e registro do modelo no MLflow."""
    logger.info("Iniciando pipeline de treinamento...")
    cfg: Dict[str, Any] = load_config()
    N_RECS = 10
    os.makedirs("artifacts", exist_ok=True)

    # 1. Preparação dos dados
    df_raw = pd.read_csv("shared/data/data_csv/raw/events.csv")
    preprocessor = DefaultEventPreprocessor()
    df = preprocessor.preprocess(df_raw)

    u_map, i_map = preprocessor.get_mappings(df)
    item_to_idx = {v: k for k, v in i_map.items()}
    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    # 2. Instanciação do Modelo
    model = ModelFactory.create_model(
        "neumf_light",
        config={**cfg["model"], "n_users": n_users, "n_items": n_items}
    )

    # 3. Treinamento
    train_dataset = RetailRocketDataset(df)
    loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    criterion = nn.MSELoss()

    with mlflow.start_run(run_name="Train-Neural-NeuMF-MLP"):
        # Registro de artefatos iniciais
        mlflow.log_params({**cfg["model"], **cfg["train"]})
        
        for epoch in range(cfg["train"]["epochs"]):
            model.train()
            total_loss = 0.0
            for u, i, w in loader:
                optimizer.zero_grad()
                pred = model(u, i)
                loss = criterion(pred, w)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            mlflow.log_metric("train.loss", avg_loss, step=epoch)
            logger.info(f"Época [{epoch + 1}/{cfg['train']['epochs']}] Loss = {avg_loss:.6f}")

        # 4. Avaliação
        gt_dict = df.groupby("user_idx")["item_idx"].apply(list).to_dict()
        resultados = avaliar_sistema_recomendacao(
            recommend_fn=recommend_wrapper_mlp_fast,
            test_users=df["user_idx"].unique().tolist(),
            gt_dict=gt_dict,
            n_items_total=len(item_to_idx),
            k=N_RECS,
            model=model,
            idx_to_item=i_map,
            user_to_idx=u_map,
            item_to_idx=item_to_idx
        )
        mlflow.log_metrics({f"eval.precision_at_{N_RECS}": resultados[f"Precision@{N_RECS}"], 
                           "eval.catalog_coverage": resultados["Coverage"]})

        # 5. Registro do Modelo
        signature = ModelSignature(
            inputs=Schema([ColSpec("long", "user_idx"), ColSpec("long", "item_idx")]),
            outputs=Schema([ColSpec("double", "prediction")])
        )
        
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="Neural-NeuMF-MLP"
        )
        
        logger.info("Treino finalizado e modelo registrado.")

if __name__ == "__main__":
    main()