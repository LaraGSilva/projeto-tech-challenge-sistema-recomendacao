import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from torch import nn
from torch.utils.data import DataLoader, Dataset

from shared.data.preprocessing import DefaultEventPreprocessor
from shared.ml.evaluate_metrics import avaliar_sistema_recomendacao
from shared.ml.model_factory import ModelFactory
from shared.utils.config import load_config

# Configuração do Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("models/neural")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuração de ambiente
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
mlflow.set_experiment("retailrocket-recommender")


def recommend_wrapper_mlp_fast(
    user_idx: int, k: int, n_items_total: int, model: nn.Module, **kwargs: Any
) -> List[int]:
    """Gera recomendações usando o modelo MLP/NeuMF."""
    model.eval()
    idx_to_item: Dict[int, int] = kwargs.get("idx_to_item")
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
        self.users = torch.from_numpy(df["user_idx"].values).long()
        self.items = torch.from_numpy(df["item_idx"].values).long()
        self.weights = torch.from_numpy(df["weight"].values).float()

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.weights[idx]


# ruff: noqa: PLR0915
def main() -> None:
    logger.info("Iniciando pipeline de treinamento...")
    print("Iniciando pipeline de treinamento...", flush=True)

    print("carregando config...", flush=True)
    cfg: Dict[str, Any] = load_config()
    n_recs = 10

    # 1. Preparação dos dados
    print("import dataset...", flush=True)
    df_raw = pd.read_csv("shared/data/data_csv/raw/events.csv")

    print("import preporcessor...", flush=True)
    preprocessor = DefaultEventPreprocessor()

    print("df preprocessor...", flush=True)
    df = preprocessor.preprocess(df_raw)

    print("get mappings...", flush=True)
    u_map, i_map = preprocessor.get_mappings(df)
    item_to_idx = {v: k for k, v in i_map.items()}
    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()
    n_categorias = 1
    item_to_cat_array = [0] * n_items

    # Criação do arquivo de mapeamento local para salvar como artefato
    mappings_to_save = {"user_to_idx": u_map, "item_to_idx": item_to_idx}
    mapping_path = OUTPUT_DIR / "mappings.npy"
    np.save(mapping_path, mappings_to_save)
    print("mappings salvos")

    # 2. Instanciação do Modelo via Factory
    model = ModelFactory.create_model(
        "neumf_light",
        config={
            **cfg["model"],
            "n_users": n_users,
            "n_items": n_items,
            "n_categorias": n_categorias,
            "item_to_cat_array": item_to_cat_array,
        },
    )

    # 3. Treinamento
    train_dataset = RetailRocketDataset(df)
    loader = DataLoader(
        train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    criterion = nn.MSELoss()
    print("inicio treino")

    with mlflow.start_run(run_name="Train-Neural-NeuMF-MLP"):
        dataset_mlflow = mlflow.data.from_pandas(
            df_raw, name="dataset_RetailRocket_Events_and_Properties"
        )
        mlflow.log_input(dataset_mlflow, context="training")
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
            mlflow.log_metric("train.loss", total_loss / len(loader), step=epoch)
            print(f'"train.loss"{total_loss}')

        print("iniciando avaliacao")
        # 4. Avaliação
        gt_dict = df.groupby("user_idx")["item_idx"].apply(list).to_dict()
        resultados = avaliar_sistema_recomendacao(
            recommend_fn=recommend_wrapper_mlp_fast,
            test_users=df["user_idx"].unique().tolist()[:20],
            gt_dict=gt_dict,
            n_items_total=df["item_idx"].nunique(),
            k=n_recs,
            model=model,
            idx_to_item=i_map,
            user_to_idx=u_map,
            item_to_idx=item_to_idx,
        )

        print("log de metricas")
        mlflow.log_metrics(
            {
                f"eval.precision_at_{n_recs}": resultados[f"Precision{n_recs}"],
                "eval.catalog_coverage": resultados["Coverage"],
            }
        )

        print("salvando o modelo")
        # 5. Registro de Artefatos e Modelo
        # Salvando o peso do modelo (.pt) e o mapeamento (.npy)
        model_pt_path = OUTPUT_DIR / "model.pth"
        torch.save(model.state_dict(), model_pt_path)

        print("artefatos")
        mlflow.log_artifact(str(model_pt_path), artifact_path="model")
        mlflow.log_artifact(str(mapping_path), artifact_path="model")

        print("salvando o modelo")
        # Log do modelo PyTorch no MLflow
        signature = ModelSignature(
            inputs=Schema([ColSpec("long", "user_idx"), ColSpec("long", "item_idx")]),
            outputs=Schema([ColSpec("double", "prediction")]),
        )
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="Neural-NeuMF-MLP",
            signature=signature,
            registered_model_name="Neural-NeuMF-MLP",
        )

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name="Neural-NeuMF-MLP",
            alias="production",
            version=model_info.registered_model_version,
        )

        logger.info("Treino finalizado e modelo registrado com artefatos.")


if __name__ == "__main__":
    main()
