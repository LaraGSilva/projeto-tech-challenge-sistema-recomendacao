import mlflow
from mlflow.types import ColSpec, Schema
import mlflow.pytorch
from mlflow.models import ModelSignature, infer_signature

from shared.utils.config import load_config
from shared.data.preprocessing import DefaultEventPreprocessor
from shared.ml.evaluate_metrics import avaliar_sistema_recomendacao
from shared.ml.model_factory import ModelFactory

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import os


os.environ["GIT_PYTHON_REFRESH"] = "quiet"



mlflow.set_experiment("retailrocket-recommender")
ARTIFACT_DIR = "/data/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --- Wrapper para a função de recomendação ---
def recommend_wrapper_mlp_fast(user_idx, k, n_items_total, model, **kwargs):
    model.eval()
    idx_to_item = kwargs.get('idx_to_item')

    if idx_to_item is None:
        raise ValueError(
            "O argumento 'idx_to_item' é obrigatório no kwargs.")

    with torch.no_grad():
        u_t = torch.full((n_items_total,), user_idx, dtype=torch.long)
        i_t = torch.arange(n_items_total, dtype=torch.long)
        scores = model(u_t, i_t).numpy()

    # Ordena os scores
    top_indices = np.argsort(scores)[::-1]

    recs = []
    for idx in top_indices:
        idx_int = int(idx)

        if idx_int in idx_to_item:
            recs.append(idx_to_item[idx_int])

        if len(recs) == k:
            break

    return recs

# --- Dataset ---
class RetailRocketDataset(Dataset):
    def __init__(self, df):
        self.users = torch.from_numpy(df['user_idx'].values).long()
        self.items = torch.from_numpy(df['item_idx'].values).long()
        self.weights = torch.from_numpy(df['weight'].values).float()

    def __len__(self): return len(self.users)
    def __getitem__(
        self, idx): return self.users[idx], self.items[idx], self.weights[idx]


# def main():
#     # 0. Configurações
#     cfg = load_config()
#     N_RECS = 10

#     # 1. Preparação
#     df_raw = pd.read_csv("shared/data/data_csv/raw/events.csv")
#     preprocessor = DefaultEventPreprocessor()
#     df = preprocessor.preprocess(df_raw)

#     # Obtendo mapeamentos do preprocessor
#     u_map, i_map = preprocessor.get_mappings(df)
#     item_to_idx = {v: k for k, v in i_map.items()}

#     n_users = df['user_idx'].nunique()
#     n_items = df['item_idx'].nunique()

#     # 2. Modelo
#     model = ModelFactory.get_model(
#         "neumf_light", n_users=n_users, n_items=n_items, **cfg['model'])

#     # 3. Treino
#     dataset = RetailRocketDataset(df)
#     loader = DataLoader(
#         dataset, batch_size=cfg['train']['batch_size'], shuffle=True)
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=cfg['train']['learning_rate'])
#     criterion = nn.MSELoss()

#     with mlflow.start_run(run_name="Train-Neural-NeuMF-MLP"):
#         print("="*80)
#         print("RUN ID:", mlflow.active_run().info.run_id)
#         print("Artifact URI:", mlflow.get_artifact_uri())
#         print("="*80)

#         df_dataset = pd.DataFrame(df_raw)
#         dataset = mlflow.data.from_pandas(
#             df_dataset, name="dataset_RetailRocket_Events_and_Properties")

#         mlflow.log_input(dataset, context="training/test")

#         for epoch in range(cfg['train']['epochs']):
#             model.train()
#             total_loss = 0.0

#             for u, i, w in loader:
#                 optimizer.zero_grad()
#                 predictions = model(u, i)
#                 loss = criterion(predictions, w)
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item()

#             # Calcula a média da perda nesta época
#             avg_loss = total_loss / len(loader)

#             # Print de verificação no console
#             print(f"Época [{epoch+1}/{cfg['train']
#                   ['epochs']}] - Loss: {avg_loss:.4f}")
#             mlflow.log_metric("train.loss", avg_loss, step=epoch)

#         # 4. Avaliação e MLflow
#         mlflow.log_params(cfg['model'])
#         mlflow.log_params(cfg['train'])

#         gt_dict = df.groupby('user_idx')['item_idx'].apply(list).to_dict()
#         test_users = df['user_idx'].unique().tolist()

#         print("Iniciando a avaliacao...")
#         resultados = avaliar_sistema_recomendacao(
#             recommend_fn=recommend_wrapper_mlp_fast,
#             test_users=test_users,
#             gt_dict=gt_dict,
#             n_items_total=len(item_to_idx),
#             k=N_RECS,
#             model=model,
#             idx_to_item=i_map,
#             user_to_idx=u_map,
#             item_to_idx=item_to_idx
#         )

#         mlflow.log_metrics({
#             f"eval.precision_at_{N_RECS}": resultados[f"Precision@{N_RECS}"],
#             f"eval.recall_at_{N_RECS}": resultados[f"Recall@{N_RECS}"],
#             f"eval.ndcg_at_{N_RECS}": resultados[f"NDCG@{N_RECS}"],
#             "eval.catalog_coverage": resultados["Coverage"]
#         })

#         example_u = torch.tensor([0], dtype=torch.long)
#         example_i = torch.tensor([0], dtype=torch.long)
#         input_example = (example_u, example_i)

#         output_example = model(example_u, example_i)

#         signature = infer_signature(input_example, output_example)

#         signature = ModelSignature(
#             inputs=Schema([ColSpec("long", "user_idx"),
#                           ColSpec("long", "item_idx")]),
#             outputs=Schema([ColSpec("double", "prediction")]))

#         # model_name = "Neural-NeuMF-MLP"
#         # mlflow.pytorch.log_model(
#         #     pytorch_model=model,
#         #     name="model",
#         #     signature=signature,
#         #     registered_model_name=model_name,
#         #     input_example=input_example,
#         #     serialization_format="pickle"
#         # )

#         # model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
#         # model_version = mlflow.register_model(model_uri, model_name)

#         # client = mlflow.tracking.MlflowClient()
#         # client.set_registered_model_alias(
#         #     name=model_name,
#         #     alias="Production",
#         #     version=model_version.version)

#         # # Salva mapeamentos como artefato
#         mappings = {"user_to_idx": u_map, "item_to_idx": item_to_idx}
#         # np.save("mappings.npy", mappings)
#         # mlflow.log_artifact("mappings.npy")

#         model_name = "Neural-NeuMF-MLP"

#         logged_model = mlflow.pytorch.log_model(
#             pytorch_model=model,
#             name="model",
#             signature=signature,
#             input_example=input_example,
#             registered_model_name=model_name
#         )

#         client = mlflow.tracking.MlflowClient()

#         latest = client.search_model_versions()

#         client.set_registered_model_alias(
#             name=model_name,
#             alias="Production",
#             version=latest
#         )

#         os.makedirs("artifacts", exist_ok=True)

#         mapping_file = "artifacts/mappings.npy"

#         np.save(mapping_file, mappings)

#         mlflow.log_artifact(mapping_file)

#         print("Mappings salvos.")

#         print("Modelo registrado:", latest.version)

#         print(f"Modelo {model_name} v{model_version.version} movido para Production.")
#         print("Treino finalizado e logs enviados ao MLflow.")

# if __name__ == "__main__":
#     main()

def main():
    print("Inicio...")
    # ==========================================================================
    # 0. Configurações
    # ==========================================================================
    cfg = load_config()
    N_RECS = 10

    os.makedirs("artifacts", exist_ok=True)

    with open("artifacts/config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ==========================================================================
    # 1. Preparação
    # ==========================================================================
    df_raw = pd.read_csv("shared/data/data_csv/raw/events.csv")

    preprocessor = DefaultEventPreprocessor()
    df = preprocessor.preprocess(df_raw)

    u_map, i_map = preprocessor.get_mappings(df)
    item_to_idx = {v: k for k, v in i_map.items()}

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    # ==========================================================================
    # 2. Modelo
    # ==========================================================================
    model = ModelFactory.get_model(
        "neumf_light",
        n_users=n_users,
        n_items=n_items,
        **cfg["model"]
    )

    # ==========================================================================
    # 3. DataLoader
    # ==========================================================================
    train_dataset = RetailRocketDataset(df)

    loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"]
    )

    criterion = nn.MSELoss()

    # ==========================================================================
    # MLflow
    # ==========================================================================
    with mlflow.start_run(run_name="Train-Neural-NeuMF-MLP"):

        print("=" * 80)
        print("Run ID:", mlflow.active_run().info.run_id)
        print("Tracking URI:", mlflow.get_tracking_uri())
        print("Artifact URI:", mlflow.get_artifact_uri())
        print("=" * 80)

        dataset_mlflow = mlflow.data.from_pandas(
            df_raw,
            name="dataset_RetailRocket_Events_and_Properties"
        )

        mlflow.log_input(dataset_mlflow, context="training")

        mlflow.log_artifact("artifacts/config.yaml")

        mlflow.log_params(cfg["model"])
        mlflow.log_params(cfg["train"])

        # ==========================================================
        # Treino
        # ==========================================================
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

            print(
                f"Época [{epoch + 1}/{cfg['train']['epochs']}] "
                f"Loss = {avg_loss:.6f}"
            )

            mlflow.log_metric(
                "train.loss",
                avg_loss,
                step=epoch
            )

        # ==========================================================
        # Avaliação
        # ==========================================================
        gt_dict = (
            df.groupby("user_idx")["item_idx"]
            .apply(list)
            .to_dict()
        )

        test_users = df["user_idx"].unique().tolist()

        print("Iniciando avaliação...")

        resultados = avaliar_sistema_recomendacao(
            recommend_fn=recommend_wrapper_mlp_fast,
            test_users=test_users,
            gt_dict=gt_dict,
            n_items_total=len(item_to_idx),
            k=N_RECS,
            model=model,
            idx_to_item=i_map,
            user_to_idx=u_map,
            item_to_idx=item_to_idx
        )

        mlflow.log_metrics({
            f"eval.precision_at_{N_RECS}": resultados[f"Precision@{N_RECS}"],
            f"eval.recall_at_{N_RECS}": resultados[f"Recall@{N_RECS}"],
            f"eval.ndcg_at_{N_RECS}": resultados[f"NDCG@{N_RECS}"],
            "eval.catalog_coverage": resultados["Coverage"]
        })

        metrics_df = pd.DataFrame([resultados])

        metrics_file = "artifacts/metrics.csv"

        metrics_df.to_csv(metrics_file, index=False)

        mlflow.log_artifact(metrics_file)

        # ==========================================================
        # Signature
        # ==========================================================
        example_u = torch.tensor([0], dtype=torch.long)
        example_i = torch.tensor([0], dtype=torch.long)

        input_example = (example_u, example_i)

        output_example = model(example_u, example_i)

        infer_signature(input_example, output_example)

        signature = ModelSignature(
            inputs=Schema([
                ColSpec("long", "user_idx"),
                ColSpec("long", "item_idx")
            ]),
            outputs=Schema([
                ColSpec("double", "prediction")
            ])
        )

        # ==========================================================
        # Modelo
        # ==========================================================
        model_name = "Neural-NeuMF-MLP"

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name
        )

        # ==========================================================
        # Recupera última versão registrada
        # ==========================================================
        client = mlflow.tracking.MlflowClient()

        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        latest_version = max(
            versions,
            key=lambda v: int(v.version)
        )

        client.set_registered_model_alias(
            name=model_name,
            alias="Production",
            version=latest_version.version
        )

        # ==========================================================
        # Mapeamentos
        # ==========================================================
        mappings = {
            "user_to_idx": u_map,
            "item_to_idx": item_to_idx
        }

        mapping_file = "artifacts/mappings.npy"

        np.save(mapping_file, mappings)

        mlflow.log_artifact(mapping_file)

        # ==========================================================
        # Logs finais
        # ==========================================================
        print("=" * 80)
        print("Modelo registrado:", model_name)
        print("Versão:", latest_version.version)
        print("Alias: Production")
        print("Artifact URI:", mlflow.get_artifact_uri())
        print("=" * 80)

        print("Treino finalizado com sucesso.")


if __name__ == "__main__":
    main()