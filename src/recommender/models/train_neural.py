"""
Módulo de treinamento do modelo NeuMF otimizado para baixa memória.
"""
from pathlib import Path
import sys

raiz_projeto = Path(__file__).resolve().parents[3]
sys.path.append(str(raiz_projeto))

from shared.ml.evaluate_metrics import avaliar_sistema_recomendacao

import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature


# 2. Agora os imports funcionam porque a raiz está no sys.path

tracking_uri = "http://recommender_mlflow:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("retailrocket-recommender")


# --- 1. CONFIGURAÇÕES E CLASSES ---

class RetailRocketDataset(Dataset):
    def __init__(self, df):
        self.users = torch.from_numpy(df['user_idx'].values).long()
        self.items = torch.from_numpy(df['item_idx'].values).long()
        self.weights = torch.from_numpy(df['weight'].values).float()

    def __len__(self): return len(self.users)
    def __getitem__(
        self, idx): return self.users[idx], self.items[idx], self.weights[idx]


class NeuMF_Light(nn.Module):
    def __init__(self, n_users, n_items, mf_dim=16):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, mf_dim)
        self.item_embed = nn.Embedding(n_items, mf_dim)
        self.fc = nn.Linear(mf_dim, 1)

    def forward(self, u, i):
        return (self.user_embed(u) * self.item_embed(i)).sum(dim=-1)


def load_and_prep_data():
    print("Lendo CSV diretamente com Pandas...")
    df_pd = pd.read_csv("data/raw/events.csv",
                        usecols=['visitorid', 'itemid', 'event', 'timestamp'],
                        dtype={'visitorid': 'int32', 'itemid': 'int32'})

    peso_map = {'view': 1, 'addtocart': 3, 'transaction': 5}
    df_pd['weight'] = df_pd['event'].map(peso_map).fillna(0).astype('int8')

    # Criando os mapeamentos (usando categorias para economia de memória)
    print("Mapeando os pesos...")
    df_pd['user_idx'] = df_pd['visitorid'].astype('category').cat.codes
    df_pd['item_idx'] = df_pd['itemid'].astype('category').cat.codes

    print("Criando dicionarios de mapeamento para o evaluad")
    # Criando dicionários de mapeamento para o evaluador
    user_to_idx = dict(
        enumerate(df_pd['visitorid'].astype('category').cat.categories))
    idx_to_item = {i: item for i, item in enumerate(
        df_pd['itemid'].astype('category').cat.categories)}
    item_to_idx = {item: i for i, item in enumerate(
        df_pd['itemid'].astype('category').cat.categories)}
    return df_pd, user_to_idx, idx_to_item, item_to_idx


def recommend_wrapper_mlp_fast(visitor_id, model, candidatos, idx_to_item, user_to_idx, item_to_idx, n=10, device="cpu"):
    if visitor_id not in user_to_idx:
        return []
    u_idx = user_to_idx[visitor_id]
    candidatos_validos = [item_to_idx[c]
                          for c in candidatos if c in item_to_idx]

    if not candidatos_validos:
        return []

    model.eval()
    with torch.no_grad():
        u_t = torch.full((len(candidatos_validos),), u_idx, device=device)
        i_t = torch.tensor(candidatos_validos, device=device)
        scores = model(u_t, i_t).cpu().numpy()

    top_items = [idx_to_item[candidatos_validos[i]]
                 for i in np.argsort(scores)[::-1][:n]]
    return top_items

# --- MAIN ---


def main():
    try:

        
        N_RECS = 10

        df, u_map, i_map, item_to_idx = load_and_prep_data()
        n_users, n_items = int(df['user_idx'].max()) + \
            1, int(df['item_idx'].max()) + 1

        dataset = RetailRocketDataset(df)
        loader = DataLoader(dataset, batch_size=4096, shuffle=True)
        model = NeuMF_Light(n_users, n_items)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        print("Iniciando o treinamento do modelo")
        for epoch in range(3):
            model.train()
            total_loss = 0
            for u, i, w in loader:
                optimizer.zero_grad()
                pred = model(u, i)
                loss = criterion(pred, w)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Época {epoch+1} - Loss: {total_loss/len(loader):.4f}")

        # Avaliação e MLflow
        with mlflow.start_run(run_name="Neural-NeuMF-MLP"):
            gt_dict = df.groupby('visitorid')['itemid'].apply(list).to_dict()
            lista_usuarios_teste = df['visitorid'].unique().tolist()

        print("Iniciando a avaliacao do sistema recomendacao")
        resultados_experimentos = avaliar_sistema_recomendacao(
            recommend_fn=recommend_wrapper_mlp_fast,
            test_users=lista_usuarios_teste,     # Agora é uma lista iterável
            gt_dict=gt_dict,
            n_items_total=n_items,               # Aqui sim, o número total (int)
            k=N_RECS,
            model=model,
            idx_to_item=i_map,                   # Passando o dicionário i_map
            user_to_idx={v: k for k, v in u_map.items()}, # Invertendo u_map (ID -> Index)
            item_to_idx=item_to_idx              # Passando o dicionário item_to_idx
        )

        # 2. Registra logs no MLflow
        dataset_mlp = mlflow.data.from_pandas(
            dataset, name="dataset_RetailRocket_Events")
        mlflow.log_input(dataset_mlp, context="training/test")

        mlflow.set_tags({
            "model_type": "NeuMF_Neural_Collaborative_Filtering",
            "framework": "pytorch",
            "phase": "neural_model"
        })

        mlflow.log_params({
            "model.mf_dim": 16,
            "model.mlp_dim": 32,
            "train.optimizer": "AdamW",
            "train.epochs": 3,
            "train.batch_size": 4096

        })

        resultados = avaliar_sistema_recomendacao(
            recommend_fn=recommend_wrapper_mlp_fast,
            test_users=list(u_map.values()),
            gt_dict=gt_dict,
            n_items_total=n_items,
            k=N_RECS,
            model=model,
            idx_to_item=i_map,
            user_to_idx={v: k for k, v in u_map.items()},
            item_to_idx=item_to_idx
        )

        print(resultados)

        mlflow.log_metrics({
            "eval.precision_at_10": resultados_experimentos[f"Precision@{N_RECS}"],
            "eval.recall_at_10": resultados_experimentos[f"Recall@{N_RECS}"],
            "eval.ndcg_at_10": resultados_experimentos[f"NDCG@{N_RECS}"],
            "eval.catalog_coverage": resultados_experimentos["Coverage"]
        })

        mlflow.pytorch.log_model(model, "modelo-neumf")
        print("Processo finalizado!")

        # Salvar para o Service
        mappings = {
            'user_to_idx': {v: k for k, v in u_map.items()}, # Invertido para o formato do service
            'item_to_idx': item_to_idx
        }
        np.save('models/mappings.npy', mappings)
        torch.save(model.state_dict(), 'models/neumf_model.pth')
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
