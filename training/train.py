import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.pytorch
from torch.utils.data import Dataset, DataLoader

# Importações da camada compartilhada
from shared.ml.model_factory import ModelFactory
from shared.ml.evaluate_metrics import avaliar_sistema_recomendacao
from shared.data.preprocessing import DefaultEventPreprocessor
from shared.utils.config import load_config

# --- Wrapper para a função de recomendação ---
def recommend_wrapper_mlp_fast(user_idx, model, k, item_to_idx, idx_to_item, n_items_total):
    """Transforma a predição do modelo em uma lista de recomendações."""
    model.eval()
    with torch.no_grad():
        u_t = torch.full((n_items_total,), user_idx, dtype=torch.long)
        i_t = torch.arange(n_items_total, dtype=torch.long)
        scores = model(u_t, i_t).numpy()
        
    top_indices = np.argsort(scores)[::-1][:k]
    return [idx_to_item[idx] for idx in top_indices]

# --- Dataset ---
class RetailRocketDataset(Dataset):
    def __init__(self, df):
        self.users = torch.from_numpy(df['user_idx'].values).long()
        self.items = torch.from_numpy(df['item_idx'].values).long()
        self.weights = torch.from_numpy(df['weight'].values).float()
    def __len__(self): return len(self.users)
    def __getitem__(self, idx): return self.users[idx], self.items[idx], self.weights[idx]

def main():
    # 0. Configurações
    cfg = load_config()
    N_RECS = 10
    
    # 1. Preparação
    df_raw = pd.read_csv("data/raw/events.csv")
    preprocessor = DefaultEventPreprocessor()
    df = preprocessor.preprocess(df_raw)
    
    # Obtendo mapeamentos do preprocessor
    u_map, i_map = preprocessor.get_mappings(df)
    item_to_idx = {v: k for k, v in i_map.items()} # Se necessário ajustar a lógica
    
    n_users = df['user_idx'].nunique()
    n_items = df['item_idx'].nunique()
    
    # 2. Modelo
    model = ModelFactory.get_model("neumf_light", n_users=n_users, n_items=n_items, **cfg['model'])
    
    # 3. Treino
    dataset = RetailRocketDataset(df)
    loader = DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    criterion = nn.MSELoss()

    for epoch in range(cfg['train']['epochs']):
        model.train()
        for u, i, w in loader:
            optimizer.zero_grad()
            loss = criterion(model(u, i), w)
            loss.backward()
            optimizer.step()
    
    # 4. Avaliação e MLflow
    with mlflow.start_run(run_name="Neural-NeuMF-MLP"):
        mlflow.log_params(cfg['model'])
        mlflow.log_params(cfg['train'])
        
        gt_dict = df.groupby('user_idx')['item_idx'].apply(list).to_dict()
        test_users = df['user_idx'].unique().tolist()

        print("Iniciando a avaliacao...")
        resultados = avaliar_sistema_recomendacao(
            recommend_fn=recommend_wrapper_mlp_fast,
            test_users=test_users,
            gt_dict=gt_dict,
            n_items_total=n_items,
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

        mlflow.pytorch.log_model(model, "modelo-neumf")
        print("Treino finalizado e logs enviados ao MLflow.")

if __name__ == "__main__":
    main()