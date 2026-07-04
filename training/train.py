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
def recommend_wrapper_mlp_fast(user_idx, k, n_items_total, model, **kwargs):
    model.eval()
    idx_to_item = kwargs.get('idx_to_item')
    
    if idx_to_item is None:
        raise ValueError("O argumento 'idx_to_item' é obrigatório no kwargs.")

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
    def __getitem__(self, idx): return self.users[idx], self.items[idx], self.weights[idx]

def main():
    # 0. Configurações
    cfg = load_config()
    N_RECS = 10
    
    # 1. Preparação
    df_raw = pd.read_csv("shared/data/data_csv/raw/events.csv")
    preprocessor = DefaultEventPreprocessor()
    df = preprocessor.preprocess(df_raw)
    
    # Obtendo mapeamentos do preprocessor
    u_map, i_map = preprocessor.get_mappings(df)
    item_to_idx = {v: k for k, v in i_map.items()} 
    
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
        total_loss = 0.0
        
        for u, i, w in loader:
            optimizer.zero_grad()
            predictions = model(u, i)
            loss = criterion(predictions, w)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calcula a média da perda nesta época
        avg_loss = total_loss / len(loader)
        
        # Print de verificação no console
        print(f"Época [{epoch+1}/{cfg['train']['epochs']}] - Loss: {avg_loss:.4f}")
        
        # (Opcional) Log da perda no MLflow para monitoramento gráfico
        mlflow.log_metric("train.loss", avg_loss, step=epoch)
    
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
            n_items_total= len(item_to_idx),
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