"""
Módulo de treinamento do modelo NeuMF otimizado para baixa memória.
"""
import sys, gc, torch, torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

raiz_projeto = Path(__file__).resolve().parents[3]
sys.path.append(str(raiz_projeto))

# 2. Agora os imports funcionam porque a raiz está no sys.path
from src.recommender.evaluation.evaluate import avaliar_sistema_recomendacao

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
    def __getitem__(self, idx): return self.users[idx], self.items[idx], self.weights[idx]

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
    df_pd = pd.read_csv("data/interim/events.csv", 
                        usecols=['visitorid', 'itemid', 'event', 'timestamp'],
                        dtype={'visitorid': 'int32', 'itemid': 'int32'})
    
    peso_map = {'view': 1, 'addtocart': 3, 'transaction': 5}
    df_pd['weight'] = df_pd['event'].map(peso_map).fillna(0).astype('int8')
    
    # Criando os mapeamentos (usando categorias para economia de memória)
    df_pd['user_idx'] = df_pd['visitorid'].astype('category').cat.codes
    df_pd['item_idx'] = df_pd['itemid'].astype('category').cat.codes
    
    # Criando dicionários de mapeamento para o evaluador
    user_to_idx = dict(enumerate(df_pd['visitorid'].astype('category').cat.categories))
    idx_to_item = {i: item for i, item in enumerate(df_pd['itemid'].astype('category').cat.categories)}
    item_to_idx = {item: i for i, item in enumerate(df_pd['itemid'].astype('category').cat.categories)}
    
    return df_pd, user_to_idx, idx_to_item, item_to_idx

def recommend_wrapper_mlp_fast(visitor_id, model, candidatos, idx_to_item, user_to_idx, item_to_idx, n=10, device="cpu"):
    if visitor_id not in user_to_idx: return []
    u_idx = user_to_idx[visitor_id]
    candidatos_validos = [item_to_idx[c] for c in candidatos if c in item_to_idx]
    
    if not candidatos_validos: return []
    
    model.eval()
    with torch.no_grad():
        u_t = torch.full((len(candidatos_validos),), u_idx, device=device)
        i_t = torch.tensor(candidatos_validos, device=device)
        scores = model(u_t, i_t).cpu().numpy()
        
    top_items = [idx_to_item[candidatos_validos[i]] for i in np.argsort(scores)[::-1][:n]]
    return top_items

# --- MAIN ---

def main():
    try:
        df, u_map, i_map, item_to_idx = load_and_prep_data()
        n_users, n_items = int(df['user_idx'].max()) + 1, int(df['item_idx'].max()) + 1
        
        dataset = RetailRocketDataset(df)
        loader = DataLoader(dataset, batch_size=4096, shuffle=True)
        model = NeuMF_Light(n_users, n_items)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

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
            # Exemplo de gt_dict (você deve adaptar conforme sua função de avaliação)
            gt_dict = df.groupby('visitorid')['itemid'].apply(list).to_dict()
            
            resultados = avaliar_sistema_recomendacao(
                recommend_fn=recommend_wrapper_mlp_fast,
                test_users=list(u_map.values()),
                gt_dict=gt_dict,
                n_items_total=n_items,
                k=10,
                model=model,
                idx_to_item=i_map,
                user_to_idx={v: k for k, v in u_map.items()},
                item_to_idx=item_to_idx
            )
            
            mlflow.log_metrics({f"eval.{k}": v for k, v in resultados.items() if isinstance(v, (int, float))})
            mlflow.pytorch.log_model(model, "modelo-neumf")
            
        print("Processo finalizado!")

    except Exception as e:
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()