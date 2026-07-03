import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Dataset, DataLoader
import mlflow

# Importações da camada compartilhada
from shared.ml.model_factory import ModelFactory
from shared.ml.evaluate_metrics import avaliar_sistema_recomendacao
from shared.data.preprocessing import DefaultEventPreprocessor


# Carrega a configuração global
def load_config(config_path="configs/model_params.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class RetailRocketDataset(Dataset):
    def __init__(self, df):
        self.users = torch.from_numpy(df['user_idx'].values).long()
        self.items = torch.from_numpy(df['item_idx'].values).long()
        self.weights = torch.from_numpy(df['weight'].values).float()
    def __len__(self): return len(self.users)
    def __getitem__(self, idx): return self.users[idx], self.items[idx], self.weights[idx]


def main():

    # 0. Carregar configurações
    cfg = load_config()
    
    # 1. Preparação (usando Strategy de Preprocessing)
    df_raw = pd.read_csv("data/raw/events.csv")
    preprocessor = DefaultEventPreprocessor()
    df = preprocessor.preprocess(df_raw)
    
    # Lógica de mapeamento (pode ser movida para shared/data no futuro)
    n_users = df['user_idx'].nunique()
    n_items = df['item_idx'].nunique()
    
   # 2. Criação do Modelo via Factory usando params do YAML
    model = ModelFactory.get_model(
        "neumf_light", 
        n_users=n_users, 
        n_items=n_items,
        mf_dim=cfg['model']['mf_dim']
    )
    
    # 3. Treinamento usando params do YAML
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
    
    # 4. Avaliação (mantendo o uso do MLflow com os parâmetros lidos)
    with mlflow.start_run(run_name="Neural-NeuMF-MLP"):
        mlflow.log_params(cfg['model'])
        mlflow.log_params(cfg['train'])
        
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
        
    # 5. Persistência
    torch.save(model.state_dict(), 'models/neumf_model.pth')
    print("Treino finalizado e modelo salvo.")

if __name__ == "__main__":
    main()