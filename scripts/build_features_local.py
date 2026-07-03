# scripts/build_features_local.py
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

# carrega o que você já tem do EDA
df = pd.read_csv("data/raw/events.csv")

df["peso"] = df["event"].map({
    "view": 1,
    "addtocart": 3,
    "transaction": 5,
})

interacoes = (
    df.groupby(["visitorid", "itemid"])["peso"]
    .sum()
    .reset_index()
)

# filtrar usuários frios (menos de 5 interações)
contagem = interacoes.groupby("visitorid")["itemid"].count()
usuarios_ativos = contagem[contagem >= 5].index
interacoes = interacoes[interacoes["visitorid"].isin(usuarios_ativos)]

# mappings id → índice
user_to_idx = {u: i for i, u in enumerate(interacoes["visitorid"].unique())}
item_to_idx = {it: i for i, it in enumerate(interacoes["itemid"].unique())}
idx_to_user = {i: u for u, i in user_to_idx.items()}
idx_to_item = {i: it for it, i in item_to_idx.items()}

rows = interacoes["visitorid"].map(user_to_idx)
cols = interacoes["itemid"].map(item_to_idx)
data = interacoes["peso"]

matrix = csr_matrix((data, (rows, cols)))

# salvar
Path("data/features").mkdir(parents=True, exist_ok=True)
save_npz("data/features/user_item_matrix.npz", matrix)

with open("data/features/mappings.pkl", "wb") as f:
    pickle.dump({
        "idx_to_user": idx_to_user, # Agora sim, o inverso
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
    }, f)

print(f"matriz: {matrix.shape} | usuários: {len(user_to_idx)} | itens: {len(item_to_idx)}")