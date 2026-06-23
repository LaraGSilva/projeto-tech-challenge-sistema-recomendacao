"""
Módulo de treinamento do modelo de recomendação Neural Collaborative Filtering (NeuMF).
Este script processa os dados, treina o modelo PyTorch e registra as métricas no MLflow.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from typing import Dict, List, Tuple, Any
from src.recommender.evaluation.evaluate import avaliar_sistema_recomendacao

# =============================================================================
# 1. CLASSES DO PYTORCH (Dataset, Modelo e Loss)
# =============================================================================


class RetailRocketBPRDatasetCPU(Dataset):
    """Dataset otimizado para CPU. Sorteia itens negativos de forma vetorizada."""

    def __init__(self, df_interactions: pd.DataFrame, item_popularity: Dict[int, int], n_items: int, power: float = 0.75):
        self.users = torch.tensor(
            df_interactions['user_idx'].values, dtype=torch.long)
        self.pos_items = torch.tensor(
            df_interactions['item_idx'].values, dtype=torch.long)
        self.weights = torch.tensor(
            df_interactions['weight'].values, dtype=torch.float)

        pop_weights = np.array([item_popularity.get(i, 0)
                               for i in range(n_items)])
        pop_weights = np.power(pop_weights, power)
        pop_sum = pop_weights.sum()
        pop_probs = pop_weights / \
            pop_sum if pop_sum > 0 else np.ones(n_items) / n_items

        print("Pré-gerando amostras negativas para otimizar a CPU...")
        self.neg_items = np.random.choice(
            n_items, size=len(self.users), p=pop_probs)

        mask_igual = (self.neg_items == df_interactions['item_idx'].values)
        if mask_igual.any():
            substitutos = np.random.choice(
                n_items, size=mask_igual.sum(), p=pop_probs)
            self.neg_items[mask_igual] = substitutos

        self.neg_items = torch.tensor(self.neg_items, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.pos_items[idx], self.neg_items[idx], self.weights[idx]


class NeuMF_RetailRocket_CPU(nn.Module):
    """Arquitetura NeuMF enxuta otimizada para treinamento em CPU."""

    def __init__(self, n_users: int, n_items: int, n_categorias: int, item_to_cat_array: np.ndarray,
                 mf_dim: int = 16, mlp_dim: int = 32, categoria_dim: int = 8,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()

        self.user_mf_embed = nn.Embedding(n_users, mf_dim)
        self.item_mf_embed = nn.Embedding(n_items, mf_dim)
        self.user_mlp_embed = nn.Embedding(n_users, mlp_dim)
        self.item_mlp_embed = nn.Embedding(n_items, mlp_dim)
        self.cat_mlp_embed = nn.Embedding(n_categorias, categoria_dim)

        self.register_buffer("item_categoria_idx", torch.tensor(
            item_to_cat_array, dtype=torch.long))

        in_dim = mlp_dim * 2 + categoria_dim
        mlp_layers = []
        for h_dim in hidden_dims:
            mlp_layers += [nn.Linear(in_dim, h_dim),
                           nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h_dim
        self.mlp = nn.Sequential(*mlp_layers)

        self.prediction_layer = nn.Linear(mf_dim + hidden_dims[-1], 1)

    def _forward_branch(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_mf = self.user_mf_embed(user_idx)
        item_mf = self.item_mf_embed(item_idx)
        gmf_vector = user_mf * item_mf

        user_mlp = self.user_mlp_embed(user_idx)
        item_mlp = self.item_mlp_embed(item_idx)
        cat_idx = self.item_categoria_idx[item_idx]
        cat_mlp = self.cat_mlp_embed(cat_idx)

        mlp_vector = torch.cat([user_mlp, item_mlp, cat_mlp], dim=-1)
        mlp_vector = self.mlp(mlp_vector)

        fusion = torch.cat([gmf_vector, mlp_vector], dim=-1)
        return self.prediction_layer(fusion).squeeze(-1)

    def forward(self, user_idx: torch.Tensor, pos_item_idx: torch.Tensor, neg_item_idx: torch.Tensor = None):
        if neg_item_idx is not None:
            return self._forward_branch(user_idx, pos_item_idx), self._forward_branch(user_idx, neg_item_idx)
        return self._forward_branch(user_idx, pos_item_idx)


class WeightedBPRLoss(nn.Module):
    """Função de perda BPR ponderada pelo tipo de interação."""

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return - (weights * torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)).mean()


# =============================================================================
# 2. FUNÇÕES DE PREPARAÇÃO DE DADOS (PySpark e Pandas)
# =============================================================================

def get_spark_session() -> SparkSession:
    """Inicializa e retorna a sessão do Spark."""
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("RetailRocket_NeuMF") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "48")
    return spark


def prepare_spark_data(spark: SparkSession) -> Tuple[Any, Any, Any, Any]:
    """Lê os dados brutos e prepara as features base via PySpark."""
    df_events = spark.read.csv(
        "data/interim/events.csv", header=True, inferSchema=True)
    df_prop_1 = spark.read.csv(
        "data/interim/item_properties_part1.csv", header=True, inferSchema=True)
    df_prop_2 = spark.read.csv(
        "data/interim/item_properties_part2.csv", header=True, inferSchema=True)

    df_prop = df_prop_1.unionByName(df_prop_2)
    window_spec = Window.partitionBy(
        "itemid", "property").orderBy(F.col("timestamp").desc())

    df_prop_last = df_prop.withColumn("rn", F.row_number().over(window_spec)) \
                          .filter(F.col("rn") == 1).drop("rn")

    df_items = df_prop_last.filter(F.col("property").isin(["categoryid", "available"])) \
                           .groupBy("itemid").pivot("property").agg(F.first("value"))

    df_eda = df_events.join(df_items, on="itemid", how="left")

    weights_expr = (
        F.when(F.col("event") == "view", F.lit(1))
         .when(F.col("event") == "addtocart", F.lit(3))
         .when(F.col("event") == "transaction", F.lit(5))
         .otherwise(F.lit(0))
    )
    df_eda_com_pesos = df_eda.withColumn("weight", weights_expr)

    cutoff = df_eda_com_pesos.approxQuantile("timestamp", [0.8], 0.01)[0]
    df_treino_raw = df_eda_com_pesos.filter(F.col("timestamp") <= cutoff)
    df_teste_raw = df_eda_com_pesos.filter(F.col("timestamp") > cutoff)

    usuarios_ativos = df_treino_raw.groupBy("visitorid") \
                                   .agg(F.countDistinct("itemid").alias("n_itens")) \
                                   .filter(F.col("n_itens") >= 5).select("visitorid").cache()

    df_treino_spark = df_treino_raw.join(usuarios_ativos, "visitorid")
    df_teste_spark = df_teste_raw.join(usuarios_ativos, "visitorid")

    return df_treino_spark, df_teste_spark, df_eda_com_pesos


def create_mappings(df_treino_spark: Any, df_eda_com_pesos: Any) -> Tuple[Dict, Dict, Dict, Dict, np.ndarray, Dict]:
    """Cria dicionários de mapeamento de IDs e o vetor de categorias."""
    unique_users = [row['visitorid']
                    for row in df_treino_spark.select('visitorid').distinct().collect()]
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}

    unique_items = [row['itemid']
                    for row in df_eda_com_pesos.select('itemid').distinct().collect()]
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    idx_to_item = {idx: iid for idx, iid in enumerate(unique_items)}

    df_eda_com_pesos = df_eda_com_pesos.fillna({"categoryid": "unknown"})
    unique_cats = [row['categoryid'] for row in df_eda_com_pesos.select(
        'categoryid').distinct().collect()]
    cat_to_idx = {cid: idx for idx, cid in enumerate(unique_cats)}

    num_items = len(unique_items)
    item_cat_df = df_eda_com_pesos.select(
        "itemid", "categoryid").distinct().toPandas()
    item_to_category_vector = np.zeros(num_items, dtype=np.int64)

    for _, row in item_cat_df.iterrows():
        if row['itemid'] in item_to_idx:
            i_idx = item_to_idx[row['itemid']]
            c_idx = cat_to_idx[row['categoryid']]
            item_to_category_vector[i_idx] = c_idx

    pop_df = df_treino_spark.groupBy("itemid").count().toPandas()
    item_popularity = {item_to_idx[r['itemid']]: r['count']
                       for _, r in pop_df.iterrows() if r['itemid'] in item_to_idx}

    return user_to_idx, item_to_idx, idx_to_item, cat_to_idx, item_to_category_vector, item_popularity


# =============================================================================
# 3. TREINAMENTO E AVALIAÇÃO
# =============================================================================

def train_model(df_train_pandas: pd.DataFrame, item_popularity: Dict, item_to_category_vector: np.ndarray,
                num_users: int, num_items: int, num_cats: int, device: torch.device) -> nn.Module:
    """Executa o loop de treinamento do modelo."""
    train_dataset = RetailRocketBPRDatasetCPU(
        df_train_pandas, item_popularity, num_items)
    train_loader = DataLoader(
        train_dataset, batch_size=4096, shuffle=True, num_workers=0)

    model = NeuMF_RetailRocket_CPU(
        num_users, num_items, num_cats, item_to_category_vector).to(device)
    criterion = WeightedBPRLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Iniciando treinamento...")
    model.train()
    for epoch in range(1, 4):
        epoch_loss = 0
        for users, pos_items, neg_items, weights in train_loader:
            users, pos_items, neg_items, weights = users.to(device), pos_items.to(
                device), neg_items.to(device), weights.to(device)
            optimizer.zero_grad()
            pos_scores, neg_scores = model(users, pos_items, neg_items)
            loss = criterion(pos_scores, neg_scores, weights)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Época {epoch} Finalizada | Loss Média: {
              epoch_loss/len(train_loader):.4f}")

    return model

# =============================================================================
# 4. FUNÇÃO PRINCIPAL E ORQUESTRAÇÃO
# =============================================================================


def recomendar_mlp(visitor_id, model, candidatos_originais, idx_to_item, user_to_idx, item_to_idx, n=10, device="cpu"):
    """Função base de inferência do modelo PyTorch."""
    if len(candidatos_originais) == 0 or visitor_id not in user_to_idx:
        return []

    user_idx = user_to_idx[visitor_id]
    candidatos_validos = [item_to_idx[c]
                          for c in candidatos_originais if c in item_to_idx]

    if len(candidatos_validos) == 0:
        return []

    model.eval()
    with torch.no_grad():
        user_tensor = torch.full(
            (len(candidatos_validos),), user_idx, dtype=torch.long, device=device)
        item_tensor = torch.tensor(
            candidatos_validos, dtype=torch.long, device=device)
        scores = model(user_tensor, item_tensor).cpu().numpy()

    top_n = min(n, len(candidatos_validos))
    top_idxs_local = np.argsort(scores)[::-1][:top_n]
    top_item_idxs = [candidatos_validos[i] for i in top_idxs_local]

    return [idx_to_item[i] for i in top_item_idxs]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spark = get_spark_session()
    N_RECS = 10  # Definindo a quantidade de recomendações

    print("Preparando dados...")
    df_treino_spark, df_teste_spark, df_eda_com_pesos = prepare_spark_data(
        spark)

    print("Criando mapeamentos e Ground Truth...")
    user_to_idx, item_to_idx, idx_to_item, cat_to_idx, item_to_category_vector, item_popularity = \
        create_mappings(df_treino_spark, df_eda_com_pesos)

    num_users, num_items, num_cats = len(
        user_to_idx), len(item_to_idx), len(cat_to_idx)

    # Construindo o Ground Truth (gt_dict) a partir do dataframe de teste
    ground_truth = df_teste_spark.groupBy("visitorid").agg(
        F.collect_set("itemid").alias("itens_relevantes"))
    gt_dict = {row["visitorid"]: row["itens_relevantes"]
               for row in ground_truth.collect()}
    usuarios_teste_python = list(gt_dict.keys())

    df_train_pandas = df_treino_spark.select(
        "visitorid", "itemid", "weight").toPandas()
    df_train_pandas['user_idx'] = df_train_pandas['visitorid'].map(user_to_idx)
    df_train_pandas['item_idx'] = df_train_pandas['itemid'].map(item_to_idx)
    df_train_pandas = df_train_pandas.dropna(subset=['user_idx', 'item_idx']).astype({
        'user_idx': 'int64', 'item_idx': 'int64'})

    model = train_model(df_train_pandas, item_popularity,
                        item_to_category_vector, num_users, num_items, num_cats, device)

    # Configuração de Candidatos para Avaliação Rápida
    top_100_populares = [item for item, _ in sorted(
        item_popularity.items(), key=lambda x: x[1], reverse=True)[:100]]
    itens_treino_por_usuario = df_train_pandas.groupby(
        'visitorid')['item_idx'].apply(list).to_dict()

    def recommend_wrapper_mlp_fast(user_id, k=10, **kwargs):
        """Wrapper adaptado para alimentar a função de avaliação."""
        mdl = kwargs.get('model')
        i_to_item = kwargs.get('idx_to_item')
        u_to_idx = kwargs.get('user_to_idx')
        i_to_idx = kwargs.get('item_to_idx')

        itens_usuario = itens_treino_por_usuario.get(user_id, [])
        candidatos_internos = list(set(itens_usuario + top_100_populares))
        candidatos_array = np.array(candidatos_internos, dtype=np.int64)
        candidatos_originais = np.array(
            [i_to_item[idx] for idx in candidatos_array if idx in i_to_item])

        return recomendar_mlp(user_id, mdl, candidatos_originais, i_to_item, u_to_idx, i_to_idx, k, device)

    # ---------------------------------------------------------
    # Configuração do MLflow
    # ---------------------------------------------------------
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment('tech-challenge-recomendacao-retailrocket')

    print("Avaliando modelo e registrando no MLflow...")
    with mlflow.start_run(run_name="Neural-NeuMF-MLP") as run:

        # 1. Executa a avaliação do sistema
        resultados_experimentos = avaliar_sistema_recomendacao(
            recommend_fn=recommend_wrapper_mlp_fast,
            test_users=usuarios_teste_python,
            gt_dict=gt_dict,
            n_items_total=num_items,
            k=N_RECS,
            model=model,
            idx_to_item=idx_to_item,
            user_to_idx=user_to_idx,
            item_to_idx=item_to_idx
        )

        # 2. Registra logs no MLflow
        dataset_mlp = mlflow.data.from_pandas(
            df_train_pandas, name="dataset_RetailRocket_Events")
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

        # 3. Registra as métricas calculadas reais
        mlflow.log_metrics({
            "eval.precision_at_10": resultados_experimentos[f"Precision@{N_RECS}"],
            "eval.recall_at_10": resultados_experimentos[f"Recall@{N_RECS}"],
            "eval.ndcg_at_10": resultados_experimentos[f"NDCG@{N_RECS}"],
            "eval.catalog_coverage": resultados_experimentos["Coverage"]
        })

        # 4. Registra o Modelo
        sample_user = np.array([12345], dtype=np.int64)
        sample_output = np.array(list(idx_to_item.values())[
                                 :N_RECS], dtype=np.int64)
        signature = infer_signature(
            model_input={"visitor_id": sample_user}, model_output=sample_output)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="modelo-neumf-retailrocket",
            signature=signature,
            registered_model_name="neumf-mlp-recommendation"
        )

    print("Processo finalizado com sucesso!")
    spark.stop()


if __name__ == "__main__":
    main()
