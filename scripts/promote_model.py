# scripts/promote_model.py
import mlflow

client = mlflow.MlflowClient("http://localhost:5000")

# pegar o run com melhor NDCG
runs = client.search_runs(
    experiment_names=["retailrocket-recommender"],
    order_by=["metrics.ndcg_at_k DESC"],
    max_results=1,
)
best_run_id = runs[0].info.run_id

# registrar no Model Registry
result = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name="BestRecommender",
)

# promover para Production
client.transition_model_version_stage(
    name="BestRecommender",
    version=result.version,
    stage="Production",
)
print(f"Modelo v{result.version} promovido para Production")