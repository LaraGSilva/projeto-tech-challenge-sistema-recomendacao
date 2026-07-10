# scripts/promote_model.py
import mlflow
from mlflow.tracking import MlflowClient


def promote_best_model(
    experiment_name: str = "retailrocket-recommender",
    model_name: str = "BestRecommender",
) -> None:
    """Busca o modelo com melhor desempenho e o promove para o estágio de Produção.

    Identifica a execução (run) com o maior valor de 'ndcg_at_k' dentro do experimento
    especificado, registra esse modelo no Model Registry e realiza a transição
    para o estágio 'Production'.

    Args:
        experiment_name (str): Nome do experimento no MLflow.
        model_name (str): Nome do modelo no Model Registry.

    Raises:
        MlflowException: Se não encontrar runs no experimento ou falhar na transição.
    """
    client: MlflowClient = MlflowClient(tracking_uri="http://localhost:5000")

    # Busca a run com melhor métrica
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(experiment_name).experiment_id],
        order_by=["metrics.ndcg_at_k DESC"],
        max_results=1,
    )

    if not runs:
        print(f"Nenhuma execução encontrada para o experimento: {experiment_name}")
        return

    best_run_id: str = runs[0].info.run_id
    print(f"Melhor run encontrada: {best_run_id}")

    # Registrar modelo
    result = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name=model_name,
    )

    # Transição para Production
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
    )

    print(
        f"Modelo {model_name} v{result.version} promovido para Production com sucesso."
    )


if __name__ == "__main__":
    promote_best_model()
