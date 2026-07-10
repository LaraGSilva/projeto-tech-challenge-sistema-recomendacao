from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, status

from api.schema import RecommendRequest, RecommendResponse
from api.service_wrapper import RecommendationService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gerencia o ciclo de vida da aplicação FastAPI.

    Inicializa o serviço de recomendação e o anexa ao estado da aplicação.
    """
    # Inicialização
    app.state.service = RecommendationService(
        model_name="Neural-NeuMF-MLP", alias="production"
    )
    yield
    # Limpeza (opcional)
    app.state.service = None


app: FastAPI = FastAPI(lifespan=lifespan)


def get_recommendation_service(request: Request) -> RecommendationService:
    """Extrai o serviço de recomendação do estado da aplicação.

    Args:
        request (Request): Requisição HTTP atual.

    Returns:
        RecommendationService: Instância do serviço de recomendação.

    Raises:
        HTTPException: Se o serviço estiver indisponível (503).
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço de recomendação indisponível.",
        )
    return service


@app.post("/recommend", response_model=RecommendResponse)
def recommend(
    request: RecommendRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendResponse:
    """Endpoint para obter recomendações personalizadas.

    Args:
        request (RecommendRequest): Dados da requisição (visitorid e k).
        service (RecommendationService): Serviço injetado via dependência.

    Returns:
        RecommendResponse: Objeto com visitorid e recomendações.
    """
    recommendations: list[str] = service.get_recommendations(
        request.visitorid, request.k
    )

    return RecommendResponse(
        visitorid=request.visitorid, recommendations=recommendations
    )
