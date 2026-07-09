from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from api.service_wrapper import RecommendationService
from api.schema import RecommendRequest, RecommendResponse

# Variável global para o serviço de recomendação
service: RecommendationService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gerencia o ciclo de vida da aplicação FastAPI.

    Inicializa o serviço de recomendação na inicialização e o limpa
    no encerramento.

    Args:
        app: Instância da aplicação FastAPI.

    Yields:
        None: Controle do ciclo de vida.
    """
    global service
    service = RecommendationService(model_name="Neural-NeuMF-MLP", alias="production")
    yield
    service = None

app: FastAPI = FastAPI(lifespan=lifespan)

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Endpoint para obter recomendações personalizadas de itens.

    Recebe um ID de visitante e o número desejado de itens, retornando
    uma lista de recomendações baseada no modelo em produção.

    Args:
        request (RecommendRequest): Objeto contendo 'visitorid' e 'k'.

    Returns:
        RecommendResponse: Objeto com 'visitorid' e lista de 'recommendations'.

    Raises:
        HTTPException: Se o serviço de recomendação não estiver inicializado (503).
    """
    if not service:
        raise HTTPException(
            status_code=503, 
            detail="Serviço de recomendação indisponível no momento."
        )
    
    recommendations: list[str] = service.get_recommendations(
        request.visitorid, 
        request.k
    )
    
    return RecommendResponse(
        visitorid=request.visitorid, 
        recommendations=recommendations
    )