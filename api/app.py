# from contextlib import asynccontextmanager
# from typing import AsyncGenerator

# from fastapi import Depends, FastAPI, HTTPException, Request, status

# from api.schema import RecommendRequest, RecommendResponse
# from api.service_wrapper import RecommendationService


# @asynccontextmanager
# async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
#     """Gerencia o ciclo de vida da aplicação FastAPI.

#     Inicializa o serviço de recomendação e o anexa ao estado da aplicação.
#     """
#     # Inicialização
#     app.state.service = RecommendationService(
#         model_name="Neural-NeuMF-MLP", alias="production"
#     )
#     yield
#     # Limpeza (opcional)
#     app.state.service = None


# app: FastAPI = FastAPI(lifespan=lifespan)


# def get_recommendation_service(request: Request) -> RecommendationService:
#     """Extrai o serviço de recomendação do estado da aplicação.

#     Args:
#         request (Request): Requisição HTTP atual.

#     Returns:
#         RecommendationService: Instância do serviço de recomendação.

#     Raises:
#         HTTPException: Se o serviço estiver indisponível (503).
#     """
#     service = getattr(request.app.state, "service", None)
#     if service is None:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Serviço de recomendação indisponível.",
#         )
#     return service


# @app.post("/recommend", response_model=RecommendResponse)
# def recommend(
#     request: RecommendRequest,
#     service: RecommendationService = Depends(get_recommendation_service),
# ) -> RecommendResponse:
#     """Endpoint para obter recomendações personalizadas.

#     Args:
#         request (RecommendRequest): Dados da requisição (visitorid e k).
#         service (RecommendationService): Serviço injetado via dependência.

#     Returns:
#         RecommendResponse: Objeto com visitorid e recomendações.
#     """
#     recommendations: list[str] = service.get_recommendations(
#         request.visitorid, request.k
#     )

#     return RecommendResponse(
#         visitorid=request.visitorid, recommendations=recommendations
#     )


from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, status

from api.schema import AllRecommendationsResponse, RecommendResponse
from api.service_wrapper import RecommendationService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.service = RecommendationService(
        model_name="Neural-NeuMF-MLP", alias="production"
    )
    yield
    app.state.service = None


app: FastAPI = FastAPI(lifespan=lifespan)


def get_recommendation_service(request: Request) -> RecommendationService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço de recomendação indisponível.",
        )
    return service


# Endpoint 1: Health
@app.get("/recommend/health")
def health():
    return {"status": "ok", "message": "modelo rodando com sucesso"}


# Endpoint 2: Retornar todas as recomendações (Lista completa)
@app.get("/recommend/list/all", response_model=AllRecommendationsResponse)
def recommend_all(
    service: RecommendationService = Depends(get_recommendation_service),
) -> AllRecommendationsResponse:
    """Retorna recomendações para todos os usuários conhecidos."""

    # 1. Recupera a lista de IDs de todos os usuários do seu serviço
    # Ajuste o acesso abaixo conforme a estrutura interna do seu service_wrapper
    all_visitor_ids = service.get_all_visitor_ids()

    all_recommendations = []

    # 2. Itera sobre a lista e chama a inferência para cada um
    for visitor_id in all_visitor_ids:
        recs = service.get_recommendations(visitor_id, k=10)
        all_recommendations.append({"visitorid": visitor_id, "recommendations": recs})

    return AllRecommendationsResponse(recommendations=all_recommendations)


# Endpoint 3: Recomendação por cliente (via Path Parameter)
@app.get("/recommend/{visitorid}", response_model=RecommendResponse)
def recommend_by_visitor(
    visitorid: int,
    k: int = 10,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendResponse:
    recommendations: list[str] = service.get_recommendations(visitorid, k)
    return RecommendResponse(visitorid=visitorid, recommendations=recommendations)
