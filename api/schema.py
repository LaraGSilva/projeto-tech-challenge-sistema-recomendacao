from typing import List

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """Schema de requisição para o endpoint de recomendações.

    Atributos:
        visitorid (int): O identificador único do usuário/visitante.
        k (int): Quantidade de itens a serem retornados. Deve estar entre 1 e 50.
    """

    visitorid: int = Field(
        ..., description="O ID do visitante para o qual queremos recomendações"
    )
    k: int = Field(
        default=5, ge=1, le=50, description="Número de itens a serem recomendados"
    )


class RecommendResponse(BaseModel):
    """Schema de resposta para o endpoint de recomendações.

    Atributos:
        visitorid (int): O identificador do usuário que solicitou as recomendações.
        recommendations (List[int]): Lista contendo os IDs dos itens recomendados.
    """

    visitorid: int = Field(..., description="O ID do visitante solicitado")
    recommendations: List[int] = Field(
        ..., description="Lista de IDs dos itens recomendados"
    )


class AllRecommendationsResponse(BaseModel):
    """Schema de resposta para o endpoint de recomendações.

    Atributos para todos os clientes:
        visitorid (int): O identificador do usuário que solicitou as recomendações.
        recommendations (List[int]): Lista contendo os IDs dos itens recomendados.
    """

    recommendations: List[dict]
