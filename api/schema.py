from pydantic import BaseModel, Field
from typing import List

# 1. Schema para o que a API recebe (Request)
class RecommendRequest(BaseModel):
    visitorid: int = Field(..., description="O ID do visitante para o qual queremos recomendações")
    k: int = Field(default=5, ge=1, le=50, description="Número de itens a serem recomendados")

# 2. Schema para o que a API responde (Response)
class RecommendResponse(BaseModel):
    visitorid: int
    recommendations: List[int] = Field(..., description="Lista de IDs dos itens recomendados")