from pydantic import BaseModel
from typing import List

class RecommendRequest(BaseModel):
    visitorid: int
    k: int = 10

class RecommendResponse(BaseModel):
    visitorid: int
    recommendations: List[int]