from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from api.service_wrapper import RecommendationService
from api.schema import RecommendRequest, RecommendResponse 

service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = RecommendationService(model_name="Neural-NeuMF-MLP",alias="production")
    yield
    service = None

app = FastAPI(lifespan=lifespan)

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    if not service:
        raise HTTPException(status_code=503, detail="Serviço indisponível")
    
    recommendations = service.get_recommendations(request.visitorid, request.k)
    
    return {
        "visitorid": request.visitorid, 
        "recommendations": recommendations
    }
