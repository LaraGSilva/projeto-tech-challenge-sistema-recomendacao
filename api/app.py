from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.service_wrapper import RecommendationService

service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    # O caminho dos modelos aqui deve ser o caminho dentro do container (/app/models/...)
    service = RecommendationService(
        model_type="neumf_light",
        model_path="/app/models/neumf_model.pth",
        mappings_path="/app/models/mappings.npy"
    )
    yield
    service = None

app = FastAPI(lifespan=lifespan)

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    if not service:
        raise HTTPException(status_code=503, detail="Serviço indisponível")
    return {"visitorid": request.visitorid, "recommendations": service.get_recommendations(request.visitorid, request.k)}