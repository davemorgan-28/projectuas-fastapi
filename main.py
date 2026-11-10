from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.services.predictor import HeartRiskPredictor

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Memuat artefak model SVR satu kali saat aplikasi dinyalakan."""
    predictor = HeartRiskPredictor(
        model_path=settings.MODEL_PATH,
        metrics_path=settings.METRICS_PATH,
    )
    predictor.load()
    app.state.predictor = predictor
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Aplikasi web edukatif untuk memperkirakan skor risiko gagal jantung (0-100) "
        "menggunakan algoritma Support Vector Regression (SVR) kernel RBF."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)