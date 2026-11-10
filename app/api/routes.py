import math
from fastapi import APIRouter, Depends, Form, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.prediction import PredictionRecord
from app.schemas.prediction import (
    ClinicalInputSchema,
    PaginatedPredictionsSchema,
    PredictionResponseSchema,
)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def save_prediction_to_db(
    db: Session, payload: ClinicalInputSchema, score: float, category: str
) -> PredictionRecord:
    """Menyimpan input klinis beserta hasil prediksi ke tabel PostgreSQL 'predictions'."""
    record = PredictionRecord(
        **payload.model_dump(),
        risk_score=score,
        risk_category=category,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# =====================================================================
# 1. ENDPOINT REST API (JSON)
# =====================================================================

@router.get("/health", tags=["System"])
def health_check(request: Request, db: Session = Depends(get_db)) -> dict:
    """Memeriksa kesehatan koneksi PostgreSQL dan kesiapan model SVR di memori."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    predictor = request.app.state.predictor
    model_loaded = predictor.pipeline is not None
    return {
        "status": "healthy" if (db_ok and model_loaded) else "degraded",
        "database_connected": db_ok,
        "model_loaded": model_loaded,
    }


@router.post(
    "/api/predict",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_201_CREATED,
    tags=["Predictions API"],
)
def api_predict(
    payload: ClinicalInputSchema,
    request: Request,
    db: Session = Depends(get_db),
) -> PredictionRecord:
    """Menerima JSON fitur klinis, memvalidasi rentang, memprediksi skor SVR, dan menyimpan ke DB."""
    predictor = request.app.state.predictor
    score, category = predictor.predict(payload)
    record = save_prediction_to_db(db, payload, score, category)
    return record


@router.get(
    "/api/predictions",
    response_model=PaginatedPredictionsSchema,
    tags=["Predictions API"],
)
def api_list_predictions(
    page: int = Query(1, ge=1, description="Nomor halaman (mulai dari 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Jumlah data per halaman"),
    category: str | None = Query(None, description="Filter opsional: Rendah, Sedang, atau Tinggi"),
    db: Session = Depends(get_db),
) -> dict:
    """Mengambil riwayat prediksi dari PostgreSQL dengan dukungan paginasi dan filter kategori."""
    base_query = select(PredictionRecord)
    count_query = select(func.count(PredictionRecord.id))

    if category in {"Rendah", "Sedang", "Tinggi"}:
        base_query = base_query.where(PredictionRecord.risk_category == category)
        count_query = count_query.where(PredictionRecord.risk_category == category)

    total = db.scalar(count_query) or 0
    total_pages = max(1, math.ceil(total / page_size))
    offset = (page - 1) * page_size

    stmt = (
        base_query.order_by(PredictionRecord.created_at.desc(), PredictionRecord.id.desc())
        .offset(offset)
        .limit(page_size)
    )
    items = list(db.scalars(stmt).all())

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "items": items,
    }


# =====================================================================
# 2. RUTE HALAMAN WEB (HTML + JINJA2)
# =====================================================================

@router.get("/", response_class=HTMLResponse, tags=["Web UI"])
def web_index(request: Request) -> HTMLResponse:
    """Menampilkan halaman formulir kalkulator skor risiko gagal jantung."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None, "form_data": {}},
    )


@router.post("/predict-web", response_class=HTMLResponse, tags=["Web UI"])
def web_predict_submit(
    request: Request,
    age: float = Form(...),
    anaemia: int = Form(...),
    creatinine_phosphokinase: float = Form(...),
    diabetes: int = Form(...),
    ejection_fraction: float = Form(...),
    high_blood_pressure: int = Form(...),
    platelets: float = Form(...),
    serum_creatinine: float = Form(...),
    serum_sodium: float = Form(...),
    sex: int = Form(...),
    smoking: int = Form(...),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Memproses pengiriman form HTML, menyimpan hasil ke DB, dan merender gauge skor risiko."""
    payload = ClinicalInputSchema(
        age=age,
        anaemia=anaemia,
        creatinine_phosphokinase=creatinine_phosphokinase,
        diabetes=diabetes,
        ejection_fraction=ejection_fraction,
        high_blood_pressure=high_blood_pressure,
        platelets=platelets,
        serum_creatinine=serum_creatinine,
        serum_sodium=serum_sodium,
        sex=sex,
        smoking=smoking,
    )
    predictor = request.app.state.predictor
    score, category = predictor.predict(payload)
    record = save_prediction_to_db(db, payload, score, category)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": record, "form_data": payload.model_dump()},
    )


@router.get("/riwayat", response_class=HTMLResponse, tags=["Web UI"])
def web_history(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Menampilkan halaman tabel riwayat prediksi dengan navigasi paginasi."""
    data = api_list_predictions(page=page, page_size=page_size, category=None, db=db)
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"pagination": data},
    )


@router.get("/tentang-model", response_class=HTMLResponse, tags=["Web UI"])
def web_about_model(request: Request) -> HTMLResponse:
    """Menampilkan halaman 'Tentang Model' berisi metrik evaluasi SVR dan penjelasan ilmiah."""
    metrics = request.app.state.predictor.metrics
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={"metrics": metrics},
    )

### 4. Entrypoint Utama FastAPI (`app/main.py`)

"""
File: app/main.py
Entrypoint aplikasi FastAPI dengan lifecycle management untuk pemuatan model SVR.
"""
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