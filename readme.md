# CardioScore SVR — Aplikasi Web Edukasi Prediksi Skor Risiko Gagal Jantung

Aplikasi web *end-to-end* berbasis **FastAPI**, **Scikit-Learn (Support Vector Regression - RBF Kernel)**, **SQLAlchemy 2.x**, **Alembic**, dan **PostgreSQL** untuk menghitung estimasi skor risiko gagal jantung (`0–100`) dari 11 parameter klinis serta menyimpan riwayat prediksinya.

> **Disclaimer Edukasi Medis**: Aplikasi ini dibuat khusus untuk tujuan pembelajaran rekayasa perangkat lunak dan *Data Science*. Skor yang dihasilkan bukan merupakan diagnosis medis maupun pengganti konsultasi dokter.

## Mengapa Fitur `time` Dikecualikan dari Input Model?

Pada dataset asli [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/ml/datasets/heart+failure+clinical+records) ([Chicco & Jurman, 2020](https://pubmed.ncbi.nlm.nih.gov/32013925/)), kolom `time` mencatat jumlah hari pemantauan (*follow-up period*, 4 hingga 285 hari) sejak pasien masuk rumah sakit hingga penelitian berakhir atau hingga pasien meninggal dunia (`DEATH_EVENT = 1`).

Fitur `time` **wajib dibuang** dari model prediksi karena dua alasan kritis:
1. **Tidak Tersedia Saat Inferensi Nyata**: Ketika pengguna atau tenaga medis mengisi formulir di awal pemeriksaan, lama waktu pemantauan di masa depan belum terjadi dan mustahil diketahui.
2. **Menyebabkan *Target Data Leakage* Berat**: Pasien yang meninggal dunia otomatis memiliki nilai `time` yang pendek karena pemantauan berhenti saat kematian terjadi (korelasi Pearson $r = -0,5270$ terhadap `DEATH_EVENT`). Memasukkan `time` akan membuat model "curang" dengan menebak kematian dari pendeknya durasi observasi, bukan dari kondisi fisiologis jantung dan ginjal pasien.

## Langkah Instalasi & Persiapan Lingkungan

1. **Buat dan aktifkan *virtual environment* Python (3.11+)**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

````

2. **Salin file konfigurasi `.env`**:

   ```
   cp .env.example .env
   
   ```

   Sesuaikan nilai `DATABASE_URL` di dalam `.env` dengan kredensial PostgreSQL lokal Anda.

3. **Unduh Dataset**: Simpan file `heart_failure_clinical_records_dataset.csv` (299 baris) dari [Kaggle Heart Failure Clinical Data](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) ke dalam path `data/heart_failure_clinical_records_dataset.csv`.

## Setup PostgreSQL & pgAdmin4 serta Migrasi Database

1. Buka **pgAdmin4**, sambungkan ke server PostgreSQL lokal (`localhost:5432`).

2. Buat role `hf_user` (aktifkan *Can login?*) dan buat database `heart_failure_db` dengan *Owner* `hf_user`.

3. Jalankan migrasi skema tabel `predictions` dengan Alembic:

   ```
   alembic upgrade head
   
   ```

## Melatih Model SVR & Menjalankan Evaluasi

1. **Jalankan EDA singkat**:

   ```
   python -m ml.eda
   
   ```

2. **Latih model SVR dengan GridSearchCV dan simpan artefak `joblib`**:

   ```
   python -m ml.train
   
   ```

   Perintah ini akan menghasilkan `artifacts/heart_risk_svr_pipeline.joblib` dan `artifacts/model_metrics.json`.

3. **Jalankan evaluasi diagnostik pada data uji**:

   ```
   python -m ml.evaluate
   
   ```

## Menjalankan Server Uvicorn

```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

```

* Antarmuka Web Kalkulator: `http://localhost:8000/`

* Halaman Riwayat Prediksi: `http://localhost:8000/riwayat`

* Halaman Tentang Model: `http://localhost:8000/tentang-model`

* Dokumentasi Interaktif OpenAPI: `http://localhost:8000/docs`

## Contoh Request `curl` ke Endpoint REST API

### 1. Memeriksa Kesehatan Sistem (`GET /health`)

```
curl -X GET "http://localhost:8000/health"

```

### 2. Mengirim Prediksi Baru (`POST /api/predict`)

```
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 68.0,
    "anaemia": 1,
    "creatinine_phosphokinase": 582.0,
    "diabetes": 0,
    "ejection_fraction": 22.0,
    "high_blood_pressure": 1,
    "platelets": 245000.0,
    "serum_creatinine": 2.4,
    "serum_sodium": 131.0,
    "sex": 1,
    "smoking": 0
  }'

```

### 3. Mengambil Riwayat Prediksi dengan Paginasi (`GET /api/predictions`)

```
curl -X GET "http://localhost:8000/api/predictions?page=1&page_size=5"

```

````

## Tahap 12 (Opsional Lengkap): Pengujian Otomatis (`pytest`) & Kontainerisasi (`Dockerfile` + `docker-compose.yml`)

### Alasan Keputusan Teknis
Untuk memastikan keandalan di lingkungan pengembangan maupun *Continuous Integration (CI)*, pengujian `pytest` menggunakan database SQLite *in-memory* (`sqlite:///:memory:`) melalui fitur `app.dependency_overrides[get_db]` agar pengujian endpoint `/api/predict` dan `/api/predictions` berjalan instan dan mandiri tanpa merusak data PostgreSQL produksi.

### 1. Unit Test Preprocessing & Pemetaan Kategori (`tests/test_preprocess.py`)

```python
"""
File: tests/test_preprocess.py
Unit test untuk memastikan pipeline preprocessing konsisten, bebas kolom 'time',
dan memetakan kategori risiko secara akurat.
"""
import numpy as np
import pandas as pd

from ml.preprocess import ALL_MODEL_FEATURES, NUMERICAL_FEATURES, build_preprocessor
from ml.train import map_risk_category


def test_time_column_is_excluded_from_features() -> None:
    """Memastikan fitur 'time' tidak pernah masuk ke dalam daftar fitur input model."""
    assert "time" not in ALL_MODEL_FEATURES
    assert len(ALL_MODEL_FEATURES) == 11


def test_preprocessor_scales_numeric_and_preserves_binary() -> None:
    """Memastikan StandardScaler menstandardisasi fitur numerik dan membiarkan fitur biner 0/1."""
    sample_data = pd.DataFrame(
        [
            {
                "age": 50.0,
                "creatinine_phosphokinase": 100.0,
                "ejection_fraction": 30.0,
                "platelets": 200000.0,
                "serum_creatinine": 1.0,
                "serum_sodium": 135.0,
                "anaemia": 0,
                "diabetes": 1,
                "high_blood_pressure": 0,
                "sex": 1,
                "smoking": 0,
                "time": 120,  # Harus otomatis terbuang oleh remainder='drop'
            },
            {
                "age": 70.0,
                "creatinine_phosphokinase": 300.0,
                "ejection_fraction": 50.0,
                "platelets": 300000.0,
                "serum_creatinine": 2.0,
                "serum_sodium": 145.0,
                "anaemia": 1,
                "diabetes": 0,
                "high_blood_pressure": 1,
                "sex": 0,
                "smoking": 1,
                "time": 240,
            },
        ]
    )

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(sample_data)

    # Output harus berdimensi (2 baris, 11 kolom) tanpa kolom 'time'
    assert transformed.shape == (2, 11)
    # 6 kolom pertama (numerik) harus memiliki mean ~ 0.0
    assert np.allclose(transformed[:, : len(NUMERICAL_FEATURES)].mean(axis=0), 0.0)
    # 5 kolom terakhir (biner) harus tetap bernilai eksak 0 atau 1
    assert set(np.unique(transformed[:, len(NUMERICAL_FEATURES) :])).issubset({0.0, 1.0})


def test_risk_category_boundaries() -> None:
    """Memastikan batas kategori Rendah (0-33), Sedang (34-66), dan Tinggi (67-100)."""
    assert map_risk_category(0.0) == "Rendah"
    assert map_risk_category(33.0) == "Rendah"
    assert map_risk_category(34.0) == "Sedang"
    assert map_risk_category(66.0) == "Sedang"
    assert map_risk_category(67.0) == "Tinggi"
    assert map_risk_category(100.0) == "Tinggi"

````

### 2. Integration Test Endpoint FastAPI (`tests/test_api.py`)

```
"""
File: tests/test_api.py
Integration test untuk endpoint /health, POST /api/predict, dan GET /api/predictions.
"""
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.main import app


class DummyPredictor:
    """Stub predictor deterministik untuk pengujian cepat endpoint API."""

    pipeline = True
    metrics = {"test_metrics": {"mae": 27.20, "rmse": 37.81, "r2": 0.3393}}

    def predict(self, payload):
        # Menghasilkan skor tinggi jika fraksi ejeksi rendah & kreatinin tinggi
        score = 74.5 if payload.ejection_fraction < 30 and payload.serum_creatinine > 1.8 else 22.0
        category = "Tinggi" if score >= 66.5 else "Rendah"
        return score, category


# Setup SQLite in-memory untuk isolasi pengujian database
SQLALCHEMY_TEST_URL = "sqlite://"
test_engine = create_engine(
    SQLALCHEMY_TEST_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
app.state.predictor = DummyPredictor()
client = TestClient(app)


def test_health_endpoint() -> None:
    """Memastikan endpoint /health mengembalikan status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True


def test_create_and_list_predictions() -> None:
    """Menguji alur penuh POST /api/predict dan GET /api/predictions dengan paginasi."""
    valid_payload = {
        "age": 70.0,
        "anaemia": 1,
        "creatinine_phosphokinase": 582.0,
        "diabetes": 0,
        "ejection_fraction": 20.0,
        "high_blood_pressure": 1,
        "platelets": 265000.0,
        "serum_creatinine": 2.1,
        "serum_sodium": 130.0,
        "sex": 1,
        "smoking": 0,
    }
    post_resp = client.post("/api/predict", json=valid_payload)
    assert post_resp.status_code == 201
    result = post_resp.json()
    assert 0.0 <= result["risk_score"] <= 100.0
    assert result["risk_category"] == "Tinggi"
    assert "edukasi" in result["disclaimer"].lower()

    # Pastikan tersimpan dan muncul di endpoint paginasi
    list_resp = client.get("/api/predictions?page=1&page_size=10")
    assert list_resp.status_code == 200
    history = list_resp.json()
    assert history["total"] >= 1
    assert history["items"][0]["id"] == result["id"]


def test_invalid_clinical_range_rejected() -> None:
    """Memastikan input di luar batas klinis (misal ejection_fraction=150%) ditolak (HTTP 422)."""
    invalid_payload = {
        "age": 60.0,
        "anaemia": 0,
        "creatinine_phosphokinase": 250.0,
        "diabetes": 0,
        "ejection_fraction": 150.0,  # Melebihi batas maksimum 85%
        "high_blood_pressure": 0,
        "platelets": 250000.0,
        "serum_creatinine": 1.1,
        "serum_sodium": 137.0,
        "sex": 1,
        "smoking": 0,
    }
    resp = client.post("/api/predict", json=invalid_payload)
    assert resp.status_code == 422

```

### 3. Kontainerisasi dengan `Dockerfile` dan `docker-compose.yml`

#### File: `Dockerfile`

```
FROM python:3.11-slim

# Mencegah pembuatan file .pyc dan mengaktifkan log langsung ke stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instal dependensi sistem untuk kompilasi/koneksi libpq (PostgreSQL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Latih model saat build jika dataset tersedia di dalam folder data/
RUN if [ -f "data/heart_failure_clinical_records_dataset.csv" ]; then python -m ml.train; fi

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

```

#### File: `docker-compose.yml`

```
services:
  postgres:
    image: postgres:16-alpine
    container_name: hf_postgres
    restart: unless-stopped
    env_file:
      - .env
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 5

  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: hf_pgadmin4
    restart: unless-stopped
    environment:
      PGADMIN_DEFAULT_EMAIL: ${PGADMIN_EMAIL:-admin@local.dev}
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD:-admin123}
    ports:
      - "5050:80"
    depends_on:
      postgres:
        condition: service_healthy

  web:
    build: .
    container_name: hf_fastapi_web
    restart: unless-stopped
    env_file:
      - .env
    environment:
      DATABASE_URL: postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  pgdata: