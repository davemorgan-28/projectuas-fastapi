from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Skema konfigurasi environment variable aplikasi."""

    APP_NAME: str = "Kalkulator Edukasi Skor Risiko Gagal Jantung"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Kredensial koneksi PostgreSQL
    DATABASE_URL: str

    # Path artefak model dan metrik
    DATASET_PATH: str = "data/heart_failure_clinical_records_dataset.csv"
    MODEL_PATH: str = "artifacts/heart_risk_svr_pipeline.joblib"
    METRICS_PATH: str = "artifacts/model_metrics.json"
    PLOTS_DIR: str = "artifacts/plots"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Mengembalikan instance Settings yang di-cache."""
    return Settings()