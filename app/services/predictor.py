import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from app.schemas.prediction import ClinicalInputSchema
from ml.preprocess import ALL_MODEL_FEATURES
from ml.train import map_risk_category


class HeartRiskPredictor:
    """Pembungkus (wrapper) artefak Pipeline SVR untuk inferensi cepat dan aman."""

    def __init__(self, model_path: str, metrics_path: str) -> None:
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.pipeline = None
        self.metrics: dict = {}

    def load(self) -> None:
        """Memuat pipeline joblib dan file JSON metrik ke memori saat aplikasi menyala."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Artefak model tidak ditemukan di '{self.model_path}'. "
                "Jalankan 'python -m ml.train' terlebih dahulu."
            )
        self.pipeline = joblib.load(self.model_path)

        if self.metrics_path.exists():
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    def predict(self, payload: ClinicalInputSchema) -> tuple[float, str]:
        """
        Menghitung skor risiko (0-100) dan kategori risiko ('Rendah', 'Sedang', 'Tinggi').
        Menggunakan DataFrame dengan urutan kolom kanonik agar ColumnTransformer bekerja presisi.
        """
        if self.pipeline is None:
            self.load()

        input_dict = payload.model_dump()
        df_row = pd.DataFrame([[input_dict[col] for col in ALL_MODEL_FEATURES]], columns=ALL_MODEL_FEATURES)

        raw_score = float(self.pipeline.predict(df_row)[0])
        clipped_score = round(float(np.clip(raw_score, 0.0, 100.0)), 2)
        category = map_risk_category(clipped_score)
        return clipped_score, category