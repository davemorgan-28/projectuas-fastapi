from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ClinicalInputSchema(BaseModel):
    """
    Validasi rentang fisiologis/klinis untuk 11 fitur input model.
    Fitur 'time' sengaja tidak ada karena menyebabkan data leakage.
    """

    age: float = Field(
        ...,
        ge=18.0,
        le=110.0,
        description="Usia pasien dalam tahun (18 - 110)",
        examples=[65.0],
    )
    anaemia: Literal[0, 1] = Field(
        ...,
        description="Status anemia / penurunan hemoglobin (0 = Tidak, 1 = Ya)",
        examples=[1],
    )
    creatinine_phosphokinase: float = Field(
        ...,
        ge=10.0,
        le=10000.0,
        description="Kadar enzim CPK dalam darah dalam mcg/L (10 - 10.000)",
        examples=[582.0],
    )
    diabetes: Literal[0, 1] = Field(
        ...,
        description="Status riwayat diabetes melitus (0 = Tidak, 1 = Ya)",
        examples=[0],
    )
    ejection_fraction: float = Field(
        ...,
        ge=10.0,
        le=85.0,
        description="Persentase darah yang dipompa keluar ventrikel kiri tiap kontraksi (10% - 85%)",
        examples=[25.0],
    )
    high_blood_pressure: Literal[0, 1] = Field(
        ...,
        description="Status hipertensi / tekanan darah tinggi (0 = Tidak, 1 = Ya)",
        examples=[1],
    )
    platelets: float = Field(
        ...,
        ge=15000.0,
        le=1000000.0,
        description="Jumlah trombosit dalam darah dalam kiloplatelets/mL (15.000 - 1.000.000)",
        examples=[263358.03],
    )
    serum_creatinine: float = Field(
        ...,
        ge=0.2,
        le=15.0,
        description="Kadar kreatinin serum dalam mg/dL (0.2 - 15.0)",
        examples=[1.9],
    )
    serum_sodium: float = Field(
        ...,
        ge=105.0,
        le=165.0,
        description="Kadar natrium serum dalam mEq/L (105 - 165)",
        examples=[130.0],
    )
    sex: Literal[0, 1] = Field(
        ...,
        description="Jenis kelamin biologis (0 = Perempuan, 1 = Laki-laki)",
        examples=[1],
    )
    smoking: Literal[0, 1] = Field(
        ...,
        description="Status merokok (0 = Tidak Merokok, 1 = Perokok)",
        examples=[0],
    )


class PredictionResponseSchema(ClinicalInputSchema):
    """Respons hasil prediksi lengkap yang telah tersimpan di database PostgreSQL."""

    id: int
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Skor risiko kontinu (0-100)")
    risk_category: Literal["Rendah", "Sedang", "Tinggi"]
    disclaimer: str = (
        "Peringatan: Skor ini dihasilkan oleh model pembelajaran mesin untuk tujuan edukasi "
        "dan bukan merupakan diagnosis medis maupun pengganti konsultasi dokter."
    )
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PaginatedPredictionsSchema(BaseModel):
    """Skema respons daftar riwayat prediksi dengan metadata paginasi."""

    total: int
    page: int
    page_size: int
    total_pages: int
    items: list[PredictionResponseSchema]