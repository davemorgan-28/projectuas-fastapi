from datetime import datetime
from sqlalchemy import DateTime, Float, Index, Integer, SmallInteger, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class PredictionRecord(Base):
    """Representasi tabel 'predictions' untuk menyimpan input klinis dan skor risiko."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)

    # 11 Fitur Klinis Input
    age: Mapped[float] = mapped_column(Float, nullable=False, comment="Usia pasien (tahun)")
    anaemia: Mapped[int] = mapped_column(SmallInteger, nullable=False, comment="Status anemia (0=Tidak, 1=Ya)")
    creatinine_phosphokinase: Mapped[float] = mapped_column(Float, nullable=False, comment="Enzim CPK (mcg/L)")
    diabetes: Mapped[int] = mapped_column(SmallInteger, nullable=False, comment="Status diabetes (0=Tidak, 1=Ya)")
    ejection_fraction: Mapped[float] = mapped_column(Float, nullable=False, comment="Fraksi ejeksi ventrikel kiri (%)")
    high_blood_pressure: Mapped[int] = mapped_column(SmallInteger, nullable=False, comment="Hipertensi (0=Tidak, 1=Ya)")
    platelets: Mapped[float] = mapped_column(Float, nullable=False, comment="Trombosit (kiloplatelets/mL)")
    serum_creatinine: Mapped[float] = mapped_column(Float, nullable=False, comment="Kreatinin serum (mg/dL)")
    serum_sodium: Mapped[float] = mapped_column(Float, nullable=False, comment="Natrium serum (mEq/L)")
    sex: Mapped[int] = mapped_column(SmallInteger, nullable=False, comment="Jenis kelamin (0=Perempuan, 1=Laki-laki)")
    smoking: Mapped[int] = mapped_column(SmallInteger, nullable=False, comment="Perokok aktif (0=Tidak, 1=Ya)")

    # Output Prediksi SVR
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, comment="Skor risiko kontinu (0.0 - 100.0)")
    risk_category: Mapped[str] = mapped_column(String(20), nullable=False, comment="Kategori: Rendah / Sedang / Tinggi")

    # Waktu pembuatan record (UTC timezone-aware)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        Index("idx_predictions_created_at_desc", created_at.desc()),
    )