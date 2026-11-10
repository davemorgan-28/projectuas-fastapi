CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    -- 11 Fitur Klinis Input (tanpa 'time')
    age DOUBLE PRECISION NOT NULL CHECK (age >= 18 AND age <= 120),
    anaemia SMALLINT NOT NULL CHECK (anaemia IN (0, 1)),
    creatinine_phosphokinase DOUBLE PRECISION NOT NULL CHECK (creatinine_phosphokinase > 0),
    diabetes SMALLINT NOT NULL CHECK (diabetes IN (0, 1)),
    ejection_fraction DOUBLE PRECISION NOT NULL CHECK (ejection_fraction >= 5 AND ejection_fraction <= 90),
    high_blood_pressure SMALLINT NOT NULL CHECK (high_blood_pressure IN (0, 1)),
    platelets DOUBLE PRECISION NOT NULL CHECK (platelets > 0),
    serum_creatinine DOUBLE PRECISION NOT NULL CHECK (serum_creatinine > 0),
    serum_sodium DOUBLE PRECISION NOT NULL CHECK (serum_sodium >= 100 AND serum_sodium <= 170),
    sex SMALLINT NOT NULL CHECK (sex IN (0, 1)),
    smoking SMALLINT NOT NULL CHECK (smoking IN (0, 1)),
    -- Output Prediksi Model SVR
    risk_score DOUBLE PRECISION NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 100.0),
    risk_category VARCHAR(20) NOT NULL CHECK (risk_category IN ('Rendah', 'Sedang', 'Tinggi')),
    -- Waktu Pencatatan
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indeks untuk mempercepat pengurutan riwayat terbaru (pagination) dan filter kategori
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_category ON predictions (risk_category);