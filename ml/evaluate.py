"""
Modul evaluasi mendalam untuk model SVR gagal jantung.
Menghasilkan plot Prediksi vs Aktual, Distribusi Residual, dan analisis per kategori risiko.
"""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml.preprocess import ALL_MODEL_FEATURES
from ml.train import RANDOM_STATE, map_risk_category


def evaluate_saved_model(
    csv_path: str = "data/heart_failure_clinical_records_dataset.csv",
    model_path: str = "artifacts/heart_risk_svr_pipeline.joblib",
    plots_dir: str = "artifacts/plots",
) -> dict:
    """Mengevaluasi artefak model pada test set (60 sampel) dan membuat grafik diagnostik."""
    output_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    X = df[ALL_MODEL_FEATURES].copy()
    y_binary = df["DEATH_EVENT"].astype(int)
    y_scaled = (y_binary * 100.0).astype(float)

    _, X_test, _, y_test = train_test_split(
        X, y_scaled, test_size=0.20, stratify=y_binary, random_state=RANDOM_STATE
    )

    pipeline = joblib.load(model_path)
    raw_preds = pipeline.predict(X_test)
    clipped_preds = np.clip(raw_preds, 0.0, 100.0)
    residuals = y_test - clipped_preds

    mae = float(mean_absolute_error(y_test, clipped_preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, clipped_preds)))
    r2 = float(r2_score(y_test, clipped_preds))

    # Plot 1: Prediksi vs Aktual (Boxplot + Stripplot)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    eval_df = pd.DataFrame(
        {
            "Aktual": np.where(y_test == 100.0, "Meninggal (100)", "Bertahan Hidup (0)"),
            "Aktual_Num": y_test,
            "Skor_Prediksi": clipped_preds,
            "Residual": residuals,
        }
    )

    sns.boxplot(
        data=eval_df,
        x="Aktual",
        y="Skor_Prediksi",
        ax=axes[0],
        palette=["#10b981", "#ef4444"],
        width=0.45,
    )
    sns.stripplot(
        data=eval_df,
        x="Aktual",
        y="Skor_Prediksi",
        ax=axes[0],
        color="#1e293b",
        alpha=0.6,
        jitter=0.15,
    )
    axes[0].axhline(33.5, color="#f59e0b", linestyle="--", label="Batas Sedang (33.5)")
    axes[0].axhline(66.5, color="#dc2626", linestyle="--", label="Batas Tinggi (66.5)")
    axes[0].set_title(f"Prediksi Skor SVR vs Status Aktual\n(MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f})")
    axes[0].set_ylabel("Skor Risiko Prediksi (0 - 100)")
    axes[0].legend(loc="upper left")

    # Plot 2: Distribusi Residual (y_aktual - y_prediksi)
    sns.histplot(residuals, bins=15, kde=True, ax=axes[1], color="#3b82f6")
    axes[1].axvline(0.0, color="#dc2626", linestyle="--", linewidth=1.5, label="Error = 0")
    axes[1].set_title("Distribusi Residual (Aktual - Prediksi) pada Data Uji")
    axes[1].set_xlabel("Residual (Skala -100 s/d +100)")
    axes[1].set_ylabel("Frekuensi Pasien")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "svr_evaluation_diagnostics.png", dpi=150)
    plt.close(fig)

    # Ringkasan per kategori risiko pada data uji
    eval_df["Kategori"] = [map_risk_category(s) for s in clipped_preds]
    category_breakdown = (
        eval_df.groupby("Kategori", observed=False)["Aktual_Num"]
        .agg(jumlah_pasien="count", rata_rata_aktual="mean")
        .to_dict(orient="index")
    )

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "residual_mean": round(float(np.mean(residuals)), 2),
        "residual_std": round(float(np.std(residuals)), 2),
        "category_breakdown": category_breakdown,
    }


if __name__ == "__main__":
    print(evaluate_saved_model())