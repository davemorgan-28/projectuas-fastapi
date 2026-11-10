"""
Modul Exploratory Data Analysis (EDA) untuk Dataset Heart Failure Clinical Records.
Menghasilkan ringkasan statistik di terminal dan menyimpan plot distribusi/korelasi.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Daftar fitur sesuai spesifikasi (tanpa kolom 'time' untuk input model)
NUMERICAL_FEATURES = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
]

BINARY_FEATURES = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
]

TARGET_COLUMN = "DEATH_EVENT"


def run_eda(
    csv_path: str = "data/heart_failure_clinical_records_dataset.csv",
    plots_dir: str = "artifacts/plots",
) -> dict:
    """Menjalankan analisis eksploratif lengkap dan menyimpan grafik diagnostik."""
    output_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"=== 1. DIMENSI DATASET: {df.shape[0]} baris x {df.shape[1]} kolom ===")

    # 1. Pemeriksaan Missing Values
    missing = df.isnull().sum()
    print("\n=== 2. JUMLAH MISSING VALUES PER KOLOM ===")
    print(missing.to_string())

    # 2. Keseimbangan Kelas Target (DEATH_EVENT)
    class_counts = df[TARGET_COLUMN].value_counts()
    class_props = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2)
    print("\n=== 3. KESEIMBANGAN KELAS TARGET (DEATH_EVENT) ===")
    for label, count in class_counts.items():
        status = "Bertahan Hidup (0)" if label == 0 else "Meninggal (1)"
        print(f"- {status}: {count} pasien ({class_props[label]}%)")

    # 3. Deteksi Outlier (Metode 1.5 x IQR) & Distribusi Fitur Numerik
    print("\n=== 4. STATISTIK DESKRIPTIF & DETEKSI OUTLIER (NUMERIK) ===")
    outlier_summary = []
    for col in NUMERICAL_FEATURES:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        outlier_summary.append(
            {
                "fitur": col,
                "mean": round(df[col].mean(), 2),
                "std": round(df[col].std(), 2),
                "min": round(df[col].min(), 2),
                "median": round(df[col].median(), 2),
                "max": round(df[col].max(), 2),
                "skewness": round(df[col].skew(), 2),
                "batas_bawah_iqr": round(lower, 2),
                "batas_atas_iqr": round(upper, 2),
                "jumlah_outlier": int(len(outliers)),
                "persen_outlier": round(len(outliers) / len(df) * 100, 2),
            }
        )
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.to_string(index=False))

    # 4. Korelasi Pearson dengan Target DEATH_EVENT
    print("\n=== 5. KORELASI FITUR DENGAN DEATH_EVENT ===")
    correlations = df.corr()[TARGET_COLUMN].sort_values(ascending=False).round(4)
    print(correlations.to_string())

    # 5. Visualisasi Boxplot Outlier & Heatmap Korelasi
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flatten(), NUMERICAL_FEATURES):
        sns.boxplot(data=df, x= TARGET_COLUMN, y=col, ax=ax, palette="Set2")
        ax.set_title(f"Distribusi {col} vs DEATH_EVENT", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "eda_boxplots.png", dpi=150)
    plt.close(fig)

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df[NUMERICAL_FEATURES + BINARY_FEATURES + [TARGET_COLUMN]].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax_corr,
    )
    ax_corr.set_title("Matriks Korelasi Fitur Klinis (Tanpa Fitur 'time')")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_correlation_heatmap.png", dpi=150)
    plt.close(fig_corr)

    return {
        "n_rows": int(df.shape[0]),
        "missing_total": int(missing.sum()),
        "class_balance": class_counts.to_dict(),
        "outliers": outlier_summary,
        "correlations": correlations.to_dict(),
    }


if __name__ == "__main__":
    run_eda()