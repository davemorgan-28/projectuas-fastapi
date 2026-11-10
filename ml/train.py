"""
Modul pelatihan model Support Vector Regression (SVR) dengan kernel RBF.
Menerapkan stratified train/test split (80/20), GridSearchCV, dan serialisasi joblib.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from ml.preprocess import ALL_MODEL_FEATURES, build_svr_pipeline

RANDOM_STATE = 42


def map_risk_category(score: float) -> str:
    """
    Memetakan skor kontinu (0-100) ke kategori risiko:
    - Rendah (0-33): 0.0 <= skor < 33.5
    - Sedang (34-66): 33.5 <= skor < 66.5
    - Tinggi (67-100): 66.5 <= skor <= 100.0
    """
    if score < 33.5:
        return "Rendah"
    if score < 66.5:
        return "Sedang"
    return "Tinggi"


def train_and_save_model(
    csv_path: str = "data/heart_failure_clinical_records_dataset.csv",
    model_output_path: str = "artifacts/heart_risk_svr_pipeline.joblib",
    metrics_output_path: str = "artifacts/model_metrics.json",
) -> dict:
    """Melatih pipeline SVR dengan GridSearchCV dan menyimpan artefak model serta metrik."""
    df = pd.read_csv(csv_path)

    # Pastikan fitur 'time' TIDAK disertakan dalam matriks fitur X
    X = df[ALL_MODEL_FEATURES].copy()
    y_binary = df["DEATH_EVENT"].astype(int)
    # Transformasi target biner (0/1) ke skala skor risiko (0.0 / 100.0)
    y_scaled = (y_binary * 100.0).astype(float)

    # Split 80/20 dengan stratifikasi berdasarkan label biner asli
    X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = train_test_split(
        X,
        y_scaled,
        y_binary,
        test_size=0.20,
        stratify=y_binary,
        random_state=RANDOM_STATE,
    )

    # Konstruksi pipeline (StandardScaler hanya akan di-fit pada X_train di tiap fold)
    base_pipeline = build_svr_pipeline()

    # Grid hyperparameter disesuaikan dengan skala target 0-100
    param_grid = {
        "regressor__C": [1.0, 10.0, 25.0, 50.0, 100.0],
        "regressor__gamma": ["scale", 0.01, 0.05, 0.1, 0.2],
        "regressor__epsilon": [1.0, 5.0, 10.0, 15.0],
    }

    # Stratified 5-Fold CV agar proporsi kematian (32.1%) identik di setiap fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(skf.split(X_train, y_bin_train))

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_splits,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    # Fit GridSearchCV HANYA pada data train
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_

    # Evaluasi pada data train dan data test (dengan clipping ke rentang 0-100)
    raw_train_pred = best_pipeline.predict(X_train)
    train_pred = np.clip(raw_train_pred, 0.0, 100.0)

    raw_test_pred = best_pipeline.predict(X_test)
    test_pred = np.clip(raw_test_pred, 0.0, 100.0)

    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "excluded_feature": "time",
        "best_params": {
            "C": grid_search.best_params_["regressor__C"],
            "gamma": grid_search.best_params_["regressor__gamma"],
            "epsilon": grid_search.best_params_["regressor__epsilon"],
        },
        "cv_best_rmse": round(float(-grid_search.best_score_), 2),
        "train_metrics": {
            "mae": round(float(mean_absolute_error(y_train, train_pred)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_train, train_pred))), 2),
            "r2": round(float(r2_score(y_train, train_pred)), 4),
        },
        "test_metrics": {
            "mae": round(float(mean_absolute_error(y_test, test_pred)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, test_pred))), 2),
            "r2": round(float(r2_score(y_test, test_pred)), 4),
        },
    }

    # Simpan pipeline terlatih dan metadata metrik
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_output_path)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("=== PELATIHAN MODEL SVR SELESAI ===")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_and_save_model()
