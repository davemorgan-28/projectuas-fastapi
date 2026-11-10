"""
Modul konstruksi Preprocessing Pipeline menggunakan ColumnTransformer dan scikit-learn Pipeline.
Menjamin transformasi fitur saat training dan inferensi API identik 100%.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Urutan kanonik 11 fitur input model (tanpa 'time')
NUMERICAL_FEATURES: list[str] = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
]

BINARY_FEATURES: list[str] = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
]

ALL_MODEL_FEATURES: list[str] = NUMERICAL_FEATURES + BINARY_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Membangun ColumnTransformer:
    - Fitur numerik: distandardisasi ke mean=0, std=1 dengan StandardScaler.
    - Fitur biner (0/1): diteruskan apa adanya (passthrough) agar interpretasi diskrit terjaga.
    - Kolom lain (seperti 'time' jika tidak sengaja terbawa): otomatis dibuang (remainder='drop').
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_svr_pipeline(
    C: float = 50.0,
    gamma: str | float = 0.05,
    epsilon: float = 5.0,
) -> Pipeline:
    """
    Membungkus ColumnTransformer dan SVR (kernel RBF) ke dalam satu scikit-learn Pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)),
        ]
    )
    return pipeline
