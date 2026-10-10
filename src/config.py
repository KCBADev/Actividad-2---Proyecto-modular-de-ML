"""Configuración central del proyecto.

Todo lo que podría cambiar entre experimentos (rutas, columnas, semilla y
proporciones de la división) vive aquí, para que el resto de los módulos
no tenga valores "mágicos" repetidos.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
# Se calculan a partir de este archivo, no del directorio desde el que se
# ejecuta Python. Así `python train.py` funciona igual desde cualquier lugar.
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "housing.csv"

PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_PATHS = {
    "train": PROCESSED_DIR / "train.csv",
    "valid": PROCESSED_DIR / "valid.csv",
    "test": PROCESSED_DIR / "test.csv",
}

RESULTS_DIR = ROOT_DIR / "results"

# Respaldo por si data/raw/housing.csv no existe (mismo archivo usado en clase).
RAW_DATA_URL = (
    "https://raw.githubusercontent.com/IvTole/Intro_IA_ML_CUGDL/"
    "refs/heads/main/data/housing/housing.csv"
)

# ---------------------------------------------------------------------------
# Reproducibilidad y división de datos
# ---------------------------------------------------------------------------
SEED = 42

# Proporciones sobre el TOTAL de filas: 60% entrenamiento, 20% validación, 20% prueba.
TEST_SIZE = 0.20
VALID_SIZE = 0.20

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
GEO_FEATURES = ["longitude", "latitude"]

NUM_FEATURES = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

CAT_FEATURES = ["ocean_proximity"]

# Variables que se crean dentro del pipeline (ver src/features.py).
DERIVED_FEATURES = ["rooms_per_household", "bedrooms_per_household"]

# Columnas que se leen del CSV como predictores.
FEATURES = GEO_FEATURES + NUM_FEATURES + CAT_FEATURES

TARGET = "median_house_value"

# ---------------------------------------------------------------------------
# Selección de modelo
# ---------------------------------------------------------------------------
# Se elige el modelo con menor MAE en VALIDACIÓN. Prueba no participa en la elección.
SELECTION_METRIC = "MAE"
