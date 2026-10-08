"""Preprocesamiento y construcción del pipeline completo.

Todo lo que "aprende" de los datos (mediana para imputar, media y desviación
para escalar, categorías del one-hot) queda dentro de un Pipeline. Al llamar
`pipeline.fit(X_train, y_train)` esos parámetros se calculan SOLO con
entrenamiento; validación y prueba únicamente se transforman con `predict`.
"""

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import CAT_FEATURES, DERIVED_FEATURES, GEO_FEATURES, NUM_FEATURES
from src.features import add_ratio_features, ratio_feature_names

# Columnas numéricas DESPUÉS de crear las variables derivadas.
NUMERIC_COLUMNS = GEO_FEATURES + NUM_FEATURES + DERIVED_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Imputa y escala numéricas; codifica la categórica.

    - SimpleImputer(median): en vez de borrar las ~1% de filas con
      `total_bedrooms` nulo, se imputan con la mediana de ENTRENAMIENTO.
    - StandardScaler: se aplica también a latitud/longitud para que todas las
      numéricas queden en la misma escala (importa en modelos lineales o por
      distancia; a los árboles les es indiferente).
    - OneHotEncoder(handle_unknown="ignore"): la categoría ISLAND solo tiene 5
      filas en todo el dataset; si no cae en entrenamiento, el encoder no debe
      fallar al verla en validación o prueba.
    """
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_COLUMNS),
            ("cat", categorical, CAT_FEATURES),
        ],
        remainder="drop",  # Si llega una columna no declarada en config, no se usa en silencio.
    )


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Variables derivadas -> preprocesamiento -> modelo.

    Se construye un pipeline NUEVO en cada llamada, para que ningún modelo
    reutilice un preprocesador ya ajustado por otro.
    """
    return Pipeline(
        [
            (
                "features",
                FunctionTransformer(add_ratio_features, feature_names_out=ratio_feature_names),
            ),
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
