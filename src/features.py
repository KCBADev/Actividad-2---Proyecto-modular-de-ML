"""Variables derivadas.

Estas razones se calculan fila por fila: no estiman ningún parámetro a partir
de otras observaciones. Por eso es seguro aplicarlas a cualquier partición sin
fuga de información, y se integran al pipeline con un FunctionTransformer para
que entrenamiento, validación y prueba reciban exactamente la misma operación.
"""

import pandas as pd

from src.config import DERIVED_FEATURES


def ratio_feature_names(transformer, input_features) -> list[str]:
    """Nombres de salida para el FunctionTransformer: originales + derivadas.

    Permite usar `pipeline[:-1].get_feature_names_out()` para saber qué
    columna corresponde a cada coeficiente o importancia del modelo.
    """
    return list(input_features) + DERIVED_FEATURES


def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
    """Agrega habitaciones y recámaras por hogar sin modificar las columnas originales.

    `households` se reemplaza 0 -> 1 en el denominador como regla explícita para
    evitar divisiones entre cero (en este dataset no hay ceros, pero la regla
    protege ante datos nuevos).

    Si `total_bedrooms` es nulo, `bedrooms_per_household` también lo será; el
    imputador del preprocesador se encarga de esos valores.
    """
    out = X.copy()
    households = out["households"].replace(0, 1)
    out["rooms_per_household"] = out["total_rooms"] / households
    out["bedrooms_per_household"] = out["total_bedrooms"] / households
    return out
