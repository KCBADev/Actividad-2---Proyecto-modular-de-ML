"""Catálogo de modelos a comparar.

Agregar un modelo nuevo = agregar una entrada a `get_models()`. Ni la carga ni
la evaluación cambian: todos comparten las mismas funciones.
"""

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.config import SEED

# Nombre legible para tablas y reportes.
MODEL_LABELS = {
    "referencia": "Referencia: mediana",
    "lineal": "Regresión lineal",
    "arbol": "Árbol de decisión",
    "bosque": "Bosque aleatorio",
}

BASELINE_KEY = "referencia"


def get_baseline() -> BaseEstimator:
    """Predice siempre la mediana de entrenamiento.

    No es un candidato a elegir: sirve para saber si un modelo aprende algo
    más que "dar el valor típico".
    """
    return DummyRegressor(strategy="median")


def get_models(seed: int = SEED) -> dict[str, BaseEstimator]:
    """Modelos candidatos. Los hiperparámetros son los mismos del notebook de clase."""
    return {
        "lineal": LinearRegression(),
        "arbol": DecisionTreeRegressor(max_depth=6, random_state=seed),
        "bosque": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
    }
