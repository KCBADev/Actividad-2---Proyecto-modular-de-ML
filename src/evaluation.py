"""Entrenamiento, comparación en validación y evaluación final en prueba.

`ModelEvaluator` es una clase porque guarda estado que comparten todos los
modelos: las mismas particiones de entrenamiento/validación y los pipelines ya
ajustados (para no reentrenar el modelo elegido antes de evaluarlo en prueba).
Las métricas, en cambio, son una función simple.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from src.config import SELECTION_METRIC
from src.models import BASELINE_KEY, MODEL_LABELS


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE y RMSE están en dólares; R2 es adimensional."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


class ModelEvaluator:
    """Entrena con train, mide en train y valid, y selecciona con valid.

    Prueba NO es un atributo de esta clase a propósito: no hay forma de
    consultarla accidentalmente mientras se comparan modelos.
    """

    def __init__(self, X_train, y_train, X_valid, y_valid):
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid
        self.fitted: dict[str, Pipeline] = {}
        self.results: list[dict] = []

    def evaluate(self, key: str, pipeline: Pipeline) -> dict:
        """Ajusta el pipeline SOLO con entrenamiento y lo evalúa en ambas particiones."""
        pipeline.fit(self.X_train, self.y_train)
        self.fitted[key] = pipeline

        train_m = regression_metrics(self.y_train, pipeline.predict(self.X_train))
        valid_m = regression_metrics(self.y_valid, pipeline.predict(self.X_valid))

        row = {
            "clave": key,
            "Modelo": MODEL_LABELS.get(key, key),
            "MAE train": train_m["MAE"],
            "MAE valid": valid_m["MAE"],
            "RMSE valid": valid_m["RMSE"],
            "R2 valid": valid_m["R2"],
        }
        self.results.append(row)
        return row

    def compare(self, pipelines: dict[str, Pipeline]) -> pd.DataFrame:
        """Evalúa todos los pipelines y regresa la tabla ordenada por la métrica de selección."""
        for key, pipeline in pipelines.items():
            print(f"  Entrenando {MODEL_LABELS.get(key, key)} ...")
            self.evaluate(key, pipeline)
        return self.results_table()

    def results_table(self) -> pd.DataFrame:
        table = pd.DataFrame(self.results)
        return table.sort_values(f"{SELECTION_METRIC} valid").reset_index(drop=True)

    def select_best(self, metric: str = SELECTION_METRIC) -> tuple[str, Pipeline]:
        """Modelo con menor `metric` en validación, excluyendo la referencia."""
        candidates = [r for r in self.results if r["clave"] != BASELINE_KEY]
        if not candidates:
            raise ValueError("No hay modelos candidatos para seleccionar.")
        best = min(candidates, key=lambda r: r[f"{metric} valid"])
        return best["clave"], self.fitted[best["clave"]]

    def improvement_over_baseline(self, key: str, metric: str = SELECTION_METRIC) -> float | None:
        """Reducción relativa del error frente a predecir siempre la mediana."""
        by_key = {r["clave"]: r for r in self.results}
        if BASELINE_KEY not in by_key:
            return None
        base = by_key[BASELINE_KEY][f"{metric} valid"]
        return 1 - by_key[key][f"{metric} valid"] / base


def evaluate_on_test(pipeline: Pipeline, X_test, y_test) -> dict[str, float]:
    """Evaluación final. Se llama UNA vez, con el modelo ya elegido en validación.

    El pipeline ya está ajustado con entrenamiento; aquí solo se predice.
    """
    y_pred = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    # Valores como float nativo para poder guardarlos en JSON.
    return {k: float(np.round(v, 4)) for k, v in metrics.items()}
