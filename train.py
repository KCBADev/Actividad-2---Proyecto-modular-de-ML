"""Punto de entrada: coordina el flujo completo de ML.

Uso (desde la raíz del proyecto, después de `python -m src.split`):
    python train.py                       # compara lineal, árbol y bosque
    python train.py --models lineal arbol # compara solo esos dos

Flujo:
    1. Cargar entrenamiento y validación (prueba sigue sin abrirse).
    2. Construir un pipeline (features + preprocesamiento + modelo) por candidato.
    3. Entrenar cada uno con train y compararlos en valid, junto a la referencia.
    4. Elegir el de menor MAE en validación.
    5. Solo entonces cargar prueba y evaluar UNA vez el modelo elegido.
"""

import argparse
import json

import pandas as pd

from src.config import RESULTS_DIR, SEED, SELECTION_METRIC
from src.data_loader import load_xy
from src.evaluation import ModelEvaluator, evaluate_on_test
from src.models import BASELINE_KEY, MODEL_LABELS, get_baseline, get_models
from src.preprocessing import build_pipeline


def parse_args() -> argparse.Namespace:
    available = list(get_models())
    parser = argparse.ArgumentParser(description="Entrena y compara modelos de regresión.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=available,
        default=available,
        help=f"Modelos a comparar (default: todos). Opciones: {', '.join(available)}",
    )
    return parser.parse_args()


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    args = parse_args()
    pd.set_option("display.float_format", "{:,.2f}".format)

    # 1. Datos de desarrollo -------------------------------------------------
    X_train, y_train = load_xy("train")
    X_valid, y_valid = load_xy("valid")
    print(f"Entrenamiento: {len(X_train):,} filas | Validación: {len(X_valid):,} filas")

    # 2. Un pipeline nuevo por modelo, más la referencia ---------------------
    candidates = get_models(seed=SEED)
    pipelines = {BASELINE_KEY: build_pipeline(get_baseline())}
    pipelines.update({key: build_pipeline(candidates[key]) for key in args.models})

    # 3. Comparación en validación -------------------------------------------
    print_section("Comparación en VALIDACIÓN (entrenados solo con entrenamiento)")
    evaluator = ModelEvaluator(X_train, y_train, X_valid, y_valid)
    table = evaluator.compare(pipelines)
    print()
    print(table.drop(columns="clave").to_string(index=False))

    # 4. Selección -----------------------------------------------------------
    best_key, best_pipeline = evaluator.select_best()
    best_row = table.set_index("clave").loc[best_key]
    gain = evaluator.improvement_over_baseline(best_key)

    print_section("Selección del modelo")
    print(f"Criterio: menor {SELECTION_METRIC} en validación (la referencia no compite).")
    print(f"Modelo elegido: {MODEL_LABELS[best_key]}")
    print(f"  MAE train = {best_row['MAE train']:,.2f} | MAE valid = {best_row['MAE valid']:,.2f}")
    if gain is not None:
        print(f"  Reduce el MAE de validación {gain:.1%} frente a predecir la mediana.")

    # 5. Evaluación final en prueba (una sola vez) ---------------------------
    X_test, y_test = load_xy("test")
    test_metrics = evaluate_on_test(best_pipeline, X_test, y_test)

    print_section(f"Evaluación final en PRUEBA: {MODEL_LABELS[best_key]}")
    print(f"Prueba: {len(X_test):,} filas")
    print(
        f"  MAE  valid = {best_row['MAE valid']:>12,.2f}  |  MAE  test = {test_metrics['MAE']:>12,.2f}\n"
        f"  RMSE valid = {best_row['RMSE valid']:>12,.2f}  |  RMSE test = {test_metrics['RMSE']:>12,.2f}\n"
        f"  R2   valid = {best_row['R2 valid']:>12.4f}  |  R2   test = {test_metrics['R2']:>12.4f}"
    )
    print("\nEstos resultados se reportan; no se usan para volver a elegir modelo.")

    # 6. Guardar resultados ---------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "comparacion_validacion.csv", index=False)
    with open(RESULTS_DIR / "evaluacion_prueba.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo_elegido": MODEL_LABELS[best_key],
                "criterio": f"menor {SELECTION_METRIC} en validación",
                "semilla": SEED,
                "metricas_prueba": test_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResultados guardados en {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
