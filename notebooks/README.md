# Notebook de referencia

Este proyecto reorganiza el notebook de marimo de la Actividad 1:

- Original del curso: [`01_ml_recap/actividad_01_housing.py`](https://github.com/IvTole/Intro_IA_ML_CUGDL/blob/main/01_ml_recap/actividad_01_housing.py)
- Estructura de referencia: [`02_ml_modules`](https://github.com/IvTole/Intro_IA_ML_CUGDL/tree/main/02_ml_modules)

Coloquen aquí la versión resuelta del equipo como `actividad_01_housing.py`.
Para abrirla: `pip install marimo` y `marimo edit notebooks/actividad_01_housing.py`.

## Correspondencia notebook → módulos

| Sección del notebook | Módulo |
|---|---|
| 1. Cargar y revisar los datos | `src/data_loader.py` |
| 2. Separar antes de aprender (TODO 1) | `src/split.py` |
| 3. Crear variables (TODO 2) | `src/features.py` |
| `construir_pipeline` | `src/preprocessing.py` |
| 4. Diccionario de modelos (TODO 3) | `src/models.py` |
| Tabla con referencia (mediana) | `src/evaluation.py` |
| 5–6. Comparar, elegir y evaluar en prueba (TODO 4) | `train.py` |
