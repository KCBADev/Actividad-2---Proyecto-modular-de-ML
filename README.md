# Proyecto modular de ML · Valor de vivienda en California

Reorganización del notebook de marimo de la Actividad 1 (Introducción a la IA y ML, CUGDL) en un proyecto de Python modular, con un flujo de ML que separa entrenamiento, validación y prueba, compara modelos y evalúa en prueba una sola vez.

## Problema y datos

**Problema de regresión:** predecir `median_house_value`, el valor mediano (en dólares) de las viviendas de una **zona censal** de California. Cada fila describe una zona, no una casa: una predicción es el valor típico de la zona, no el precio de una vivienda específica.

**Datos:** *California Housing* (censo de 1990), el mismo archivo usado en clase. 20,640 zonas y 10 columnas:

| Tipo | Columnas |
|---|---|
| Geográficas | `longitude`, `latitude` |
| Numéricas | `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income` |
| Categórica | `ocean_proximity` (`<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`) |
| Objetivo | `median_house_value` |

`total_bedrooms` tiene 207 faltantes (1.0%). Además se crean dos variables por fila: `rooms_per_household` y `bedrooms_per_household`.

## Integrantes

| Nombre | GitHub |
|---|---|
| Kevin Cristopher Bautista Aviña | [@KCBADev](https://github.com/KCBADev) |
| Diego López | [@diegolopezzz](https://github.com/diegolopezzz) |
| Gael Torres | [@Gael201122](https://github.com/Gael201122) |

## Organización de los archivos

```
.
├── train.py               # Punto de entrada: coordina todo el flujo
├── requirements.txt       # Dependencias
├── data/
│   ├── raw/housing.csv    # Datos originales (incluidos en el repo)
│   └── processed/         # train/valid/test.csv (se generan, no se suben)
├── notebooks/             # Notebook de referencia y correspondencia con los módulos
├── results/               # Métricas de cada corrida (se generan, no se suben)
├── src/
│   ├── config.py          # Rutas, variables predictoras, objetivo, semilla, proporciones
│   ├── data_loader.py     # Carga de datos y separación X / y
│   ├── split.py           # División 60/20/20 en entrenamiento, validación y prueba
│   ├── features.py        # Variables derivadas por hogar
│   ├── preprocessing.py   # Imputación, escalado, one-hot y armado del Pipeline
│   ├── models.py          # Modelos candidatos y referencia
│   └── evaluation.py      # Métricas, comparación en validación y evaluación en prueba
└── .github/pull_request_template.md
```

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/KCBADev/Actividad-2---Proyecto-modular-de-ML.git
cd Actividad-2---Proyecto-modular-de-ML

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Con `uv`: `uv venv && uv pip install -r requirements.txt`.

**Datos:** ya vienen en `data/raw/housing.csv`. Si el archivo no existe, `src/split.py` lo descarga automáticamente del [repositorio del curso](https://github.com/IvTole/Intro_IA_ML_CUGDL/tree/main/data/housing) y lo guarda en esa ruta.

## Ejecución

Todos los comandos se corren desde la raíz del repositorio.

```bash
# 1. Preparar los datos: genera data/processed/{train,valid,test}.csv
python -m src.split

# 2. Entrenar, comparar en validación, elegir y evaluar en prueba
python train.py

# Opcional: comparar solo algunos modelos (lineal, arbol, bosque)
python train.py --models lineal arbol
```

La división se hace una vez y se guarda en disco: todos los modelos (y todos los integrantes) usan exactamente las mismas filas. `train.py` escribe `results/comparacion_validacion.csv` y `results/evaluacion_prueba.json`.

## Flujo de ML

1. **División 60/20/20** con semilla 42 (`src/split.py`). Primero se aparta prueba (20%) y del 80% restante se toma el 25% para validación.
2. **Pipeline por modelo** (`src/preprocessing.py`): variables derivadas → imputación por mediana + `StandardScaler` en numéricas, `OneHotEncoder` en `ocean_proximity` → modelo. Al llamar `fit` con entrenamiento, la mediana, las medias, las desviaciones y las categorías se calculan **solo con entrenamiento**; validación y prueba solo se transforman.
3. **Comparación en validación** junto a una referencia que predice siempre la mediana de entrenamiento.
4. **Selección:** menor MAE en validación. La referencia no compite.
5. **Prueba:** se carga hasta después de elegir y se evalúa una sola vez. Su resultado se reporta, no se usa para volver a elegir.

Decisiones respecto al notebook y a `02_ml_modules`:

- **Imputar en vez de borrar filas.** El notebook eliminaba las 207 filas con faltantes. Aquí se imputan con la mediana de entrenamiento dentro del pipeline, así no se cambia la población analizada y no hay fuga de información.
- **`handle_unknown="ignore"` en el one-hot.** `ISLAND` tiene solo 5 filas en todo el dataset (con nuestra semilla: 1 en entrenamiento, 4 en validación, 0 en prueba). Sin esa opción, si no cayera ninguna en entrenamiento, predecir en validación lanzaría un error.
- **Rutas absolutas desde `config.py`.** El proyecto corre igual sin importar desde qué carpeta se ejecute.
- **Prueba nunca vive en `ModelEvaluator`.** La clase solo conoce entrenamiento y validación, para que no se pueda consultar prueba por accidente al comparar.

## Modelos y resultados

| Modelo | Hiperparámetros | MAE entrenamiento | MAE validación | RMSE validación | R² validación |
|---|---|---:|---:|---:|---:|
| Referencia: mediana | `DummyRegressor(strategy="median")` | 87,514 | 88,775 | 118,889 | −0.06 |
| Regresión lineal | por defecto | 49,656 | 48,711 | 67,255 | 0.66 |
| Árbol de decisión | `max_depth=6` | 45,599 | 46,644 | 66,641 | 0.67 |
| **Bosque aleatorio** | `n_estimators=100` | 12,353 | **32,465** | 49,019 | 0.82 |

**Modelo elegido:** Bosque aleatorio (menor MAE en validación).

| Bosque aleatorio | MAE | RMSE | R² |
|---|---:|---:|---:|
| Validación | 32,465 | 49,019 | 0.82 |
| **Prueba** | **33,926** | **51,324** | **0.81** |

### Interpretación

- **Qué significa el MAE.** En prueba, el valor mediano predicho para una zona se desvía en promedio unos **$33,900** del real. Frente a predecir siempre la mediana ($88,775 de MAE en validación), el bosque reduce el error en un **63%**.
- **Sobreajuste del bosque.** Su MAE de entrenamiento ($12,353) es mucho menor que el de validación ($32,465): memoriza buena parte de entrenamiento. Aun así es el que mejor generaliza, y por eso se elige con validación y no con entrenamiento.
- **Subajuste de lineal y árbol.** Sus errores de entrenamiento y validación son casi iguales (~$46–50k): no sobreajustan, pero son demasiado simples para la relación entre variables y valor.
- **Prueba vs validación.** El error de prueba es ~4.5% mayor que el de validación. No tienen que coincidir porque son muestras distintas; que sean cercanos indica que validación fue una estimación razonable. El modelo no se cambió después de ver prueba.
- **RMSE > MAE** en todos los modelos: hay errores grandes en algunas zonas que el RMSE penaliza más.
- **Qué usa el bosque.** `median_income` concentra casi la mitad de la importancia (0.48), seguida de `ocean_proximity = INLAND` (0.14) y las coordenadas (~0.18 entre ambas). El ingreso y la ubicación explican la mayor parte del valor de una zona. (Se obtiene con `pipeline[:-1].get_feature_names_out()` y `feature_importances_`.)

## Limitaciones y problemas conocidos

- **Valores topados.** 965 zonas (4.7%) tienen `median_house_value = 500,001`: el valor real es mayor pero quedó censurado. El modelo aprende ese tope y su error en zonas caras está subestimado.
- **El modelo final se entrena solo con el 60% de los datos.** Como en el notebook, no se reentrena con entrenamiento + validación antes de prueba.
- **Hiperparámetros fijos y una sola partición.** No hay búsqueda de hiperparámetros ni validación cruzada; la elección depende de una sola división aleatoria.
- **División aleatoria, no geográfica.** Zonas vecinas pueden quedar en entrenamiento y prueba, así que el error medido es optimista para regiones completamente nuevas.
- **`ISLAND` casi sin información:** el modelo la ve en una sola fila de entrenamiento.
- **Imputación simple.** La mediana ignora la relación entre `total_bedrooms` y `total_rooms`; imputar a partir de esa razón podría ser mejor.
- **Datos de 1990:** no representan precios actuales.

## Tabla de contribuciones

| Integrante | Contribución | Archivos principales | PR | Commit | Revisó |
|---|---|---|---|---|---|
| Kevin Cristopher Bautista Aviña | Base del proyecto: configuración, carga y división de datos | `src/config.py`, `src/data_loader.py`, `src/split.py`, `requirements.txt`, `.gitignore`, `data/`, `notebooks/` | #PENDIENTE | `xxxxxxx` | PR #1 |
| Diego López | Variables derivadas, preprocesamiento y modelos | `src/features.py`, `src/preprocessing.py`, `src/models.py` | #1 | `779f8c2` | PR de Gael |
| Gael Torres | Evaluación, flujo principal y documentación | `src/evaluation.py`, `train.py`, `README.md` | #PENDIENTE | `xxxxxxx` | PR de Kevin |

## Referencias

- Curso: [IvTole/Intro_IA_ML_CUGDL](https://github.com/IvTole/Intro_IA_ML_CUGDL) (`01_ml_recap`, `02_ml_modules`)
- scikit-learn: [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
