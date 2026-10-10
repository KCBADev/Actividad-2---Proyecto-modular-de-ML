"""Carga de datos y separación de predictores (X) y objetivo (y).

Nota: el módulo se llama `data_loader` y no `io` para no confundirlo con el
módulo `io` de la biblioteca estándar de Python.
"""

from pathlib import Path

import pandas as pd

from src.config import FEATURES, RAW_DATA_PATH, RAW_DATA_URL, SPLIT_PATHS, TARGET


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Lee el conjunto original completo.

    Si el archivo local no existe, lo descarga de la URL de respaldo y lo guarda
    en `data/raw/` para no depender de internet en ejecuciones posteriores.
    """
    path = Path(path)
    if not path.exists():
        print(f"No se encontró {path}. Descargando desde {RAW_DATA_URL} ...")
        df = pd.read_csv(RAW_DATA_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)


def load_split(name: str) -> pd.DataFrame:
    """Lee una de las particiones generadas por `python -m src.split`.

    Args:
        name: "train", "valid" o "test".
    """
    if name not in SPLIT_PATHS:
        raise ValueError(f"Partición desconocida: {name!r}. Usa una de {list(SPLIT_PATHS)}.")

    path = SPLIT_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Primero genera las particiones con:\n"
            "    python -m src.split"
        )
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa predictores y objetivo, verificando que existan las columnas esperadas."""
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en los datos: {missing}")
    return df[FEATURES].copy(), df[TARGET].copy()


def load_xy(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Atajo: carga una partición y la regresa ya separada en X, y."""
    return split_xy(load_split(name))
