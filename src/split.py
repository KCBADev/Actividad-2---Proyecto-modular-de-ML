"""División del conjunto original en entrenamiento, validación y prueba.

Uso (desde la raíz del proyecto):
    python -m src.split

Genera data/processed/{train,valid,test}.csv con proporciones 60/20/20.
La división se hace UNA sola vez y se guarda en disco, así todos los modelos
y todos los integrantes trabajan exactamente con las mismas filas.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, SEED, SPLIT_PATHS, TEST_SIZE, VALID_SIZE
from src.data_loader import load_raw


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    valid_size: float = VALID_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide en dos pasos: primero se aparta prueba, luego se separa validación.

    `valid_size` es una fracción del TOTAL. Como el segundo split opera sobre
    lo que quedó después de quitar prueba (80%), hay que reescalarla:
    0.20 / 0.80 = 0.25 del conjunto de desarrollo.
    """
    df_dev, df_test = train_test_split(df, test_size=test_size, random_state=seed)

    valid_fraction_of_dev = valid_size / (1 - test_size)
    df_train, df_valid = train_test_split(
        df_dev, test_size=valid_fraction_of_dev, random_state=seed
    )

    _check_split(df, df_train, df_valid, df_test)
    return df_train, df_valid, df_test


def _check_split(df, df_train, df_valid, df_test) -> None:
    """Comprobaciones equivalentes a las del notebook de clase."""
    train_idx, valid_idx, test_idx = set(df_train.index), set(df_valid.index), set(df_test.index)
    assert train_idx.isdisjoint(valid_idx), "Hay filas repetidas entre train y valid."
    assert train_idx.isdisjoint(test_idx), "Hay filas repetidas entre train y test."
    assert valid_idx.isdisjoint(test_idx), "Hay filas repetidas entre valid y test."
    assert len(df_train) + len(df_valid) + len(df_test) == len(df), "Se perdieron filas."


def main() -> None:
    df = load_raw()
    df_train, df_valid, df_test = split_dataset(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in {"train": df_train, "valid": df_valid, "test": df_test}.items():
        part.to_csv(SPLIT_PATHS[name], index=False)

    summary = pd.DataFrame(
        {
            "Conjunto": ["Entrenamiento", "Validación", "Prueba"],
            "Filas": [len(df_train), len(df_valid), len(df_test)],
            "Fracción": [len(p) / len(df) for p in (df_train, df_valid, df_test)],
            "Faltantes en total_bedrooms": [
                int(p["total_bedrooms"].isna().sum()) for p in (df_train, df_valid, df_test)
            ],
        }
    )
    print(f"Filas totales: {len(df):,}")
    print(summary.to_string(index=False, float_format="{:.2%}".format))
    print(f"\nParticiones guardadas en {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
