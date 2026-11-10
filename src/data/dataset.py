"""Lectura del dataset del proyecto (el CSV que genera build_dataset.py)."""

from pathlib import Path

import pandas as pd

from src.schema import TIMESTAMP


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Lee el dataset con `timestamp_utc` como índice en UTC, ordenado."""
    df = pd.read_csv(path)
    if TIMESTAMP not in df.columns:
        raise ValueError(
            f"{path} no tiene la columna '{TIMESTAMP}'; genera el dataset con "
            "python -m src.data.build_dataset."
        )
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], utc=True)
    return df.set_index(TIMESTAMP).sort_index()
