"""Convenio de tiempo del dataset.

Cada fila se etiqueta con el final de su hora, así que el día D va de
D 01:00 a D+1 00:00 (igual que en src/data/build_dataset.py).
"""

import pandas as pd

ONE_HOUR = pd.Timedelta(hours=1)
ONE_DAY = pd.Timedelta(days=1)


def day_of(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Día al que pertenece cada marca de tiempo."""
    return (pd.DatetimeIndex(index) - ONE_HOUR).normalize()


def select_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filas de los días `start` a `end`, ambos incluidos.

    `df` debe estar indexado por `timestamp_utc` y ordenado (load_dataset).
    """
    first = pd.Timestamp(start, tz="UTC") + ONE_HOUR
    last = pd.Timestamp(end, tz="UTC") + ONE_DAY
    return df.loc[first:last]
