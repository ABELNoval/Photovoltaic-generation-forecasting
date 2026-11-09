"""Lectura de los metadatos de los 175 sistemas de Utrecht."""

from pathlib import Path

import pandas as pd

from src.data.paths import METADATA_CSV

STUDY_START = pd.Timestamp("2014-01-01", tz="UTC")
STUDY_END = pd.Timestamp("2017-12-31", tz="UTC")


def load_metadata(path: Path = METADATA_CSV) -> pd.DataFrame:
    """Devuelve los metadatos indexados por ID, con la ubicación aproximada.

    El archivo usa `;` como separador y trae columnas vacías al final. La
    ubicación viene como un recuadro (north, south, west, east); se usa su
    centro como coordenada del sistema. Capacidades en W.
    """
    meta = pd.read_csv(path, sep=";").dropna(axis=1, how="all").set_index("ID")
    meta["begin_ts"] = pd.to_datetime(meta["begin_ts"], utc=True)
    meta["end_ts"] = pd.to_datetime(meta["end_ts"], utc=True)
    meta["latitude"] = (meta["north"] + meta["south"]) / 2
    meta["longitude"] = (meta["west"] + meta["east"]) / 2
    return meta


def full_period_systems(meta: pd.DataFrame) -> pd.DataFrame:
    """Sistemas con mediciones durante todo el periodo de estudio (2014–2017)."""
    return meta[(meta["begin_ts"] <= STUDY_START) & (meta["end_ts"] >= STUDY_END)]
