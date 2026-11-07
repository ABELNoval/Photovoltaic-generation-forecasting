"""Funciones comunes a todos los modelos: datos de entrada, ajuste de las
predicciones y guardado de resultados."""

from pathlib import Path

import joblib
import pandas as pd

from src.models.base import ModelInterface
from src.schema import DAY_FLAG, INPUT_FEATURES, TARGET, TIMESTAMP

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def split_xy(df: pd.DataFrame, daytime_only: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el dataset en `X` (columnas de entrada) e `y` (`power_norm`).

    Por defecto deja solo las horas de luz, que son las que se usan para
    entrenar y para calcular las métricas.
    """
    if TIMESTAMP in df.columns:
        df = df.set_index(TIMESTAMP)
    if daytime_only:
        df = df[df[DAY_FLAG] == 1]
    return df[INPUT_FEATURES], df[TARGET]


def postprocess_predictions(predictions: pd.Series, is_day: pd.Series) -> pd.Series:
    """Pone a cero las horas de noche y recorta el resto a [0, 1].

    La potencia normalizada no puede ser negativa ni superar la capacidad;
    los modelos de regresión no lo saben y a veces se salen del rango.
    """
    is_day = is_day.reindex(predictions.index).fillna(0).astype(bool)
    return predictions.clip(0.0, 1.0).where(is_day, 0.0)


def save_predictions(predictions: pd.Series, name: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Guarda `results/<name>_predictions.csv` con `timestamp_utc,prediction`."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}_predictions.csv"
    out = predictions.rename("prediction").rename_axis(TIMESTAMP).reset_index()
    out.to_csv(path, index=False)
    return path


def save_model(model: ModelInterface, path: Path) -> None:
    """Guarda un modelo entrenado para reutilizarlo (p. ej. en forecast.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> ModelInterface:
    """Carga un modelo guardado con `save_model`."""
    return joblib.load(path)
