"""Modelo de ejemplo: plantilla para implementar la interfaz común.

Predice, para cada hora del día, la potencia media de esa hora en el
entrenamiento. No es una de las referencias del proyecto (persistencia,
persistencia reescalada y regresión lineal); solo sirve para copiar la
estructura y para probar el flujo completo sin depender de otro modelo.
"""

import pandas as pd

from src.models.base import ModelInterface


class HourlyMeanModel(ModelInterface):
    name = "example"

    def __init__(self):
        self.hourly_mean_: pd.Series | None = None
        self.global_mean_: float = 0.0

    def train(self, X: pd.DataFrame, y: pd.Series) -> "HourlyMeanModel":
        hours = pd.DatetimeIndex(X.index).hour
        self.hourly_mean_ = y.groupby(hours).mean()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.hourly_mean_ is None:
            raise RuntimeError("Hay que llamar a train antes de predict.")
        hours = pd.DatetimeIndex(X.index).hour
        values = self.hourly_mean_.reindex(hours).fillna(self.global_mean_)
        return pd.Series(values.to_numpy(), index=X.index, name="prediction")
