"""Referencia de regresión lineal (docs/planning.md, sección 4, Miembro 2).

Usa solo las columnas obligatorias y sin escalado: como referencia, sus
entradas son fijas, y la regresión lineal ordinaria no depende de la escala.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.base import ModelInterface
from src.schema import REQUIRED_FEATURES


class LinearRegressionBaseline(ModelInterface):
    name = "linear_regression"

    def __init__(self):
        self.model_: LinearRegression | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionBaseline":
        self.model_ = LinearRegression().fit(X[REQUIRED_FEATURES], y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Hay que llamar a train antes de predict.")
        values = self.model_.predict(X[REQUIRED_FEATURES])
        return pd.Series(values, index=X.index, name="prediction")
