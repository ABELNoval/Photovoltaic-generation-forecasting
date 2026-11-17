"""Modelo kNN implementado desde cero (docs/planning.md, sección 4, Miembro 6).

Para cada hora a pronosticar busca, entre las horas de entrenamiento, las `k`
cuyas condiciones (radiación, nubosidad, temperatura y calendario) son más
parecidas, y predice el promedio de su potencia. No usa scikit-learn: solo
numpy. scikit-learn queda como referencia de comparación para el informe.

Las variables se estandarizan con la media y la desviación del entrenamiento
(`train`), nunca con los datos que se predicen.
"""

import numpy as np
import pandas as pd

from src.models.base import ModelInterface
from src.schema import REQUIRED_FEATURES

# Filas de X que se comparan con todo el entrenamiento a la vez. Acota la
# memoria: con el dataset completo (unas 9000 horas de entrenamiento) el pico
# es de unos 100 MB; con bloques de 1024 filas supera 1,5 GB.
CHUNK_SIZE = 64


class KNNModel(ModelInterface):
    name = "knn"

    def __init__(self, k: int = 5, features: list[str] | None = None):
        if k < 1:
            raise ValueError(f"k debe ser al menos 1, no {k}.")
        self.k = k
        self.features = list(REQUIRED_FEATURES if features is None else features)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> "KNNModel":
        if len(X) < self.k:
            raise ValueError(f"Hay {len(X)} filas de entrenamiento y k={self.k}; k no puede superarlas.")
        values = X[self.features].to_numpy(dtype=float)
        self.mean_ = values.mean(axis=0)
        std = values.std(axis=0)
        # Una variable constante en el entrenamiento no distingue horas: se deja
        # en 0 en lugar de dividir entre 0.
        self.std_ = np.where(std > 0, std, 1.0)
        self.X_train_ = (values - self.mean_) / self.std_
        self.y_train_ = y.to_numpy(dtype=float)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.X_train_ is None or self.y_train_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Hay que llamar a train antes de predict.")
        values = (X[self.features].to_numpy(dtype=float) - self.mean_) / self.std_

        predictions = np.empty(len(values))
        for start in range(0, len(values), CHUNK_SIZE):
            chunk = values[start : start + CHUNK_SIZE]
            # Distancia euclidiana al cuadrado: ordena igual que la distancia
            # y evita la raíz.
            diff = chunk[:, None, :] - self.X_train_[None, :, :]
            squared = np.einsum("ijk,ijk->ij", diff, diff)
            nearest = np.argpartition(squared, self.k - 1, axis=1)[:, : self.k]
            predictions[start : start + CHUNK_SIZE] = self.y_train_[nearest].mean(axis=1)
        return pd.Series(predictions, index=X.index, name="prediction")
