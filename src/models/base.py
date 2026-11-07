"""Interfaz común de los modelos de pronóstico.

Todos los modelos (MLP, SVR, Random Forest, kNN y las referencias) la
implementan, de modo que la evaluación puede entrenarlos, compararlos y
combinarlos sin conocer su código interno.
"""

from abc import ABC, abstractmethod

import pandas as pd


class ModelInterface(ABC):
    """Contrato que debe cumplir cada modelo.

    - `X`: DataFrame indexado por `timestamp_utc` con todas las columnas de
      entrada (`src.schema.INPUT_FEATURES`). Cada modelo usa siempre las
      obligatorias y elige cuáles de las secundarias aprovecha.
    - `y`: Series `power_norm` con el mismo índice que `X`.

    El escalado de variables, si el modelo lo necesita, se ajusta dentro de
    `train` y solo con los datos que recibe, para no filtrar información de
    validación o prueba.
    """

    #: Identificador corto, usado en el nombre de los archivos de resultados.
    name: str

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> "ModelInterface":
        """Ajusta el modelo con horas de luz del periodo de entrenamiento.

        Devuelve el propio modelo para permitir `model.train(X, y).predict(X)`.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Devuelve la potencia normalizada prevista para cada fila de `X`.

        La Series resultante tiene el mismo índice que `X`. No hace falta
        recortar ni poner a cero la noche: lo hace `postprocess_predictions`.
        """
