"""Modelos que ejecuta src/evaluation/run.py.

Cada entrada recibe el dataset observado y devuelve un modelo sin entrenar.
Las referencias de persistencia lo necesitan; el resto lo ignora. Para añadir
un modelo basta una línea, por ejemplo:

    lambda history: SVRModel(),
"""

from collections.abc import Callable

import pandas as pd

from src.evaluation.baselines.linear_regression import LinearRegressionBaseline
from src.evaluation.baselines.persistence import (
    DayAheadPersistence,
    RescaledPersistence,
)
from src.models.base import ModelInterface

ModelFactory = Callable[[pd.DataFrame], ModelInterface]

MODEL_FACTORIES: list[ModelFactory] = [
    DayAheadPersistence,
    RescaledPersistence,
    lambda history: LinearRegressionBaseline(),
]
