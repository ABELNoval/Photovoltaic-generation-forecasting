from typing import Literal

from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.models.base import ModelInterface
from src.schema import REQUIRED_FEATURES, SECONDARY_FEATURES

Kernel = Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]


class SVRModel(ModelInterface):
    name = "svr"

    def __init__(
        self,
        C: float = 10.0,
        epsilon: float = 0.01,
        kernel: Kernel = "rbf",
        gamma: float = 0.003,
        secondary_features=(),
        random_state: int = 42,
    ):
        # Hiperparámetros
        self.C = C
        self.epsilon = epsilon
        self.kernel: Kernel = kernel
        self.gamma = gamma
        # Columnas secundarias

        self.secondary_features: list[str] = list(secondary_features)
        self.random_state = random_state

        self.columns_: list[str] | None = None
        self.pipeline_: Pipeline | None = None

    def _select_columns(self) -> list[str]:
        invalid = set(self.secondary_features) - set(SECONDARY_FEATURES)
        if invalid:
            raise ValueError(f"Columnas secundarias no válidas: {sorted(invalid)}")

        return list(REQUIRED_FEATURES) + list(self.secondary_features)

    def train(self, X: DataFrame, y: Series) -> "SVRModel":
        # Columnas elegidas, guardadas para usar las mismas en predict
        self.columns_ = self._select_columns()
        self.pipeline_ = Pipeline([("scaler", StandardScaler()),
        ("svr",SVR(C=self.C,epsilon=self.epsilon,kernel=self.kernel,gamma=self.gamma,),),])
        self.pipeline_.fit(X[self.columns_], y)
        return self

    def predict(self, X: DataFrame) -> Series:
        if self.pipeline_ is None:
                raise RuntimeError("El modelo no está entrenado: llama a train() primero.")
        # Mismas columnas y mismo escalado que en train
        raw = self.pipeline_.predict(X[self.columns_])
        return Series(raw, index=X.index)

