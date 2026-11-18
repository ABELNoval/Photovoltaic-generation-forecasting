"""Multilayer perceptron for day-ahead photovoltaic power forecasting.

This is the initial Phase 3 implementation. It uses the mandatory project
features and a fixed baseline configuration; hyperparameter tuning is out of
scope for this phase.
"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import ModelInterface
from src.schema import REQUIRED_FEATURES


class MLPModel(ModelInterface):
    """MLP regression model implementing the shared model interface."""

    name = "mlp"

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (8,),
        activation: str = "relu",
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_: Pipeline | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> "MLPModel":
        """Fit the scaler and MLP using only the supplied training data."""
        self.model_ = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=self.hidden_layer_sizes,
                        activation=self.activation,
                        solver="adam",
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        early_stopping=False,
                    ),
                ),
            ]
        )
        self.model_.fit(X.loc[:, REQUIRED_FEATURES], y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict power for every row, preserving the input index."""
        if self.model_ is None:
            raise RuntimeError("Hay que llamar a train antes de predict.")

        values = self.model_.predict(X.loc[:, REQUIRED_FEATURES])
        return pd.Series(values, index=X.index, name="prediction")