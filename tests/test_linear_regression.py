import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines.linear_regression import LinearRegressionBaseline
from src.schema import REQUIRED_FEATURES, SECONDARY_FEATURES


def _data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2014-05-01 01:00", periods=n, freq="h", tz="UTC")
    columns = REQUIRED_FEATURES + SECONDARY_FEATURES
    X = pd.DataFrame(rng.normal(size=(n, len(columns))), index=index, columns=columns)
    y = X[REQUIRED_FEATURES] @ np.linspace(0.1, 1.0, len(REQUIRED_FEATURES)) + 0.3
    return X, y


def test_linear_regression_recovers_linear_target_with_x_index():
    X, y = _data()
    predictions = LinearRegressionBaseline().train(X, y).predict(X)
    assert predictions.index.equals(X.index)
    np.testing.assert_allclose(predictions.to_numpy(), y.to_numpy(), atol=1e-9)


def test_linear_regression_requires_training_before_predict():
    X, _ = _data()
    with pytest.raises(RuntimeError, match="train"):
        LinearRegressionBaseline().predict(X)
