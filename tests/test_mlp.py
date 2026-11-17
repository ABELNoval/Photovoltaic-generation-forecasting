import numpy as np
import pandas as pd
import pytest

from src.models.mlp import MLPModel
from src.schema import REQUIRED_FEATURES


def _data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2014-05-01 01:00", periods=n, freq="h", tz="UTC")
    X = pd.DataFrame(
        rng.normal(size=(n, len(REQUIRED_FEATURES))),
        index=index,
        columns=REQUIRED_FEATURES,
    )
    y = pd.Series(rng.random(n), index=index)
    return X, y


def test_mlp_trains_and_predicts_with_x_index():
    X, y = _data()
    model = MLPModel(hidden_layer_sizes=(3,), max_iter=500)
    assert model.train(X, y) is model
    predictions = model.predict(X)
    assert predictions.index.equals(X.index)
    assert predictions.notna().all()


def test_mlp_requires_training_before_predict():
    X, _ = _data()
    with pytest.raises(RuntimeError, match="train"):
        MLPModel().predict(X)
