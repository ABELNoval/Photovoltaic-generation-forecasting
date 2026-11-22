import numpy as np
import pandas as pd
import pytest

from src.models.svr_model import SVRModel
from src.schema import REQUIRED_FEATURES, SECONDARY_FEATURES


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2014-05-01 01:00", periods=n, freq="h", tz="UTC")
    columns = REQUIRED_FEATURES + SECONDARY_FEATURES
    X = pd.DataFrame(
        rng.normal(size=(n, len(columns))),
        index=index,
        columns=columns,
    )
    y = pd.Series(rng.uniform(size=n), index=index)
    return X, y


def test_svr_trains_and_predicts_with_x_index():
    X, y = _data()
    model = SVRModel(C=1.0, epsilon=0.01, kernel="rbf", gamma=0.003)

    assert model.train(X, y) is model
    predictions = model.predict(X)

    assert predictions.index.equals(X.index)
    assert predictions.notna().all()
    assert len(predictions) == len(X)


def test_svr_uses_required_and_selected_secondary_features():
    X, y = _data()
    model = SVRModel(secondary_features=["humidity", "pressure"]).train(X, y)

    assert model.columns_ == REQUIRED_FEATURES + ["humidity", "pressure"]
    assert model.predict(X).notna().all()


def test_svr_rejects_invalid_secondary_features():
    X, y = _data()

    with pytest.raises(ValueError, match="wind_speed"):
        SVRModel(secondary_features=["wind_speed"]).train(X, y)


def test_svr_scales_with_training_data_only():
    X, y = _data()
    model = SVRModel(C=1.0).train(X, y)
    shifted = X + 1000.0

    assert model.predict(shifted).notna().all()
    assert model.pipeline_ is not None
    scaler = model.pipeline_.named_steps["scaler"]
    np.testing.assert_allclose(
        scaler.mean_,
        X[REQUIRED_FEATURES].mean().to_numpy(),
    )


def test_svr_requires_training_before_predict():
    X, _ = _data()

    with pytest.raises(RuntimeError, match="train"):
        SVRModel().predict(X)
