import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src.models.knn import KNNModel
from src.schema import REQUIRED_FEATURES, SECONDARY_FEATURES


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2014-05-01 01:00", periods=n, freq="h", tz="UTC")
    columns = REQUIRED_FEATURES + SECONDARY_FEATURES
    X = pd.DataFrame(rng.normal(size=(n, len(columns))), index=index, columns=columns)
    y = pd.Series(rng.uniform(size=n), index=index)
    return X, y


def test_knn_with_k_one_reproduces_training_targets():
    X, y = _data()
    predictions = KNNModel(k=1).train(X, y).predict(X)
    assert predictions.index.equals(X.index)
    np.testing.assert_allclose(predictions.to_numpy(), y.to_numpy())


def test_knn_averages_the_k_nearest_hours():
    # Un solo atributo útil: las tres horas más cercanas a 0.4 son 0.0, 0.5 y 1.0.
    X, _ = _data(n=5)
    X[REQUIRED_FEATURES] = 0.0
    X["ghi"] = [0.0, 0.5, 1.0, 10.0, 20.0]
    y = pd.Series([0.1, 0.2, 0.3, 0.9, 0.9], index=X.index)
    query = X.iloc[[0]].copy()
    query["ghi"] = 0.4
    prediction = KNNModel(k=3).train(X, y).predict(query)
    assert prediction.iloc[0] == pytest.approx((0.1 + 0.2 + 0.3) / 3)


def test_knn_matches_scikit_learn_reference():
    X, y = _data(n=200, seed=1)
    train_X, train_y, test_X = X.iloc[:150], y.iloc[:150], X.iloc[150:]
    ours = KNNModel(k=5).train(train_X, train_y).predict(test_X)

    scaler = StandardScaler().fit(train_X[REQUIRED_FEATURES])
    reference = KNeighborsRegressor(n_neighbors=5).fit(scaler.transform(train_X[REQUIRED_FEATURES]), train_y)
    expected = reference.predict(scaler.transform(test_X[REQUIRED_FEATURES]))
    np.testing.assert_allclose(ours.to_numpy(), expected)


def test_knn_scales_with_training_statistics_only():
    X, y = _data()
    model = KNNModel().train(X, y)
    shifted = X + 1000.0  # otros datos no deben cambiar el escalado ya ajustado
    assert model.predict(shifted).notna().all()
    assert model.mean_ is not None
    np.testing.assert_allclose(model.mean_, X[REQUIRED_FEATURES].mean().to_numpy())


def test_knn_handles_constant_feature_and_ignores_unused_columns():
    X, y = _data()
    X["temperature"] = 5.0
    predictions = KNNModel().train(X, y).predict(X.drop(columns=SECONDARY_FEATURES))
    assert predictions.notna().all()


def test_knn_rejects_invalid_k():
    X, y = _data(n=3)
    with pytest.raises(ValueError, match="k"):
        KNNModel(k=0)
    with pytest.raises(ValueError, match="k"):
        KNNModel(k=4).train(X, y)


def test_knn_requires_training_before_predict():
    X, _ = _data()
    with pytest.raises(RuntimeError, match="train"):
        KNNModel().predict(X)
