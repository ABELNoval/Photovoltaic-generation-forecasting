import numpy as np
import pandas as pd
import pytest

from src.evaluation.periods import day_of
from src.evaluation.validation import cross_validate, sequential_folds
from src.models.base import ModelInterface
from src.schema import DAY_FLAG, INPUT_FEATURES, TARGET


def _dataset(start, n_days):
    """Días completos (01:00 .. 00:00) con luz de 08:00 a 16:00."""
    index = pd.date_range(f"{start} 01:00", periods=24 * n_days, freq="h", tz="UTC")
    df = pd.DataFrame(0.5, index=index, columns=INPUT_FEATURES)
    df[DAY_FLAG] = ((index.hour >= 8) & (index.hour <= 16)).astype(int)
    df[TARGET] = 0.4 * df[DAY_FLAG]
    return df


def test_every_fold_validates_only_on_later_whole_days():
    folds = sequential_folds(_dataset("2014-03-01", 60), n_splits=4)
    assert len(folds) == 4
    previous_train = 0
    for fold in folds:
        train_days = day_of(pd.DatetimeIndex(fold.train.index))
        val_days = day_of(pd.DatetimeIndex(fold.validation.index))
        assert train_days.max() < val_days.min()
        assert fold.train.index.max() < fold.validation.index.min()
        assert len(fold.train) % 24 == 0 and len(fold.validation) % 24 == 0  # días completos
        assert len(fold.train) > previous_train  # ventana creciente
        previous_train = len(fold.train)


def test_folds_never_reach_2016_or_2017():
    df = pd.concat([_dataset("2015-11-01", 61), _dataset("2016-01-01", 30), _dataset("2017-01-01", 30)])
    for fold in sequential_folds(df, n_splits=3):
        assert fold.validation.index.max() <= pd.Timestamp("2016-01-01 00:00", tz="UTC")


def test_folds_split_days_with_data_across_gaps():
    df = pd.concat([_dataset("2015-06-01", 30), _dataset("2015-11-01", 30)])  # hueco de meses
    folds = sequential_folds(df, n_splits=3)
    assert all(not fold.validation.empty for fold in folds)


def test_sequential_folds_validates_arguments():
    df = _dataset("2014-03-01", 3)
    with pytest.raises(ValueError, match="al menos 2"):
        sequential_folds(df, n_splits=1)
    with pytest.raises(ValueError, match="3 días"):
        sequential_folds(df, n_splits=3)


class _MeanModel(ModelInterface):
    name = "mean"

    def train(self, X, y):
        self.mean_ = float(y.mean())
        return self

    def predict(self, X):
        return pd.Series(self.mean_, index=X.index)


def test_cross_validate_scores_each_fold_on_daylight_hours():
    df = _dataset("2014-03-01", 40)
    table = cross_validate({"mean": lambda history: _MeanModel()}, df, n_splits=3)
    assert table["fold"].tolist() == [1, 2, 3]
    assert (table["train_end"] < table["validation_start"]).all()
    np.testing.assert_allclose(table["mae"], 0.0, atol=1e-12)  # potencia constante de día
    assert (table["n_hours"] % 9 == 0).all()  # 9 horas de luz por día validado


class _GappyModel(_MeanModel):
    """No predice las 12:00, como una persistencia tras un día sin datos."""

    name = "gappy"

    def predict(self, X):
        return super().predict(X).mask(X.index.hour == 12)


def test_cross_validate_scores_all_candidates_on_the_same_hours():
    df = _dataset("2014-03-01", 40)
    table = cross_validate({"mean": lambda h: _MeanModel(), "gappy": lambda h: _GappyModel()}, df, n_splits=3)
    per_fold = table.pivot(index="fold", columns="candidate", values="n_hours")
    assert (per_fold["mean"] == per_fold["gappy"]).all()
    assert (per_fold["mean"] % 8 == 0).all()  # sin las 12:00: 8 horas por día para los dos


def test_cross_validate_requires_candidates():
    with pytest.raises(ValueError, match="candidatos"):
        cross_validate({}, _dataset("2014-03-01", 10))
