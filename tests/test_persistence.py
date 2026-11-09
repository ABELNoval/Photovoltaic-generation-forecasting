import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines.persistence import DayAheadPersistence, RescaledPersistence
from src.schema import GHI, TARGET


def _days(n_days, **columns):
    """n_days días completos desde el 2014-05-01 (01:00 .. 00:00 del día siguiente)."""
    index = pd.date_range("2014-05-01 01:00", periods=24 * n_days, freq="h", tz="UTC")
    return pd.DataFrame(columns, index=index)


def test_persistence_predicts_same_hour_of_previous_day():
    history = _days(2, **{TARGET: np.arange(48) / 100})
    X = history.iloc[24:]
    predictions = DayAheadPersistence(history).predict(X)
    assert predictions.index.equals(X.index)
    np.testing.assert_allclose(predictions.to_numpy(), np.arange(24) / 100)


def test_persistence_returns_nan_without_previous_day():
    history = _days(2, **{TARGET: np.ones(48)}).iloc[24:]  # falta el día anterior
    assert DayAheadPersistence(history).predict(history).isna().all()


def test_rescaled_persistence_scales_by_daily_irradiance():
    history = _days(2, **{TARGET: np.full(48, 0.5), GHI: [100.0] * 24 + [999.0] * 24})
    X = _days(2, **{GHI: np.full(48, 50.0)}).iloc[24:]  # pronóstico: la mitad que ayer
    predictions = RescaledPersistence(history).predict(X)
    np.testing.assert_allclose(predictions.to_numpy(), 0.25)


def test_rescaled_persistence_returns_nan_when_previous_day_had_no_irradiance():
    history = _days(2, **{TARGET: np.full(48, 0.5), GHI: np.zeros(48)})
    assert RescaledPersistence(history).predict(history.iloc[24:]).isna().all()


def test_baselines_require_their_history_columns():
    with pytest.raises(ValueError, match=GHI):
        RescaledPersistence(_days(1, **{TARGET: np.zeros(24)}))
    with pytest.raises(ValueError, match=TARGET):
        DayAheadPersistence(_days(1, **{GHI: np.zeros(24)}))
