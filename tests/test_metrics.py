import pandas as pd
import pytest

from src.evaluation.metrics import METRICS, evaluate


def _series(values):
    index = pd.date_range("2014-05-01 10:00", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


def test_evaluate_matches_hand_computed_example():
    # Errores de día: +0.1, -0.1, 0.0. La hora de noche (error 0.9) no cuenta.
    y_true = _series([0.2, 0.5, 0.8, 0.0])
    y_pred = _series([0.3, 0.4, 0.8, 0.9])
    is_day = _series([1, 1, 1, 0])

    result = evaluate(y_true, y_pred, is_day)

    assert result["mae"] == pytest.approx(0.2 / 3)
    assert result["rmse"] == pytest.approx((0.02 / 3) ** 0.5)
    assert result["nrmse"] == pytest.approx(100 * (0.02 / 3) ** 0.5)
    # media 0.5; SS_tot = 0.18; SS_res = 0.02
    assert result["r2"] == pytest.approx(1 - 0.02 / 0.18)


def test_evaluate_returns_every_metric_in_order():
    y = _series([0.2, 0.5])
    assert list(evaluate(y, y, _series([1, 1]))) == list(METRICS)


def test_evaluate_rejects_misaligned_series():
    y_true = _series([0.2, 0.5])
    y_pred = _series([0.2, 0.5]).shift(1, freq="h")
    with pytest.raises(ValueError, match="mismo índice"):
        evaluate(y_true, y_pred, _series([1, 1]))


def test_evaluate_rejects_missing_daytime_values():
    with pytest.raises(ValueError, match="faltantes"):
        evaluate(_series([0.2, 0.5]), _series([0.2, None]), _series([1, 1]))


def test_evaluate_ignores_missing_values_at_night():
    result = evaluate(_series([0.2, 0.5, 0.0]), _series([0.2, 0.5, None]), _series([1, 1, 0]))
    assert result["mae"] == pytest.approx(0.0)


def test_evaluate_rejects_period_without_daylight():
    with pytest.raises(ValueError, match="horas de luz"):
        evaluate(_series([0.0]), _series([0.0]), _series([0]))
