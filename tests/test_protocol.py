import pandas as pd
import pytest

from src.evaluation.protocol import (
    check_periods,
    check_test_period,
    check_training_period,
    tuning_data,
)


def test_training_must_stay_in_tuning_years():
    check_training_period(("2014-01-01", "2015-12-31"))
    with pytest.raises(ValueError, match="2016 se reserva"):
        check_training_period(("2015-06-01", "2016-01-01"))


def test_final_evaluation_may_train_with_2016_but_never_2017():
    check_training_period(("2014-01-01", "2016-12-31"), allow_final=True)
    with pytest.raises(ValueError, match="sale del periodo"):
        check_training_period(("2014-01-01", "2017-01-01"), allow_final=True)


def test_test_period_rejects_2017_unless_final_is_requested():
    check_test_period(("2016-01-01", "2016-12-31"))
    with pytest.raises(ValueError, match="--final"):
        check_test_period(("2016-12-01", "2017-01-15"))
    check_test_period(("2017-01-01", "2017-12-31"), allow_final=True)


def test_test_period_must_follow_training():
    check_periods(("2014-03-01", "2014-04-30"), ("2014-05-01", "2014-05-31"))
    with pytest.raises(ValueError, match="posterior"):
        check_periods(("2014-03-01", "2014-04-30"), ("2014-04-30", "2014-05-31"))


def test_reversed_period_is_rejected():
    with pytest.raises(ValueError, match="antes de empezar"):
        check_test_period(("2016-02-01", "2016-01-01"))


def test_tuning_data_keeps_last_hour_of_2015_and_drops_2016():
    index = pd.date_range("2015-12-31 23:00", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"x": [1, 2, 3]}, index=index)  # 23:00, 00:00 (aún 2015-12-31), 01:00 (2016)
    assert list(tuning_data(df).index) == list(index[:2])


def test_tuning_data_requires_tuning_rows():
    index = pd.date_range("2016-05-01 01:00", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="periodo de ajuste"):
        tuning_data(pd.DataFrame({"x": [1, 2]}, index=index))
