import pandas as pd

from src.evaluation.periods import day_of, select_period


def _hourly(start, periods):
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    return pd.DataFrame({"x": range(periods)}, index=index)


def test_day_of_assigns_midnight_to_previous_day():
    index = pd.DatetimeIndex(["2014-05-01 01:00", "2014-05-02 00:00"], tz="UTC")
    assert list(day_of(index)) == [pd.Timestamp("2014-05-01", tz="UTC")] * 2


def test_select_period_follows_end_of_hour_convention():
    df = _hourly("2014-05-01 00:00", 49)  # 05-01 00:00 .. 05-03 00:00
    selected = select_period(df, "2014-05-01", "2014-05-01")
    assert len(selected) == 24
    assert selected.index[0] == pd.Timestamp("2014-05-01 01:00", tz="UTC")
    assert selected.index[-1] == pd.Timestamp("2014-05-02 00:00", tz="UTC")
