import pandas as pd
import pytest

from src.data.pv_power import hourly_coverage, to_hourly


def test_to_hourly_labels_at_end_and_requires_45_minutes():
    index = pd.DatetimeIndex(
        list(pd.date_range("2014-01-01 10:16", periods=44, freq="min", tz="UTC"))
        + list(pd.date_range("2014-01-01 11:01", periods=16, freq="min", tz="UTC"))
    )
    power = pd.Series(
        [100.0] * 44 + [200.0] * 16,
        index=index,
    )

    hourly = to_hourly(power, min_minutes=45)

    assert hourly.index[0] == pd.Timestamp("2014-01-01 11:00:00", tz="UTC")
    assert pd.isna(hourly.iloc[0])


def test_to_hourly_accepts_45_observed_minutes():
    power = pd.Series(
        [100.0] * 45,
        index=pd.date_range("2014-01-01 10:16", periods=45, freq="min", tz="UTC"),
    )

    hourly = to_hourly(power, min_minutes=45)

    assert hourly.iloc[0] == 100.0


@pytest.mark.parametrize(
    "tz",
    [
        None,
        "UTC-04:00",
    ],
)
def test_to_hourly_normalizes_index_to_utc(tz):
    index = pd.date_range("2014-01-01 10:16", periods=45, freq="min", tz=tz)
    power = pd.Series([100.0] * 45, index=index)

    hourly = to_hourly(power, min_minutes=45)

    hourly_index = pd.DatetimeIndex(hourly.index)
    assert hourly_index.tz is not None
    assert str(hourly_index.tz) == "UTC"


def test_to_hourly_rejects_duplicate_timestamps():
    power = pd.Series(
        [100.0, 200.0],
        index=pd.DatetimeIndex(
            ["2014-01-01 10:00:00+00:00", "2014-01-01 10:00:00+00:00"]
        ),
    )

    with pytest.raises(ValueError, match="duplicados"):
        to_hourly(power)


def test_hourly_coverage_is_reported_by_year():
    index = pd.DatetimeIndex(
        [
            "2014-01-01 01:00:00+00:00",
            "2014-12-31 23:00:00+00:00",
            "2015-01-01 00:00:00+00:00",
        ]
    )
    hourly = pd.Series([1.0, pd.NA, 2.0], index=index)

    coverage = hourly_coverage(
        hourly,
        start=pd.Timestamp("2014-01-01", tz="UTC"),
        end=pd.Timestamp("2014-12-31", tz="UTC"),
    )

    assert coverage.loc[2014, "hours_expected"] == 8760
    assert coverage.loc[2014, "hours_with_data"] == 2
    assert coverage.loc[2014, "missing_hours"] == 8758
