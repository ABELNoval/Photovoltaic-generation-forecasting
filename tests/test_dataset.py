import numpy as np
import pandas as pd
import pytest

import src.data.build_dataset as dataset_builder
from src.data.dataset import load_dataset
from src.data.paths import PROCESSED_DIR
from src.schema import (
    CALENDAR_FEATURES,
    CORE_FEATURES,
    DAY_FLAG,
    SECONDARY_FEATURES,
    TARGET,
    TIMESTAMP,
)


def test_complete_id003_dataset_matches_shared_contract():
    df = load_dataset(PROCESSED_DIR / "dataset_ID003.csv")
    expected_columns = [
        TARGET,
        *CORE_FEATURES,
        *SECONDARY_FEATURES,
        *CALENDAR_FEATURES,
        DAY_FLAG,
    ]

    assert list(df.columns) == expected_columns
    assert df.index[0] == pd.Timestamp("2014-01-01 01:00", tz="UTC")
    assert df.index[-1] == pd.Timestamp("2018-01-01 00:00", tz="UTC")
    index = df.index
    assert isinstance(index, pd.DatetimeIndex)
    assert str(index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert df[TARGET].between(0, 1).all()


def test_clean_clips_negative_power_sets_missing_night_to_zero_and_interpolates():
    index = pd.date_range("2014-01-01 00:00", periods=5, freq="h", tz="UTC")
    source = pd.DataFrame(
        {
            "ghi": [10.0, 0.0, 10.0, 10.0, 10.0],
            "power_w": [-50.0, np.nan, 100.0, np.nan, 300.0],
        },
        index=index,
    )

    cleaned = dataset_builder.clean(source, ac_capacity=1000.0)

    assert cleaned.loc[index[0], "_negative"]
    assert cleaned.loc[index[0], TARGET] == 0.0
    assert cleaned.loc[index[1], "_night_zero"]
    assert cleaned.loc[index[1], TARGET] == 0.0
    assert cleaned.loc[index[3], "_interpolated"]
    assert cleaned.loc[index[3], TARGET] == 0.2


@pytest.mark.parametrize(
    ("gap_length", "should_interpolate"),
    [(dataset_builder.MAX_GAP_HOURS, True), (dataset_builder.MAX_GAP_HOURS + 1, False)],
)
def test_fill_short_gaps_respects_maximum_gap(gap_length, should_interpolate):
    index = pd.date_range("2014-01-01", periods=gap_length + 2, freq="h", tz="UTC")
    values = pd.Series(
        [0.0, *([np.nan] * gap_length), float(gap_length + 1)],
        index=index,
    )

    filled, interpolated = dataset_builder._fill_short_gaps(
        values,
        dataset_builder.MAX_GAP_HOURS,
    )

    assert interpolated.sum() == (gap_length if should_interpolate else 0)
    if should_interpolate:
        assert filled.iloc[1:-1].tolist() == list(map(float, range(1, gap_length + 1)))
    else:
        assert filled.iloc[1:-1].isna().all()


def test_build_dataset_discards_day_with_unfilled_daytime_gap(monkeypatch):
    index = pd.date_range("2014-12-31 01:00", periods=48, freq="h", tz="UTC")
    source = pd.DataFrame(
        {
            **{feature: 0.0 for feature in CORE_FEATURES + SECONDARY_FEATURES},
            "power_w": 100.0,
        },
        index=index,
    )
    source["ghi"] = 100.0
    source.loc[index[9:13], "power_w"] = np.nan
    source.loc[index[14:17], "power_w"] = np.nan
    source.loc[index[30:32], "power_w"] = np.nan
    metadata = pd.DataFrame(
        {"estimated_ac_capacity": [1000.0]},
        index=["ID003"],
    )
    monkeypatch.setattr(dataset_builder, "load_metadata", lambda: metadata)
    monkeypatch.setattr(dataset_builder, "load_sources", lambda _: source)

    dataset, stats = dataset_builder.build_dataset(
        "ID003",
        start="2014-12-31",
        end="2015-01-01",
    )

    assert stats["days_dropped"] == 1
    assert stats["days"] == 1
    assert stats["rows"] == 24
    assert stats["hours_expected"] == 48
    assert stats["power_hours_with_data"] == 39
    assert stats["power_missing_hours"] == 9
    assert stats["hours_interpolated_before_day_discard"] == 5
    assert stats["hours_interpolated"] == 2
    assert stats["hours_interpolated_on_discarded_days"] == 3
    assert stats["unresolved_power_hours"] == 4
    assert stats["missing_values"] == 0
    annual_2014, annual_2015 = stats["annual"]
    assert annual_2014["year"] == 2014
    assert annual_2014["hours_expected"] == 24
    assert annual_2014["power_hours_with_data"] == 17
    assert annual_2014["days_dropped"] == 1
    assert annual_2014["days_kept"] == 0
    assert annual_2014["hours_interpolated_on_discarded_days"] == 3
    assert annual_2014["rows"] == 0
    assert annual_2015["year"] == 2015
    assert annual_2015["hours_expected"] == 24
    assert annual_2015["power_hours_with_data"] == 22
    assert annual_2015["days_dropped"] == 0
    assert annual_2015["days_kept"] == 1
    assert annual_2015["hours_interpolated"] == 2
    assert annual_2015["rows"] == 24
    assert annual_2015["daytime_rows"] == 24
    assert dataset[TIMESTAMP].iloc[0] == pd.Timestamp("2015-01-01 01:00", tz="UTC")
    assert dataset[TIMESTAMP].iloc[-1] == pd.Timestamp("2015-01-02 00:00", tz="UTC")


def test_load_dataset_returns_sorted_utc_index(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        f"{TIMESTAMP},power_norm\n"
        "2014-03-01 02:00:00+00:00,0.2\n"
        "2014-03-01 01:00:00+00:00,0.1\n"
    )
    df = load_dataset(path)
    assert df.index.name == TIMESTAMP
    index = df.index
    assert isinstance(index, pd.DatetimeIndex)
    assert str(index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df["power_norm"].tolist() == [0.1, 0.2]


def test_load_dataset_requires_timestamp_column(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("power_norm\n0.1\n")
    with pytest.raises(ValueError, match=TIMESTAMP):
        load_dataset(path)
