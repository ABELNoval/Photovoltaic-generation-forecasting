import pytest

from src.data.dataset import load_dataset
from src.schema import TIMESTAMP


def test_load_dataset_returns_sorted_utc_index(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        f"{TIMESTAMP},power_norm\n"
        "2014-03-01 02:00:00+00:00,0.2\n"
        "2014-03-01 01:00:00+00:00,0.1\n"
    )
    df = load_dataset(path)
    assert df.index.name == TIMESTAMP
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df["power_norm"].tolist() == [0.1, 0.2]


def test_load_dataset_requires_timestamp_column(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("power_norm\n0.1\n")
    with pytest.raises(ValueError, match=TIMESTAMP):
        load_dataset(path)
