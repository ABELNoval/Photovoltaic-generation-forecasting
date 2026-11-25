import numpy as np
import pandas as pd
import pytest

from src.evaluation.sky import classify_sky, compare_by_sky
from src.schema import CLOUD_TOTAL, DAY_FLAG, TARGET


def test_classify_sky_uses_okta_thresholds():
    cloud = pd.Series([0, 25, 25.5, 74.9, 75, 100])
    expected = ["clear", "clear", "partly_cloudy", "partly_cloudy", "overcast", "overcast"]
    assert classify_sky(cloud).tolist() == expected


def test_classify_sky_rejects_missing_or_out_of_range_values():
    with pytest.raises(ValueError, match="Faltan 1"):
        classify_sky(pd.Series([10.0, np.nan]))
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        classify_sky(pd.Series([0.4, 101.0]))


def test_compare_by_sky_splits_the_common_hours():
    index = pd.date_range("2014-05-01 10:00", periods=4, freq="h", tz="UTC")
    observed = pd.DataFrame(
        {TARGET: [0.6, 0.6, 0.2, 0.2], DAY_FLAG: 1, CLOUD_TOTAL: [10, 0, 90, 80]}, index=index
    )
    predictions = {"good": observed[TARGET].copy(), "flat": pd.Series(0.4, index=index)}
    table = compare_by_sky(predictions, observed, index)

    assert table["sky_type"].astype(str).tolist() == ["clear", "clear", "overcast", "overcast"]
    assert table["n_hours"].tolist() == [2, 2, 2, 2]  # sin horas parciales: no sale ese grupo
    assert table.loc[0, "model"] == "good" and table.loc[0, "rmse"] == pytest.approx(0.0)
    assert table.loc[1, "mae"] == pytest.approx(0.2)
    assert table["r2"].isna().all()  # potencia constante en cada grupo: R² sin definir


def test_compare_by_sky_keeps_r2_when_defined():
    index = pd.date_range("2014-05-01 10:00", periods=3, freq="h", tz="UTC")
    observed = pd.DataFrame({TARGET: [0.2, 0.5, 0.8], DAY_FLAG: 1, CLOUD_TOTAL: [0, 5, 10]}, index=index)
    table = compare_by_sky({"exact": observed[TARGET].copy()}, observed, index)
    assert table.loc[0, "r2"] == pytest.approx(1.0)
