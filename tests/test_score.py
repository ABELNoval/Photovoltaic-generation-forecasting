import numpy as np
import pandas as pd
import pytest

from src.evaluation.score import (
    COMPARISON_FILE,
    compare,
    load_predictions,
    read_predictions,
    score_results,
)
from src.models.utils import save_predictions
from src.schema import CLOUD_TOTAL, DAY_FLAG, TARGET, TIMESTAMP


def _observed():
    index = pd.date_range("2014-05-01 10:00", periods=4, freq="h", tz="UTC", name=TIMESTAMP)
    return pd.DataFrame({TARGET: [0.2, 0.5, 0.8, 0.0], DAY_FLAG: [1, 1, 1, 0]}, index=index)


def test_compare_scores_all_models_on_the_same_hours():
    observed = _observed()
    exact = pd.Series([0.2, 0.5, 0.8, 0.0], index=observed.index)
    with_gap = pd.Series([np.nan, 0.6, 0.8, 0.0], index=observed.index)

    with pytest.warns(UserWarning, match="horas"):
        table = compare({"exact": exact, "with_gap": with_gap}, observed, min_coverage=0.5)

    assert table["model"].tolist() == ["exact", "with_gap"]  # ordenada por RMSE
    assert table["n_hours"].tolist() == [2, 2]  # la hora sin predicción sale para todos
    assert table.loc[0, "rmse"] == pytest.approx(0.0)
    assert table["coverage"].tolist() == pytest.approx([1.0, 2 / 3])


def test_compare_rejects_predictions_without_common_hours():
    observed = _observed()
    elsewhere = pd.Series([0.1], index=pd.DatetimeIndex(["2016-05-01 12:00"], tz="UTC"))
    with pytest.raises(ValueError, match="old_model"):
        compare({"old_model": elsewhere}, observed)


def test_load_predictions_round_trips_saved_files(tmp_path):
    observed = _observed()
    save_predictions(observed[TARGET], "exact", results_dir=tmp_path)
    loaded = load_predictions(tmp_path)
    assert list(loaded) == ["exact"]
    # Se compara por valor: la resolución (ns/us) puede diferir tras leer el CSV.
    assert list(loaded["exact"].index) == list(observed.index)


def test_load_predictions_requires_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.evaluation.run"):
        load_predictions(tmp_path)


def test_compare_rejects_models_below_minimum_coverage():
    observed = _observed()
    partial = pd.Series([np.nan, 0.5, 0.8, 0.0], index=observed.index)
    with pytest.raises(ValueError, match="partial: 67%"):
        compare({"partial": partial}, observed)


def test_compare_rejects_power_predicted_at_night():
    observed = _observed()
    shifted = pd.Series([0.2, 0.5, 0.8, 0.3], index=observed.index)  # 0.3 en la hora de noche
    with pytest.raises(ValueError, match="shifted: predice potencia en 1 horas de noche"):
        compare({"shifted": shifted}, observed)


def _write(tmp_path, text, name="model"):
    path = tmp_path / f"{name}_predictions.csv"
    path.write_text(f"{TIMESTAMP},prediction\n{text}")
    return path


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("2014-05-01 10:00:00,0.1\n", "sin zona horaria"),
        ("2014-05-01 10:30:00+00:00,0.1\n", "no caen en punto"),
        ("2014-05-01 10:00:00+00:00,0.1\n2014-05-01 10:00:00+00:00,0.2\n", "repetidas"),
        ("2014-05-01 10:00:00+00:00,abc\n", "no son números"),
        ("2014-05-01 10:00:00+00:00,1.2\n", "fuera de"),
        ("not-a-date+00:00,0.1\n", "no se pueden leer"),
        ("", "no tiene filas"),
    ],
)
def test_read_predictions_rejects_malformed_files(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        read_predictions(_write(tmp_path, rows))


def test_read_predictions_requires_columns(tmp_path):
    path = tmp_path / "bad_predictions.csv"
    path.write_text("time,value\n2014-05-01 10:00:00+00:00,0.1\n")
    with pytest.raises(ValueError, match="faltan las columnas"):
        read_predictions(path)


def test_read_predictions_converts_offsets_to_utc(tmp_path):
    loaded = read_predictions(_write(tmp_path, "2014-05-01 12:00:00+02:00,0.1\n"))
    assert loaded.index[0] == pd.Timestamp("2014-05-01 10:00", tz="UTC")


def test_compare_warns_about_hours_missing_from_the_dataset():
    observed = _observed().drop(pd.Timestamp("2014-05-01 11:00", tz="UTC"))
    full = pd.Series([0.2, 0.5, 0.8, 0.0], index=_observed().index)  # incluye las 11:00
    with pytest.warns(UserWarning, match="full: 1"):
        compare({"full": full}, observed)


def test_score_results_writes_the_comparison_table(tmp_path):
    index = pd.date_range("2014-05-01 01:00", periods=24, freq="h", tz="UTC", name=TIMESTAMP)
    is_day = ((index.hour >= 8) & (index.hour <= 16)).astype(int)
    df = pd.DataFrame({TARGET: 0.5 * is_day, DAY_FLAG: is_day, CLOUD_TOTAL: 10}, index=index)
    save_predictions(df[TARGET], "exact", results_dir=tmp_path)
    save_predictions(0.4 * df[DAY_FLAG], "low", results_dir=tmp_path)

    table = score_results(df, ("2014-05-01", "2014-05-01"), tmp_path)

    assert (tmp_path / COMPARISON_FILE).exists()
    assert table["model"].tolist() == ["exact", "low"]
    assert table["n_hours"].tolist() == [9, 9]


def test_score_results_refuses_2017_without_final(tmp_path):
    index = pd.date_range("2017-05-01 10:00", periods=2, freq="h", tz="UTC", name=TIMESTAMP)
    df = pd.DataFrame({TARGET: [0.5, 0.5], DAY_FLAG: [1, 1], CLOUD_TOTAL: [10, 10]}, index=index)
    with pytest.raises(ValueError, match="--final"):
        score_results(df, ("2017-05-01", "2017-05-01"), tmp_path)
