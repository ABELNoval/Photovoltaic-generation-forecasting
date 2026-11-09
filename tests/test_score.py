import numpy as np
import pandas as pd
import pytest

from src.evaluation.score import compare, load_predictions
from src.models.utils import save_predictions
from src.schema import DAY_FLAG, TARGET, TIMESTAMP


def _observed():
    index = pd.date_range("2014-05-01 10:00", periods=4, freq="h", tz="UTC", name=TIMESTAMP)
    return pd.DataFrame({TARGET: [0.2, 0.5, 0.8, 0.0], DAY_FLAG: [1, 1, 1, 0]}, index=index)


def test_compare_scores_all_models_on_the_same_hours():
    observed = _observed()
    exact = pd.Series([0.2, 0.5, 0.8, 0.0], index=observed.index)
    with_gap = pd.Series([np.nan, 0.6, 0.8, 0.0], index=observed.index)

    with pytest.warns(UserWarning, match="horas"):
        table = compare({"exact": exact, "with_gap": with_gap}, observed)

    assert table["model"].tolist() == ["exact", "with_gap"]  # ordenada por RMSE
    assert table["n_hours"].tolist() == [2, 2]  # la hora sin predicción sale para todos
    assert table.loc[0, "rmse"] == pytest.approx(0.0)


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
