import pandas as pd
import pytest

from src.evaluation.run import run_models
from src.models.base import ModelInterface
from src.schema import DAY_FLAG, INPUT_FEATURES, TARGET


class _ConstantModel(ModelInterface):
    name = "constant"

    def train(self, X, y):
        return self

    def predict(self, X):
        return pd.Series(0.5, index=X.index)


def _dataset(start, n_days):
    index = pd.date_range(f"{start} 01:00", periods=24 * n_days, freq="h", tz="UTC")
    df = pd.DataFrame(0.5, index=index, columns=INPUT_FEATURES)
    df[DAY_FLAG] = ((index.hour >= 8) & (index.hour <= 16)).astype(int)
    df[TARGET] = 0.4 * df[DAY_FLAG]
    return df


def test_run_models_saves_postprocessed_predictions(tmp_path):
    df = _dataset("2014-05-01", 4)
    paths = run_models(df, ("2014-05-01", "2014-05-02"), ("2014-05-03", "2014-05-04"), [lambda h: _ConstantModel()], tmp_path)
    saved = pd.read_csv(paths[0])
    assert len(saved) == 48
    assert set(saved["prediction"]) == {0.0, 0.5}  # noche a cero


@pytest.mark.parametrize(
    ("train", "test", "message"),
    [
        (("2016-01-01", "2016-01-31"), ("2016-02-01", "2016-02-28"), "2016 se reserva"),
        (("2015-01-01", "2015-01-31"), ("2017-01-01", "2017-01-31"), "--final"),
        (("2014-05-01", "2014-05-10"), ("2014-05-05", "2014-05-20"), "posterior"),
    ],
)
def test_run_models_enforces_the_temporal_protocol(tmp_path, train, test, message):
    with pytest.raises(ValueError, match=message):
        run_models(_dataset("2014-05-01", 2), train, test, [lambda h: _ConstantModel()], tmp_path)
    assert not list(tmp_path.iterdir())  # falla antes de entrenar o escribir nada
