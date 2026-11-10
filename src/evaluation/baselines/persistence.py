"""Referencias de persistencia (docs/planning.md, sección 4, Miembro 2).

Ambas usan la potencia medida el día anterior, que en un pronóstico real del
día siguiente ya se conoce. Como esa potencia no llega en `X`, cada referencia
recibe al crearse el dataset observado (`history`) y la busca allí por tiempo,
no por posición, para que los días descartados en la limpieza no la descuadren.
"""

import pandas as pd

from src.evaluation.periods import ONE_DAY, day_of
from src.models.base import ModelInterface
from src.schema import GHI, TARGET


def _require_columns(history: pd.DataFrame, columns: list[str], model: str) -> None:
    missing = [column for column in columns if column not in history.columns]
    if missing:
        raise ValueError(
            f"{model}: a history le faltan las columnas {missing}; pasa el dataset "
            "completo de src.data.dataset.load_dataset."
        )


def _previous_day(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Valor de `series` 24 horas antes de cada marca de `index` (NaN si falta)."""
    return pd.Series(series.reindex(index - ONE_DAY).to_numpy(), index=index)


class DayAheadPersistence(ModelInterface):
    """El día siguiente será igual al anterior, hora a hora."""

    name = "persistence"

    def __init__(self, history: pd.DataFrame):
        _require_columns(history, [TARGET], type(self).__name__)
        self.history = history

    def train(self, X: pd.DataFrame, y: pd.Series) -> "DayAheadPersistence":
        return self  # no aprende nada

    def predict(self, X: pd.DataFrame) -> pd.Series:
        index = pd.DatetimeIndex(X.index)
        return _previous_day(self.history[TARGET], index).rename("prediction")


class RescaledPersistence(ModelInterface):
    """Persistencia escalada por la radiación del día objetivo frente a la de ayer.

    P(D, h) = P(D-1, h) · ΣGHI(D) / ΣGHI(D-1). El factor es diario: conserva la
    forma de la curva de ayer y no se dispara al amanecer, cuando la GHI horaria
    es casi cero. ΣGHI(D) sale de `X`, que hace de pronóstico del día objetivo, y
    supone que `X` trae todas las horas de luz de cada día que predice (así lo
    pasa src/evaluation/run.py). ΣGHI(D-1) sale de la radiación observada. Si
    ayer no hubo radiación o falta el día, la predicción es NaN.
    """

    name = "rescaled_persistence"

    def __init__(self, history: pd.DataFrame):
        _require_columns(history, [TARGET, GHI], type(self).__name__)
        self.history = history

    def train(self, X: pd.DataFrame, y: pd.Series) -> "RescaledPersistence":
        return self  # no aprende nada

    def predict(self, X: pd.DataFrame) -> pd.Series:
        index = pd.DatetimeIndex(X.index)
        days = day_of(index)
        target_ghi = X[GHI].groupby(days).sum()
        observed_ghi = self.history[GHI].groupby(day_of(pd.DatetimeIndex(self.history.index))).sum()
        previous_ghi = observed_ghi.reindex(pd.DatetimeIndex(target_ghi.index) - ONE_DAY)
        previous_ghi = previous_ghi.where(previous_ghi > 0)  # sin radiación: factor indefinido
        factor = pd.Series(target_ghi.to_numpy() / previous_ghi.to_numpy(), index=target_ghi.index)

        power = _previous_day(self.history[TARGET], index)
        return (power * factor.reindex(days).to_numpy()).rename("prediction")
