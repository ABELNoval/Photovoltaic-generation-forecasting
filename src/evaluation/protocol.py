"""Partición temporal del proyecto (docs/planning.md, sección 5) y las
comprobaciones que la hacen cumplir en el código.

| Uso                                          | Días                     |
|----------------------------------------------|--------------------------|
| Ajuste: entrenamiento y validación secuencial | 2014-01-01 .. 2015-12-31 |
| Comparación entre modelos                    | 2016-01-01 .. 2016-12-31 |
| Evaluación final, una sola vez               | 2017-01-01 .. 2017-12-31 |

Reglas que se comprueban:

- Se entrena solo con días de ajuste. En la evaluación final se puede
  entrenar también con 2016, pero nunca con 2017.
- La prueba empieza después del último día de entrenamiento.
- Ningún periodo de prueba toca 2017 salvo que se pida la evaluación final
  de forma explícita (`allow_final=True`, opción `--final` en la línea de
  órdenes), para que no se use por descuido al comparar o ajustar.

Los periodos son pares de días `(primero, último)`, ambos incluidos, con el
convenio de periods.py: el día 2015-12-31 termina en 2016-01-01 00:00, que
sigue siendo del ajuste.
"""

import pandas as pd

from src.evaluation.periods import select_period

Period = tuple[str, str]

TUNING: Period = ("2014-01-01", "2015-12-31")
COMPARISON: Period = ("2016-01-01", "2016-12-31")
FINAL: Period = ("2017-01-01", "2017-12-31")


def _days(period: Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    first, last = (pd.Timestamp(day, tz="UTC").normalize() for day in period)
    if first > last:
        raise ValueError(f"El periodo {period[0]}..{period[1]} termina antes de empezar; revisa el orden de las fechas.")
    return first, last


def _within(period: Period, bounds: Period) -> bool:
    first, last = _days(period)
    low, high = _days(bounds)
    return low <= first and last <= high


def _overlaps(period: Period, other: Period) -> bool:
    first, last = _days(period)
    low, high = _days(other)
    return first <= high and low <= last


def check_training_period(period: Period, allow_final: bool = False) -> None:
    """Falla si el entrenamiento sale del ajuste (o de 2014–2016 en la final)."""
    allowed = (TUNING[0], COMPARISON[1]) if allow_final else TUNING
    if not _within(period, allowed):
        raise ValueError(
            f"El entrenamiento {period[0]}..{period[1]} sale del periodo permitido "
            f"{allowed[0]}..{allowed[1]}; 2016 se reserva para comparar y 2017 para la "
            "evaluación final. Elige días dentro de ese periodo."
        )


def check_test_period(period: Period, allow_final: bool = False) -> None:
    """Falla si la prueba toca 2017 sin haber pedido la evaluación final."""
    _days(period)
    if _overlaps(period, FINAL) and not allow_final:
        raise ValueError(
            f"La prueba {period[0]}..{period[1]} incluye días de 2017, que se reservan "
            "para la evaluación final. Usa un periodo anterior o, solo para esa "
            "evaluación, la opción --final."
        )


def check_periods(train: Period, test: Period, allow_final: bool = False) -> None:
    """Comprueba entrenamiento y prueba y que la prueba sea posterior."""
    check_training_period(train, allow_final)
    check_test_period(test, allow_final)
    if _days(test)[0] <= _days(train)[1]:
        raise ValueError(
            f"La prueba {test[0]}..{test[1]} empieza antes de que termine el entrenamiento "
            f"{train[0]}..{train[1]}; la prueba debe ser posterior para no medir con datos vistos."
        )


def tuning_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filas del periodo de ajuste: lo único que debe ver el ajuste de hiperparámetros."""
    selected = select_period(df, *TUNING)
    if selected.empty:
        raise ValueError(
            f"El dataset no tiene datos en el periodo de ajuste {TUNING[0]}..{TUNING[1]}; "
            "comprueba que es data/sample.csv o data/processed/dataset_ID003.csv."
        )
    return selected
