"""Validación cruzada secuencial dentro del periodo de ajuste (2014–2015).

Ventana creciente por días: los días con datos del ajuste se reparten en
`n_splits + 1` bloques consecutivos; el pliegue k entrena con los bloques
0..k-1 y valida con el bloque k, siempre posterior. Así cada validación usa
solo observaciones futuras respecto a su entrenamiento, como el pronóstico
real del día siguiente.

- Se parte por días y no por horas para que ningún día quede repartido entre
  entrenamiento y validación: la persistencia reescalada suma la radiación
  del día completo.
- Se cuentan los días con datos y no los del calendario, para que los huecos
  del dataset (casi sin datos entre septiembre de 2015 y marzo de 2016) no
  dejen pliegues vacíos.
- 2016 y 2017 no llegan nunca: todo pasa antes por `protocol.tuning_data`.

Uso, desde la raíz del repositorio (valida todos los modelos del registro,
con su configuración por defecto):

    python -m src.evaluation.validation
    python -m src.evaluation.validation --data data/processed/dataset_ID003.csv --n-splits 5
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.data.dataset import load_dataset
from src.evaluation.metrics import METRICS, evaluate
from src.evaluation.periods import day_of
from src.evaluation.protocol import tuning_data
from src.evaluation.registry import MODEL_FACTORIES, ModelFactory
from src.evaluation.score import DEFAULT_DATA
from src.models.utils import postprocess_predictions, split_xy
from src.schema import DAY_FLAG, TARGET

DEFAULT_SPLITS = 5


@dataclass(frozen=True)
class Fold:
    """Un pliegue: filas completas (día y noche) de entrenamiento y validación."""

    number: int
    train: pd.DataFrame
    validation: pd.DataFrame


def sequential_folds(df: pd.DataFrame, n_splits: int = DEFAULT_SPLITS) -> list[Fold]:
    """Pliegues de ventana creciente sobre los días de ajuste de `df`.

    `df` debe venir de load_dataset; se recorta al periodo de ajuste aquí.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits debe ser al menos 2, no {n_splits}; con uno solo no hay validación cruzada.")
    tuning = tuning_data(df)
    row_days = day_of(pd.DatetimeIndex(tuning.index))
    days = row_days.unique()
    if len(days) <= n_splits:
        raise ValueError(
            f"El periodo de ajuste tiene {len(days)} días y se piden {n_splits} pliegues; "
            "usa menos pliegues o un dataset con más días."
        )

    folds = []
    for number, (train_pos, val_pos) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(days), start=1):
        train = tuning[row_days.isin(days[train_pos])]
        validation = tuning[row_days.isin(days[val_pos])]
        # Invariante de la partición: si fallara, la validación vería el pasado
        # de otro modo que el pronóstico real y las métricas no valdrían.
        if train.index.max() >= validation.index.min():
            raise RuntimeError(f"El pliegue {number} valida con horas anteriores a su entrenamiento.")
        folds.append(Fold(number, train, validation))
    return folds


def _predict_fold(factory: ModelFactory, history: pd.DataFrame, fold: Fold) -> pd.Series:
    model = factory(history)
    X_train, y_train = split_xy(fold.train)
    X_val, _ = split_xy(fold.validation, daytime_only=False)
    raw = model.train(X_train, y_train).predict(X_val)
    return postprocess_predictions(raw, fold.validation[DAY_FLAG])


def cross_validate(
    candidates: dict[str, ModelFactory], df: pd.DataFrame, n_splits: int = DEFAULT_SPLITS
) -> pd.DataFrame:
    """Entrena y mide cada candidato en cada pliegue; una fila por candidato y pliegue.

    `candidates` asocia una etiqueta a cada fábrica: modelos distintos o
    configuraciones del mismo modelo que se quieren comparar, p. ej.
    `{"k=5": lambda h: KNNModel(k=5), "k=10": lambda h: KNNModel(k=10)}`.

    En cada pliegue todos se miden sobre las mismas horas: las de luz con
    potencia observada que todos predicen (`n_hours`). Así sus medias son
    comparables, como en score.py. Las persistencias reciben como historia
    solo el periodo de ajuste: ven el día anterior a cada día validado, que en
    un pronóstico real ya se conoce.
    """
    if not candidates:
        raise ValueError("No hay candidatos que validar; pasa al menos una fábrica de modelo.")
    history = tuning_data(df)
    rows = []
    for fold in sequential_folds(history, n_splits):
        predictions = {label: _predict_fold(factory, history, fold) for label, factory in candidates.items()}
        validation = fold.validation
        hours = validation.index[(validation[DAY_FLAG] == 1) & validation[TARGET].notna()]
        for pred in predictions.values():
            hours = hours.intersection(pred.dropna().index)
        if hours.empty:
            raise ValueError(
                f"El pliegue {fold.number} no tiene horas de luz que predigan todos los candidatos "
                f"({sorted(candidates)}); revisa que cada modelo prediga las horas de X_val."
            )
        observed = validation.loc[hours]
        train_days, val_days = (day_of(pd.DatetimeIndex(part.index)) for part in (fold.train, validation))
        for label, pred in predictions.items():
            rows.append(
                {
                    "candidate": label,
                    "fold": fold.number,
                    "train_start": f"{train_days.min():%Y-%m-%d}",
                    "train_end": f"{train_days.max():%Y-%m-%d}",
                    "validation_start": f"{val_days.min():%Y-%m-%d}",
                    "validation_end": f"{val_days.max():%Y-%m-%d}",
                    **evaluate(observed[TARGET], pred.loc[hours], observed[DAY_FLAG]),
                    "n_hours": len(hours),
                }
            )
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    """Media de cada métrica por candidato sobre los pliegues, ordenada por RMSE."""
    return folds.groupby("candidate")[list(METRICS)].mean().sort_values("rmse")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset (por defecto data/sample.csv)")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_SPLITS, help="número de pliegues")
    args = parser.parse_args()

    df = load_dataset(args.data)
    history = tuning_data(df)
    candidates = {factory(history).name: factory for factory in MODEL_FACTORIES}
    folds = cross_validate(candidates, df, args.n_splits)
    print(folds.to_string(index=False, float_format=_four_decimals))
    print("\nMedia por candidato:")
    print(summarize(folds).to_string(float_format=_four_decimals))


def _four_decimals(value: float) -> str:
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
