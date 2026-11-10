"""Flujo completo: entrena los modelos del registro, guarda sus predicciones
y genera la tabla comparativa.

Entrena con las horas de luz del periodo de entrenamiento, predice las 24
horas del periodo de prueba, ajusta las predicciones igual para todos
(noche a 0, rango [0, 1]) y las guarda en results/<name>_predictions.csv.
Al final llama a score.py con el mismo periodo de prueba.

Uso, desde la raíz del repositorio:

    python -m src.evaluation.run
    python -m src.evaluation.run --data data/processed/dataset_ID003.csv \\
        --train-start 2014-01-01 --train-end 2015-12-31 --test-start 2016-01-01 --test-end 2016-12-31
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data.dataset import load_dataset
from src.evaluation.periods import select_period
from src.evaluation.registry import MODEL_FACTORIES, ModelFactory
from src.evaluation.score import add_test_period_arguments, print_table, score_results
from src.models.utils import (
    RESULTS_DIR,
    postprocess_predictions,
    save_predictions,
    split_xy,
)
from src.schema import DAY_FLAG

DEFAULT_TRAIN = ("2014-03-01", "2014-04-30")


def _period(df: pd.DataFrame, start: str, end: str, label: str) -> pd.DataFrame:
    selected = select_period(df, start, end)
    if selected.empty:
        raise ValueError(
            f"El periodo de {label} {start}..{end} no tiene datos; elige fechas dentro de "
            f"{df.index.min():%Y-%m-%d}..{df.index.max():%Y-%m-%d}."
        )
    return selected


def run_models(
    df: pd.DataFrame,
    train_period: tuple[str, str],
    test_period: tuple[str, str],
    factories: list[ModelFactory] = MODEL_FACTORIES,
    results_dir: Path = RESULTS_DIR,
) -> list[Path]:
    """Entrena, predice y guarda cada modelo. Devuelve los archivos escritos."""
    train_df = _period(df, *train_period, "entrenamiento")
    test_df = _period(df, *test_period, "prueba")
    X_train, y_train = split_xy(train_df)
    X_test, _ = split_xy(test_df, daytime_only=False)

    paths = []
    for factory in factories:
        model = factory(df)
        raw = model.train(X_train, y_train).predict(X_test)
        predictions = postprocess_predictions(raw, test_df[DAY_FLAG])
        paths.append(save_predictions(predictions, model.name, results_dir))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-start", default=DEFAULT_TRAIN[0], help="primer día de entrenamiento")
    parser.add_argument("--train-end", default=DEFAULT_TRAIN[1], help="último día de entrenamiento, incluido")
    add_test_period_arguments(parser)
    args = parser.parse_args()

    df = load_dataset(args.data)
    test_period = (args.test_start, args.test_end)
    for path in run_models(df, (args.train_start, args.train_end), test_period, results_dir=args.results_dir):
        print(f"Predicciones: {path}")
    print_table(score_results(df, test_period, args.results_dir), args.results_dir)


if __name__ == "__main__":
    main()
