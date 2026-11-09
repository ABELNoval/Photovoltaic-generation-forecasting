"""Tabla comparativa a partir de las predicciones guardadas en results/.

Lee los `results/<name>_predictions.csv` (de run.py o entregados por cada
miembro) y los mide contra lo observado. Todos los modelos se miden sobre las
mismas horas: las de luz que todos predicen. Si no, la comparación no es justa.

Uso, desde la raíz del repositorio:

    python -m src.evaluation.score
    python -m src.evaluation.score --test-start 2016-01-01 --test-end 2016-12-31 --data data/processed/dataset_ID003.csv
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from src.data.dataset import load_dataset
from src.data.paths import DATA_DIR
from src.evaluation.metrics import evaluate
from src.evaluation.periods import select_period
from src.models.utils import RESULTS_DIR
from src.schema import DAY_FLAG, TARGET, TIMESTAMP

PREDICTIONS_SUFFIX = "_predictions.csv"
COMPARISON_FILE = "comparison.csv"
DEFAULT_DATA = DATA_DIR / "sample.csv"
DEFAULT_TEST = ("2014-05-01", "2014-05-31")


def load_predictions(results_dir: Path = RESULTS_DIR) -> dict[str, pd.Series]:
    """Lee las predicciones guardadas con `save_predictions`, por nombre de modelo."""
    paths = sorted(Path(results_dir).glob(f"*{PREDICTIONS_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(
            f"No hay archivos *{PREDICTIONS_SUFFIX} en {results_dir}; "
            "ejecuta antes python -m src.evaluation.run."
        )
    predictions = {}
    for path in paths:
        df = pd.read_csv(path)
        index = pd.DatetimeIndex(pd.to_datetime(df[TIMESTAMP], utc=True), name=TIMESTAMP)
        predictions[path.name.removesuffix(PREDICTIONS_SUFFIX)] = pd.Series(df["prediction"].to_numpy(), index=index)
    return predictions


def compare(predictions: dict[str, pd.Series], observed: pd.DataFrame) -> pd.DataFrame:
    """Métricas de cada modelo sobre las horas de luz comunes, ordenadas por RMSE."""
    daytime = observed.index[(observed[DAY_FLAG] == 1) & observed[TARGET].notna()]
    covered = {name: pred.dropna().index.intersection(daytime) for name, pred in predictions.items()}

    hours = daytime
    for index in covered.values():
        hours = hours.intersection(index)
    if hours.empty:
        culprits = [name for name, index in covered.items() if index.empty] or sorted(predictions)
        raise ValueError(
            f"Las predicciones de {culprits} no tienen horas de luz en común con el periodo "
            "de prueba; borra los archivos viejos de results/ o vuelve a ejecutar "
            "python -m src.evaluation.run."
        )
    if any(len(index) != len(hours) for index in covered.values()):
        detail = ", ".join(f"{name}: {len(index)}" for name, index in covered.items())
        warnings.warn(
            f"Se comparan {len(hours)} horas comunes; no todos los modelos predicen las "
            f"mismas horas de luz ({detail}). Si algún archivo es de otra ejecución, "
            "vuelve a ejecutar python -m src.evaluation.run."
        )

    y_true, is_day = observed.loc[hours, TARGET], observed.loc[hours, DAY_FLAG]
    rows = [
        {"model": name, **evaluate(y_true, pred.reindex(y_true.index), is_day), "n_hours": len(hours)}
        for name, pred in predictions.items()
    ]
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def score_results(df: pd.DataFrame, test_period: tuple[str, str], results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Compara las predicciones de `results_dir` en el periodo de prueba y guarda la tabla."""
    table = compare(load_predictions(results_dir), select_period(df, *test_period))
    table.to_csv(Path(results_dir) / COMPARISON_FILE, index=False)
    return table


def add_test_period_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset (por defecto data/sample.csv)")
    parser.add_argument("--test-start", default=DEFAULT_TEST[0], help="primer día de prueba")
    parser.add_argument("--test-end", default=DEFAULT_TEST[1], help="último día de prueba, incluido")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="carpeta de predicciones")


def print_table(table: pd.DataFrame, results_dir: Path) -> None:
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Guardado en {Path(results_dir) / COMPARISON_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_test_period_arguments(parser)
    args = parser.parse_args()
    table = score_results(load_dataset(args.data), (args.test_start, args.test_end), args.results_dir)
    print_table(table, args.results_dir)


if __name__ == "__main__":
    main()
