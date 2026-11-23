"""Tabla comparativa a partir de las predicciones guardadas en results/.

Lee los `results/<name>_predictions.csv` (de run.py o entregados por cada
miembro) y los mide contra lo observado. Todos los modelos se miden sobre las
mismas horas: las de luz que todos predicen. Si no, la comparación no es justa.

Antes de medir se rechaza lo que daría una comparación engañosa:

- archivos mal formados: sin columnas `timestamp_utc,prediction`, con fechas
  ilegibles o sin zona horaria, horas repetidas o que no caen en punto,
  predicciones no numéricas o fuera de [0, 1];
- predicciones distintas de cero en horas de noche, señal de que no pasaron
  por `postprocess_predictions`, de que usan otro convenio horario o de que
  vienen de otra versión del dataset;
- modelos que cubren menos de `MIN_COVERAGE` de las horas de luz del periodo,
  casi siempre archivos viejos de otra ejecución.

Además avisa de predicciones en horas que el dataset no tiene dentro del
periodo: los archivos generados con run.py solo predicen filas del dataset,
así que suelen venir de otra versión del dataset. Se mira el contenido y no
la fecha de modificación, que cambia con git o al copiar archivos.

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
from src.evaluation.protocol import check_test_period
from src.models.utils import RESULTS_DIR
from src.schema import DAY_FLAG, TARGET, TIMESTAMP

PREDICTIONS_SUFFIX = "_predictions.csv"
PREDICTION = "prediction"
COMPARISON_FILE = "comparison.csv"
DEFAULT_DATA = DATA_DIR / "sample.csv"
DEFAULT_TEST = ("2014-05-01", "2014-05-31")

# Fracción mínima de horas de luz del periodo que debe predecir cada modelo.
# Las persistencias pierden los días que siguen a un día sin datos; por
# debajo de este umbral lo habitual es un archivo de otra ejecución.
MIN_COVERAGE = 0.9

# Una zona horaria explícita al final: "Z", "+00:00" o "+0000".
_OFFSET = r"(?:Z|[+-]\d{2}:?\d{2})$"


def read_predictions(path: Path) -> pd.Series:
    """Lee y valida un archivo `<name>_predictions.csv`."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = [column for column in (TIMESTAMP, PREDICTION) if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: faltan las columnas {missing}; guárdalo con src.models.utils.save_predictions.")
    if df.empty:
        raise ValueError(f"{path.name}: no tiene filas; vuelve a generar las predicciones.")

    raw = df[TIMESTAMP].astype(str).str.strip()
    if not raw.str.contains(_OFFSET).all():
        raise ValueError(
            f"{path.name}: hay fechas sin zona horaria; escríbelas en UTC con offset "
            "(p. ej. 2016-05-01 12:00:00+00:00), como hace save_predictions."
        )
    index = pd.DatetimeIndex(pd.to_datetime(raw, utc=True, format="ISO8601", errors="coerce"), name=TIMESTAMP)
    if index.isna().any():
        raise ValueError(f"{path.name}: {int(index.isna().sum())} fechas no se pueden leer; usa el formato ISO 8601.")
    if not (index == index.floor("h")).all():
        raise ValueError(
            f"{path.name}: hay horas que no caen en punto; cada fila debe ir etiquetada "
            "con el final de su hora (12:00 resume 11:00–12:00)."
        )
    if index.has_duplicates:
        repeated = index[index.duplicated()].unique()
        raise ValueError(f"{path.name}: {len(repeated)} horas repetidas (la primera, {repeated[0]}); deja una fila por hora.")

    values = pd.to_numeric(df[PREDICTION], errors="coerce")
    if (values.isna() & df[PREDICTION].notna()).any():
        raise ValueError(f"{path.name}: hay predicciones que no son números.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError(
            f"{path.name}: hay predicciones fuera de [0, 1]; pásalas por "
            "src.models.utils.postprocess_predictions antes de guardarlas."
        )
    return pd.Series(values.to_numpy(), index=index, name=PREDICTION).sort_index()


def prediction_paths(results_dir: Path = RESULTS_DIR) -> list[Path]:
    paths = sorted(Path(results_dir).glob(f"*{PREDICTIONS_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(
            f"No hay archivos *{PREDICTIONS_SUFFIX} en {results_dir}; ejecuta antes python -m src.evaluation.run."
        )
    return paths


def load_predictions(results_dir: Path = RESULTS_DIR) -> dict[str, pd.Series]:
    """Lee las predicciones guardadas con `save_predictions`, por nombre de modelo."""
    return {path.name.removesuffix(PREDICTIONS_SUFFIX): read_predictions(path) for path in prediction_paths(results_dir)}


def _check_night(predictions: dict[str, pd.Series], observed: pd.DataFrame) -> None:
    night = observed.index[observed[DAY_FLAG] == 0]
    for name, pred in predictions.items():
        lit = pred.reindex(night).gt(0)
        if lit.any():
            raise ValueError(
                f"{name}: predice potencia en {int(lit.sum())} horas de noche del periodo (la primera, "
                f"{lit.idxmax()}). Puede ser de otra versión del dataset, usar otro convenio horario o no "
                "haber pasado por postprocess_predictions; vuelve a generarlo con python -m src.evaluation.run."
            )


def _warn_unknown_hours(predictions: dict[str, pd.Series], observed: pd.DataFrame) -> None:
    first, last = observed.index.min(), observed.index.max()
    unknown = {}
    for name, pred in predictions.items():
        inside = pred.index[(pred.index >= first) & (pred.index <= last)]
        extra = inside.difference(observed.index)
        if not extra.empty:
            unknown[name] = len(extra)
    if unknown:
        detail = ", ".join(f"{name}: {count}" for name, count in sorted(unknown.items()))
        warnings.warn(
            f"Hay predicciones en horas del periodo que el dataset no tiene ({detail}); pueden venir "
            "de otra versión del dataset. Si es así, vuelve a generarlas con python -m src.evaluation.run."
        )


def common_hours(
    predictions: dict[str, pd.Series], observed: pd.DataFrame, min_coverage: float = MIN_COVERAGE
) -> tuple[pd.DatetimeIndex, dict[str, float]]:
    """Horas de luz que todos predicen y la cobertura de cada modelo.

    La cobertura es la fracción de horas de luz con potencia observada que el
    modelo predice. Falla si algún modelo no llega a `min_coverage`.
    """
    _check_night(predictions, observed)
    _warn_unknown_hours(predictions, observed)
    daytime = observed.index[(observed[DAY_FLAG] == 1) & observed[TARGET].notna()]
    if daytime.empty:
        raise ValueError("El periodo de prueba no tiene horas de luz con potencia observada; elige otras fechas.")
    covered = {name: pred.dropna().index.intersection(daytime) for name, pred in predictions.items()}
    coverage = {name: len(index) / len(daytime) for name, index in covered.items()}

    short = {name: value for name, value in coverage.items() if value < min_coverage}
    if short:
        detail = ", ".join(f"{name}: {value:.0%}" for name, value in sorted(short.items()))
        raise ValueError(
            f"Predicciones que no cubren el {min_coverage:.0%} de las horas de luz del periodo ({detail}). "
            "Si son archivos viejos, bórralos de results/ o vuelve a ejecutar python -m src.evaluation.run; "
            "si el hueco es esperado, baja el umbral con --min-coverage."
        )

    hours = daytime
    for index in covered.values():
        hours = hours.intersection(index)
    if hours.empty:
        raise ValueError(
            f"Las predicciones de {sorted(predictions)} no tienen horas de luz en común con el periodo "
            "de prueba; borra los archivos viejos de results/ o vuelve a ejecutar python -m src.evaluation.run."
        )
    if any(len(index) != len(hours) for index in covered.values()):
        detail = ", ".join(f"{name}: {len(index)}" for name, index in covered.items())
        warnings.warn(
            f"Se comparan {len(hours)} horas comunes; no todos los modelos predicen las "
            f"mismas horas de luz ({detail}). Si algún archivo es de otra ejecución, "
            "vuelve a ejecutar python -m src.evaluation.run."
        )
    return hours, coverage


def _table(
    predictions: dict[str, pd.Series], observed: pd.DataFrame, hours: pd.DatetimeIndex, coverage: dict[str, float]
) -> pd.DataFrame:
    y_true, is_day = observed.loc[hours, TARGET], observed.loc[hours, DAY_FLAG]
    rows = [
        {
            "model": name,
            **evaluate(y_true, pred.reindex(y_true.index), is_day),
            "n_hours": len(hours),
            "coverage": coverage[name],
        }
        for name, pred in predictions.items()
    ]
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def compare(
    predictions: dict[str, pd.Series], observed: pd.DataFrame, min_coverage: float = MIN_COVERAGE
) -> pd.DataFrame:
    """Métricas de cada modelo sobre las horas de luz comunes, ordenadas por RMSE."""
    return _table(predictions, observed, *common_hours(predictions, observed, min_coverage))


def score_results(
    df: pd.DataFrame,
    test_period: tuple[str, str],
    results_dir: Path = RESULTS_DIR,
    min_coverage: float = MIN_COVERAGE,
    allow_final: bool = False,
) -> pd.DataFrame:
    """Compara las predicciones de `results_dir` en el periodo de prueba y guarda la tabla."""
    check_test_period(test_period, allow_final)
    predictions = load_predictions(results_dir)
    observed = select_period(df, *test_period)
    hours, coverage = common_hours(predictions, observed, min_coverage)
    table = _table(predictions, observed, hours, coverage)
    table.to_csv(Path(results_dir) / COMPARISON_FILE, index=False)
    return table


def add_test_period_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset (por defecto data/sample.csv)")
    parser.add_argument("--test-start", default=DEFAULT_TEST[0], help="primer día de prueba")
    parser.add_argument("--test-end", default=DEFAULT_TEST[1], help="último día de prueba, incluido")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="carpeta de predicciones")
    parser.add_argument(
        "--min-coverage", type=float, default=MIN_COVERAGE, help="fracción mínima de horas de luz que debe cubrir cada modelo"
    )
    parser.add_argument("--final", action="store_true", help="permite 2017; solo para la evaluación final, una vez")


def print_table(table: pd.DataFrame, results_dir: Path) -> None:
    print(table.to_string(index=False, float_format=_four_decimals))
    print(f"Guardado en {Path(results_dir) / COMPARISON_FILE}")


def _four_decimals(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_test_period_arguments(parser)
    args = parser.parse_args()
    table = score_results(
        load_dataset(args.data),
        (args.test_start, args.test_end),
        args.results_dir,
        min_coverage=args.min_coverage,
        allow_final=args.final,
    )
    print_table(table, args.results_dir)


if __name__ == "__main__":
    main()
