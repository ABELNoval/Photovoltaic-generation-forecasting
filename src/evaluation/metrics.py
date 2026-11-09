"""Métricas comunes de evaluación (docs/planning.md, sección 4, Miembro 2).

Todas se calculan sobre `power_norm`, la potencia dividida entre la capacidad
AC del sistema, y solo en las horas de luz. Por eso MAE y RMSE quedan en
fracción de la capacidad, y el nRMSE "normalizado por la capacidad instalada"
(docs/proposal.md) es el mismo RMSE expresado en porcentaje.
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Error absoluto medio, en fracción de la capacidad."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Raíz del error cuadrático medio, en fracción de la capacidad."""
    return float(root_mean_squared_error(y_true, y_pred))


def nrmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """RMSE en porcentaje de la capacidad instalada."""
    return 100.0 * rmse(y_true, y_pred)


def r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Coeficiente de determinación."""
    return float(r2_score(y_true, y_pred))


# Orden fijo: es el de las columnas de la tabla comparativa.
METRICS = {"mae": mae, "rmse": rmse, "nrmse": nrmse, "r2": r2}


def evaluate(y_true: pd.Series, y_pred: pd.Series, is_day: pd.Series) -> dict[str, float]:
    """Calcula todas las métricas en las horas de luz (`is_day == 1`).

    Es el único sitio donde se filtra la noche, para que todos los modelos se
    midan igual. Las tres Series deben compartir índice; las horas de luz no
    pueden tener valores faltantes.
    """
    if not (y_true.index.equals(y_pred.index) and y_true.index.equals(is_day.index)):
        raise ValueError(
            "y_true, y_pred e is_day deben tener el mismo índice; "
            "alinéalos con .reindex(y_true.index) antes de evaluar."
        )
    daytime = is_day == 1
    true, pred = y_true[daytime], y_pred[daytime]
    if true.empty:
        raise ValueError("No hay horas de luz que evaluar; revisa el periodo y la columna is_day.")
    missing = int(true.isna().sum() + pred.isna().sum())
    if missing:
        raise ValueError(
            f"Hay {missing} valores faltantes en horas de luz; quita esas horas en "
            "todos los modelos por igual (como hace score.compare) antes de evaluar."
        )
    return {name: metric(true, pred) for name, metric in METRICS.items()}
