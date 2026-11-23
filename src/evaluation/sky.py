"""Tipo de cielo de cada hora y métricas por tipo (docs/planning.md,
sección 4, Miembro 2, actividad 6).

Cada hora se clasifica con su nubosidad total (`cloud_total`, en %), con los
umbrales de la escala en octas de la OMM:

- despejado (`clear`): hasta 2 octas, `cloud_total <= 25`;
- parcial (`partly_cloudy`): de 3 a 5 octas, `25 < cloud_total < 75`;
- cubierto (`overcast`): 6 octas o más, `cloud_total >= 75`.

Se clasifica por hora y no por día porque las métricas son horarias: un día
de mañana despejada y tarde cubierta aporta horas a los dos grupos. La regla
es fija y no depende de los datos, así que da lo mismo en cualquier periodo y
no filtra información de prueba.

Para comparar grupos entre sí conviene mirar MAE y RMSE: el R² se mide contra
la varianza de la potencia en cada grupo, que no es la misma, así que un
mismo error da R² distintos según el grupo. Cuando no está definido (menos
de dos horas o potencia constante en el grupo) se deja vacío en lugar del
0 o 1 que devolvería scikit-learn.
"""

import pandas as pd

from src.evaluation.metrics import METRICS, evaluate
from src.schema import CLOUD_TOTAL, DAY_FLAG, TARGET

CLEAR_MAX = 25.0
OVERCAST_MIN = 75.0
SKY_TYPES = ("clear", "partly_cloudy", "overcast")


def classify_sky(cloud_total: pd.Series) -> pd.Series:
    """Tipo de cielo de cada hora; falla si falta la nubosidad o sale de [0, 100]."""
    if cloud_total.isna().any():
        raise ValueError(
            f"Faltan {int(cloud_total.isna().sum())} valores de {CLOUD_TOTAL}; el dataset "
            "limpio no debería tenerlos. Revisa la construcción del dataset."
        )
    if ((cloud_total < 0) | (cloud_total > 100)).any():
        raise ValueError(f"{CLOUD_TOTAL} debe estar en [0, 100] (porcentaje); revisa las unidades del dataset.")
    sky = pd.Series(SKY_TYPES[1], index=cloud_total.index, name="sky_type")
    sky[cloud_total <= CLEAR_MAX] = SKY_TYPES[0]
    sky[cloud_total >= OVERCAST_MIN] = SKY_TYPES[2]
    return sky


def compare_by_sky(predictions: dict[str, pd.Series], observed: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Métricas de cada modelo por tipo de cielo, en las horas `hours`.

    `hours` son las horas de luz comunes de score.common_hours, para que las
    cifras sumen lo mismo que la tabla general. Los tipos sin horas no salen.
    """
    sky = classify_sky(observed.loc[hours, CLOUD_TOTAL])
    rows = []
    for sky_type in SKY_TYPES:
        selected = sky.index[sky == sky_type]
        if selected.empty:
            continue
        y_true, is_day = observed.loc[selected, TARGET], observed.loc[selected, DAY_FLAG]
        r2_defined = len(selected) >= 2 and y_true.nunique() > 1
        for name, pred in predictions.items():
            metrics = evaluate(y_true, pred.reindex(selected), is_day)
            if not r2_defined:
                metrics["r2"] = float("nan")
            rows.append({"sky_type": sky_type, "model": name, **metrics, "n_hours": len(selected)})
    table = pd.DataFrame(rows, columns=["sky_type", "model", *METRICS, "n_hours"])
    # Orden de los grupos de despejado a cubierto y, dentro de cada uno, por RMSE.
    table["sky_type"] = pd.Categorical(table["sky_type"], categories=SKY_TYPES, ordered=True)
    return table.sort_values(["sky_type", "rmse"]).reset_index(drop=True)
