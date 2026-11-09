"""Construcción del dataset del proyecto (issue #5).

Une la potencia horaria de un sistema (`pv_power.py`) con su meteorología
(`weather.py`), la limpia y añade las variables de calendario. Sirve tanto
para la muestra mínima como para el dataset completo: solo cambia el periodo.

Pasos:
1. Unir potencia y meteorología por hora (ambas etiquetadas al final de la hora).
2. Poner a 0 los valores negativos de potencia.
3. Marcar las horas de día (`is_day`: GHI > 0) y poner a 0 la potencia de
   noche que falte, porque de noche los paneles no producen.
4. Interpolar los huecos de hasta 3 horas seguidas y descartar los días con
   horas de día sin dato después de eso.
5. Normalizar la potencia por la capacidad AC (`power_norm`).
6. Añadir la hora del día y el día del año en forma cíclica.

Uso, desde la raíz del repositorio:

    python -m src.data.build_dataset ID003 --start 2014-03-01 --end 2014-05-31 --output data/sample.csv
    python -m src.data.build_dataset ID003
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.metadata import STUDY_END, STUDY_START, load_metadata
from src.data.paths import DATA_DIR, PROCESSED_DIR, RAW_DIR
from src.schema import (
    CALENDAR_FEATURES,
    CORE_FEATURES,
    DAY_FLAG,
    SECONDARY_FEATURES,
    TARGET,
    TIMESTAMP,
)

MAX_GAP_HOURS = 3

COLUMNS = [TIMESTAMP, TARGET, *CORE_FEATURES, *SECONDARY_FEATURES, *CALENDAR_FEATURES, DAY_FLAG]


def load_sources(system_id: str) -> pd.DataFrame:
    """Une la potencia horaria (W) y la meteorología del sistema por hora."""
    power = pd.read_csv(PROCESSED_DIR / f"power_{system_id}_hourly.csv", index_col=0, parse_dates=True)
    weather = pd.read_csv(RAW_DIR / f"weather_{system_id}.csv", index_col=0, parse_dates=True)
    return weather.join(power["power_w"], how="left")


def _fill_short_gaps(series: pd.Series, max_gap: int) -> tuple[pd.Series, pd.Series]:
    """Interpola linealmente solo los huecos de hasta `max_gap` valores seguidos.

    `interpolate(limit=...)` rellenaría también el principio de los huecos
    largos, así que primero se identifican los huecos por su longitud.
    Devuelve la serie rellenada y una marca de los valores interpolados.
    """
    missing = series.isna()
    gap_id = (~missing).cumsum()
    gap_length = missing.groupby(gap_id).transform("sum")
    filled = series.interpolate(method="time", limit_area="inside")
    interpolated = missing & (gap_length <= max_gap) & filled.notna()
    return series.where(~interpolated, filled), interpolated


def clean(df: pd.DataFrame, ac_capacity: float) -> pd.DataFrame:
    """Aplica los pasos 2 a 5 sin descartar todavía ningún día.

    Añade columnas `_negative`, `_night_zero` e `_interpolated` que marcan
    qué se hizo en cada fila, para el resumen de limpieza.
    """
    df = df.copy()
    power = df["power_w"]
    df["_negative"] = power < 0
    power = power.clip(lower=0)

    df[DAY_FLAG] = (df["ghi"] > 0).astype(int)
    df["_night_zero"] = power.isna() & (df[DAY_FLAG] == 0)
    power = power.mask(df["_night_zero"], 0.0)

    power, df["_interpolated"] = _fill_short_gaps(power, MAX_GAP_HOURS)
    df[TARGET] = power / ac_capacity
    return df.drop(columns="power_w")


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Hora del día y día del año como puntos de un círculo (seno y coseno),
    para que las 23:00 queden cerca de las 00:00 y diciembre cerca de enero."""
    index = pd.DatetimeIndex(df.index)
    hour_angle = 2 * np.pi * index.hour / 24
    doy_angle = 2 * np.pi * (index.day_of_year - 1) / 365.25
    return df.assign(
        hour_sin=np.sin(hour_angle),
        hour_cos=np.cos(hour_angle),
        doy_sin=np.sin(doy_angle),
        doy_cos=np.cos(doy_angle),
    )


def build_dataset(system_id: str, start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Dataset limpio del sistema, recortado a [start, end] (fechas incluidas).

    La limpieza se hace sobre todo el periodo disponible y luego se recorta,
    para que los huecos del borde puedan interpolarse con sus vecinos.
    """
    capacities = load_metadata()["estimated_ac_capacity"].to_dict()
    ac_capacity = float(capacities[system_id])
    df = add_calendar(clean(load_sources(system_id), ac_capacity))

    # Cada fila se etiqueta con el final de su hora: el día D va de D 01:00 a D+1 00:00.
    first = (pd.Timestamp(start, tz="UTC") if start else STUDY_START) + pd.Timedelta(hours=1)
    last = (pd.Timestamp(end, tz="UTC") if end else STUDY_END) + pd.Timedelta(days=1)
    df = df.loc[first:last]

    days = (pd.DatetimeIndex(df.index) - pd.Timedelta(hours=1)).normalize()
    bad_days = df[TARGET].isna().groupby(days).any()
    kept = df[~days.isin(bad_days[bad_days].index)]

    stats = {
        "negative_values": int(df["_negative"].sum()),
        "night_hours_set_to_zero": int(df["_night_zero"].sum()),
        "hours_interpolated": int(kept["_interpolated"].sum()),
        "days_dropped": int(bad_days.sum()),
        "rows": len(kept),
        "daytime_rows": int(kept[DAY_FLAG].sum()),
        "days": len(bad_days) - int(bad_days.sum()),
    }
    return kept.reset_index(names=TIMESTAMP)[COLUMNS], stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system_id", help="ID del sistema, p. ej. ID003")
    parser.add_argument("--start", help="primer día (por defecto 2014-01-01)")
    parser.add_argument("--end", help="último día, incluido (por defecto 2017-12-31)")
    parser.add_argument("--output", type=Path, help="archivo de salida (por defecto data/processed/dataset_<ID>.csv)")
    args = parser.parse_args()

    df, stats = build_dataset(args.system_id, args.start, args.end)
    out = args.output or PROCESSED_DIR / f"dataset_{args.system_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"{args.system_id}: {df[TIMESTAMP].min()} a {df[TIMESTAMP].max()}")
    print(f"  filas: {stats['rows']} ({stats['days']} días), de día: {stats['daytime_rows']}")
    print(f"  valores negativos puestos a 0: {stats['negative_values']}")
    print(f"  horas de noche sin dato puestas a 0: {stats['night_hours_set_to_zero']}")
    print(f"  horas interpoladas (huecos de hasta {MAX_GAP_HOURS} h): {stats['hours_interpolated']}")
    print(f"  días descartados por huecos largos: {stats['days_dropped']}")
    print(f"Guardado en {out.relative_to(DATA_DIR.parent) if out.is_absolute() else out}")


if __name__ == "__main__":
    main()
