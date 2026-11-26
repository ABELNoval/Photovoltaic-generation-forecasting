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
    sources = load_sources(system_id)
    df = add_calendar(clean(sources, ac_capacity))

    # Cada fila se etiqueta con el final de su hora: el día D va de D 01:00 a D+1 00:00.
    first = (pd.Timestamp(start, tz="UTC") if start else STUDY_START) + pd.Timedelta(hours=1)
    last = (pd.Timestamp(end, tz="UTC") if end else STUDY_END) + pd.Timedelta(days=1)
    df = df.loc[first:last]
    sources = sources.loc[first:last]

    days = (pd.DatetimeIndex(df.index) - pd.Timedelta(hours=1)).normalize()
    bad_days = df[TARGET].isna().groupby(days).any()
    bad_day_index = pd.DatetimeIndex(bad_days.index)
    bad_day_dates = bad_day_index[bad_days.to_numpy()]
    discarded_day_rows = days.isin(bad_day_dates)
    kept = df[~discarded_day_rows]
    result = kept.reset_index(names=TIMESTAMP)[COLUMNS]

    interval_years = (pd.DatetimeIndex(df.index) - pd.Timedelta(hours=1)).year
    kept_years = (pd.DatetimeIndex(kept.index) - pd.Timedelta(hours=1)).year
    result_years = (
        pd.DatetimeIndex(pd.to_datetime(result[TIMESTAMP], utc=True))
        - pd.Timedelta(hours=1)
    ).year
    weather_columns = [column for column in sources.columns if column != "power_w"]
    period_start = first - pd.Timedelta(hours=1)
    years = range(period_start.year, (last - pd.Timedelta(hours=1)).year + 1)
    annual = []
    for year in years:
        calendar_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        calendar_end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        interval_start = max(period_start, calendar_start)
        interval_end = min(last, calendar_end)
        expected = pd.date_range(
            interval_start + pd.Timedelta(hours=1),
            interval_end,
            freq="h",
        )
        raw = sources.reindex(expected)
        row_mask = interval_years == year
        kept_mask = kept_years == year
        discarded_mask = row_mask & discarded_day_rows
        day_mask = bad_day_index.year == year
        annual.append(
            {
                "year": year,
                "hours_expected": len(expected),
                "power_hours_with_data": int(raw["power_w"].notna().sum()),
                "power_missing_hours": int(raw["power_w"].isna().sum()),
                "power_coverage_fraction": (
                    float(raw["power_w"].notna().sum() / len(expected))
                    if len(expected)
                    else 0.0
                ),
                "weather_missing_values": int(
                    raw[weather_columns].isna().sum().sum()
                ),
                "negative_values": int(df.loc[row_mask, "_negative"].sum()),
                "night_hours_set_to_zero": int(
                    df.loc[row_mask, "_night_zero"].sum()
                ),
                "hours_interpolated": int(
                    kept.loc[kept_mask, "_interpolated"].sum()
                ),
                "hours_interpolated_on_discarded_days": int(
                    df.loc[discarded_mask, "_interpolated"].sum()
                ),
                "unresolved_power_hours": int(
                    df.loc[row_mask, TARGET].isna().sum()
                ),
                "days_kept": int((~bad_days.loc[day_mask]).sum()),
                "days_dropped": int(bad_days.loc[day_mask].sum()),
                "rows": int((result_years == year).sum()),
                "daytime_rows": int(
                    result.loc[result_years == year, DAY_FLAG].sum()
                ),
                "missing_values": int(
                    result.loc[result_years == year].isna().sum().sum()
                ),
            }
        )

    hours_expected = sum(row["hours_expected"] for row in annual)
    power_hours_with_data = sum(row["power_hours_with_data"] for row in annual)
    interpolated_on_discarded_days = sum(
        row["hours_interpolated_on_discarded_days"] for row in annual
    )

    stats = {
        "negative_values": int(df["_negative"].sum()),
        "night_hours_set_to_zero": int(df["_night_zero"].sum()),
        "hours_interpolated": int(kept["_interpolated"].sum()),
        "hours_interpolated_before_day_discard": int(df["_interpolated"].sum()),
        "hours_interpolated_on_discarded_days": interpolated_on_discarded_days,
        "unresolved_power_hours": int(df[TARGET].isna().sum()),
        "days_dropped": int(bad_days.sum()),
        "rows": len(kept),
        "daytime_rows": int(kept[DAY_FLAG].sum()),
        "days": len(bad_days) - int(bad_days.sum()),
        "hours_expected": hours_expected,
        "power_hours_with_data": power_hours_with_data,
        "power_missing_hours": hours_expected - power_hours_with_data,
        "power_coverage_fraction": (
            power_hours_with_data / hours_expected if hours_expected else 0.0
        ),
        "weather_missing_values": int(
            sources[weather_columns].reindex(
                pd.date_range(
                    period_start + pd.Timedelta(hours=1),
                    last,
                    freq="h",
                )
            ).isna().sum().sum()
        ),
        "missing_values": int(result.isna().sum().sum()),
        "annual": annual,
    }
    return result, stats


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
    print(
        f"  horas interpoladas (huecos de hasta {MAX_GAP_HOURS} h): "
        f"{stats['hours_interpolated']} conservadas de "
        f"{stats['hours_interpolated_before_day_discard']} estimadas"
    )
    print(
        "  potencia faltante después de limpiar: "
        f"{stats['unresolved_power_hours']} horas; "
        f"{stats['days_dropped']} días descartados"
    )
    print(
        f"  cobertura de potencia: {stats['power_hours_with_data']} de "
        f"{stats['hours_expected']} horas ({stats['power_coverage_fraction']:.3%}); "
        f"faltantes: {stats['power_missing_hours']}"
    )
    print(
        "  valores meteorológicos faltantes en horas esperadas: "
        f"{stats['weather_missing_values']}"
    )
    print(f"  valores faltantes en el dataset final: {stats['missing_values']}")
    print("\nCobertura de potencia por año:")
    coverage = pd.DataFrame(stats["annual"]).set_index("year")
    print(
        coverage[
            [
                "hours_expected",
                "power_hours_with_data",
                "power_missing_hours",
                "power_coverage_fraction",
                "weather_missing_values",
            ]
        ]
        .rename(
            columns={
                "hours_expected": "horas_esperadas",
                "power_hours_with_data": "potencia_con_dato",
                "power_missing_hours": "potencia_faltante",
                "power_coverage_fraction": "cobertura_potencia",
                "weather_missing_values": "faltantes_meteorologia",
            }
        )
        .to_string(float_format=lambda value: f"{value:.3f}")
    )
    print("\nLimpieza y dataset final por año:")
    print(
        coverage[
            [
                "negative_values",
                "night_hours_set_to_zero",
                "hours_interpolated",
                "hours_interpolated_on_discarded_days",
                "unresolved_power_hours",
                "days_kept",
                "days_dropped",
                "rows",
                "daytime_rows",
                "missing_values",
            ]
        ]
        .rename(
            columns={
                "negative_values": "negativos_corregidos",
                "night_hours_set_to_zero": "noches_a_cero",
                "hours_interpolated": "interpoladas_conservadas",
                "hours_interpolated_on_discarded_days": "interpoladas_dias_descartados",
                "unresolved_power_hours": "potencia_aun_faltante",
                "days_kept": "dias_conservados",
                "days_dropped": "dias_descartados",
                "rows": "filas_finales",
                "daytime_rows": "filas_diurnas",
                "missing_values": "celdas_faltantes",
            }
        )
        .to_string()
    )
    print(f"Guardado en {out.relative_to(DATA_DIR.parent) if out.is_absolute() else out}")


if __name__ == "__main__":
    main()
