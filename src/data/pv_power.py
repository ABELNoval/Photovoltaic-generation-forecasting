"""Selección del sistema y extracción de su potencia (issue #3).

El archivo de potencia (~3 GB) es una tabla ancha: una columna `DateTime`
(UTC, cada minuto) y una columna por sistema (`ID001` … `ID175`) en W. Se
lee por bloques y solo con las columnas necesarias, sin cargarlo entero.

Uso, desde la raíz del repositorio:

    python -m src.data.pv_power availability
    python -m src.data.pv_power extract ID003
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data.metadata import STUDY_END, STUDY_START, full_period_systems, load_metadata
from src.data.paths import PROCESSED_DIR, PV_POWER_CSV

CHUNK_ROWS = 500_000

# Franja que es de día todo el año en Utrecht (UTC): de 09:00 a 15:00.
DAYTIME_HOURS = range(9, 15)

MIN_MINUTES_PER_HOUR = 45


def _read_chunks(path: Path, columns: list[str]):
    """Itera el CSV de potencia por bloques, con `DateTime` como índice UTC."""
    for chunk in pd.read_csv(path, usecols=["DateTime", *columns], chunksize=CHUNK_ROWS):
        chunk.index = pd.to_datetime(chunk.pop("DateTime"), utc=True)
        yield chunk


def daytime_availability(
    system_ids: list[str],
    path: Path = PV_POWER_CSV,
    start: pd.Timestamp = STUDY_START,
    end: pd.Timestamp = STUDY_END,
) -> pd.DataFrame:
    """Fracción de minutos diurnos con dato para cada sistema en [start, end].

    Devuelve una columna `availability` con el total del periodo y una por
    año. Se compara con todos los minutos diurnos del periodo, de modo que
    las filas que faltan en el archivo cuentan como no disponibles.
    """
    end_of_period = end + pd.Timedelta(days=1)
    counts = []
    for chunk in _read_chunks(path, system_ids):
        chunk = chunk[(chunk.index >= start) & (chunk.index < end_of_period)]
        index = pd.DatetimeIndex(chunk.index)
        chunk = chunk[index.hour.isin(DAYTIME_HOURS)]
        counts.append(chunk.notna().groupby(pd.DatetimeIndex(chunk.index).year).sum())
    counts = pd.concat(counts).groupby(level=0).sum()

    days = pd.Series(pd.date_range(start.normalize(), end_of_period, freq="D", inclusive="left"))
    expected = days.groupby(days.dt.year).size() * len(DAYTIME_HOURS) * 60

    by_year = counts.div(expected, axis=0).T
    by_year.columns = [str(year) for year in by_year.columns]
    by_year.insert(0, "availability", counts.sum() / expected.sum())
    return by_year


def rank_candidates(meta: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    """Ordena los sistemas de mayor a menor disponibilidad diurna.

    La propuesta pedía al menos un 95 %, pero ningún sistema lo alcanza en
    2014–2017: entre septiembre de 2015 y marzo de 2016 faltan mediciones en
    casi todos los sistemas a la vez. Por eso se elige el de mayor
    disponibilidad. Se descartan los sistemas sin coordenadas, porque sin
    ellas no se puede descargar su meteorología.
    """
    table = meta.join(availability, how="inner").dropna(subset=["latitude", "longitude"])
    columns = ["estimated_ac_capacity", "tilt", "azimuth", *availability.columns]
    return table.sort_values("availability", ascending=False)[columns]


def extract_system(system_id: str, path: Path = PV_POWER_CSV) -> pd.Series:
    """Potencia de un sistema a 1 minuto, en W."""
    return pd.concat(chunk[system_id] for chunk in _read_chunks(path, [system_id]))


def to_hourly(power: pd.Series, min_minutes: int = MIN_MINUTES_PER_HOUR) -> pd.Series:
    """Media horaria, etiquetada con el **final** de la hora.

    La fila de las 12:00 es la media de 11:01–12:00, igual que la radiación
    de Open-Meteo, que es la media de la hora anterior. Las horas con menos
    de `min_minutes` minutos con dato quedan vacías.
    """
    resampler = power.resample("h", label="right", closed="right")
    hourly = resampler.mean()
    return hourly.where(resampler.count() >= min_minutes).rename("power_w")


def _cmd_availability(args) -> None:
    meta = load_metadata()
    candidates = full_period_systems(meta)
    print(f"{len(candidates)} de {len(meta)} sistemas cubren 2014-2017. Calculando disponibilidad...")

    start = pd.Timestamp(args.start, tz="UTC") if args.start else STUDY_START
    end = pd.Timestamp(args.end, tz="UTC") if args.end else STUDY_END
    availability = daytime_availability(list(candidates.index), args.input, start, end)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table = candidates.join(availability)
    table.to_csv(PROCESSED_DIR / "availability.csv")

    ranking = rank_candidates(candidates, availability)
    print("\nSistemas con mayor disponibilidad diurna (total y por año):")
    print(ranking.head(10).to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nTabla completa: {PROCESSED_DIR / 'availability.csv'}")


def _cmd_extract(args) -> None:
    hourly = to_hourly(extract_system(args.system_id, args.input))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"power_{args.system_id}_hourly.csv"
    hourly.rename_axis("timestamp_utc").to_csv(out)
    print(f"{args.system_id}: {hourly.notna().sum()} horas con dato de {len(hourly)} "
        f"({hourly.index.min()} a {hourly.index.max()})")
    print(f"Guardado en {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=PV_POWER_CSV, help="CSV de potencia (por defecto, el filtrado ac)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_avail = sub.add_parser("availability", help="disponibilidad diurna de los candidatos y ranking")
    p_avail.add_argument("--start", help="inicio del periodo (por defecto 2014-01-01)")
    p_avail.add_argument("--end", help="fin del periodo, incluido (por defecto 2017-12-31)")
    p_avail.set_defaults(func=_cmd_availability)

    p_extract = sub.add_parser("extract", help="extrae la potencia horaria de un sistema")
    p_extract.add_argument("system_id", help="ID del sistema, p. ej. ID003")
    p_extract.set_defaults(func=_cmd_extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
