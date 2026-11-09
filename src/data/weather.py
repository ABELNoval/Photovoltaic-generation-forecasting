"""Descarga de la meteorología horaria de Open-Meteo (issue #4).

Usa el archivo histórico (https://open-meteo.com/en/docs/historical-weather-api):
gratuito, sin clave y en UTC. Las columnas se renombran a las del esquema
(`src/schema.py`).

Convenio de tiempo: cada fila lleva la hora en su **final**. La radiación es
la media de la hora anterior y la precipitación su suma; el resto de
variables son el valor instantáneo en esa hora. Coincide con la potencia
horaria de `pv_power.to_hourly`.

Uso, desde la raíz del repositorio:

    python -m src.data.weather ID107
    python -m src.data.weather --lat 52.09 --lon 5.05 --name prueba
"""

import argparse

import pandas as pd
import requests

from src.data.metadata import load_metadata
from src.data.paths import RAW_DIR

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variable de Open-Meteo -> columna del esquema.
VARIABLES = {
    "shortwave_radiation": "ghi",
    "cloud_cover": "cloud_total",
    "cloud_cover_low": "cloud_low",
    "cloud_cover_mid": "cloud_mid",
    "cloud_cover_high": "cloud_high",
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind",
    "surface_pressure": "pressure",
    "precipitation": "precipitation",
}

DEFAULT_START = "2014-01-01"
DEFAULT_END = "2017-12-31"


def _fetch_period(latitude: float, longitude: float, start: str, end: str) -> dict:
    """Una petición a la API. Devuelve la respuesta JSON completa."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=120)
    response.raise_for_status()
    try:
        return response.json()
    except requests.JSONDecodeError:
        # La API a veces responde 200 con un texto de error (p. ej. un timeout).
        raise RuntimeError(f"Open-Meteo no devolvió datos ({start} a {end}): {response.text[:200]}")


def fetch_weather(latitude: float, longitude: float, start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.DataFrame:
    """Descarga la meteorología horaria entre `start` y `end` (incluidos).

    Se pide un año por petición: varios años de golpe superan el tiempo
    límite del servidor.

    Unidades: ghi en W/m², nubosidad y humedad en %, temperatura en °C,
    viento en m/s, presión en hPa y precipitación en mm.
    """
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    parts = []
    for year in range(first.year, last.year + 1):
        period_start = max(first, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(last, pd.Timestamp(year=year, month=12, day=31))
        payload = _fetch_period(latitude, longitude, f"{period_start:%Y-%m-%d}", f"{period_end:%Y-%m-%d}")
        hourly = payload["hourly"]
        part = pd.DataFrame({VARIABLES[name]: hourly[name] for name in VARIABLES})
        part.index = pd.to_datetime(hourly["time"]).tz_localize("UTC")
        parts.append(part)

    weather = pd.concat(parts)
    weather.index.name = "timestamp_utc"
    # Punto de la rejilla de Open-Meteo que se usó realmente (no el pedido).
    weather.attrs["grid_latitude"] = payload["latitude"]
    weather.attrs["grid_longitude"] = payload["longitude"]
    return weather


def _cmd(args) -> None:
    if args.system_id:
        system = load_metadata().loc[args.system_id]
        latitude, longitude, name = system["latitude"], system["longitude"], args.system_id
    elif args.lat is not None and args.lon is not None:
        latitude, longitude, name = args.lat, args.lon, args.name
    else:
        raise SystemExit("Indica un ID de sistema o --lat y --lon.")

    weather = fetch_weather(latitude, longitude, args.start, args.end)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"weather_{name}.csv"
    weather.to_csv(out)
    print(f"Pedido: {latitude:.4f}, {longitude:.4f} | rejilla usada: "
          f"{weather.attrs['grid_latitude']:.4f}, {weather.attrs['grid_longitude']:.4f}")
    print(f"{len(weather)} horas ({weather.index.min()} a {weather.index.max()}), "
          f"{int(weather.isna().sum().sum())} valores sin dato")
    print(f"Guardado en {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system_id", nargs="?", help="ID del sistema; usa sus coordenadas de metadata.csv")
    parser.add_argument("--lat", type=float, help="latitud, si no se da un ID")
    parser.add_argument("--lon", type=float, help="longitud, si no se da un ID")
    parser.add_argument("--name", default="custom", help="sufijo del archivo cuando se usan --lat/--lon")
    parser.add_argument("--start", default=DEFAULT_START, help="primer día (por defecto 2014-01-01)")
    parser.add_argument("--end", default=DEFAULT_END, help="último día, incluido (por defecto 2017-12-31)")
    _cmd(parser.parse_args())


if __name__ == "__main__":
    main()
