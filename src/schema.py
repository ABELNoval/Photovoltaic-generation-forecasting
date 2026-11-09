"""Nombres de columnas del dataset compartido (docs/planning.md, sección 5).

Es la única fuente de los nombres de columnas: si el esquema cambia al
confirmarse con la muestra mínima, se actualiza aquí y en planning.md.
"""

TIMESTAMP = "timestamp_utc"  # final de la hora: la fila de las 12:00 resume 11:00–12:00
TARGET = "power_norm"
DAY_FLAG = "is_day"  # 1 si la hora tiene radiación (GHI > 0) en Open-Meteo
GHI = "ghi"  # radiación de onda corta; la usa la persistencia reescalada

# Obligatorias para todos los modelos.
CORE_FEATURES = [
    GHI,
    "cloud_total",
    "cloud_low",
    "cloud_mid",
    "cloud_high",
    "temperature",
]
CALENDAR_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
REQUIRED_FEATURES = CORE_FEATURES + CALENDAR_FEATURES

# Opcionales: cada modelo decide si las usa, eligiendo solo con la
# validación de 2014–2015, y lo documenta en su texto final.
SECONDARY_FEATURES = ["humidity", "wind", "pressure", "precipitation"]

INPUT_FEATURES = REQUIRED_FEATURES + SECONDARY_FEATURES
