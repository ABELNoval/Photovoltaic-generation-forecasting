"""Rutas de datos; solo el dataset ID003 acordado se versiona (ver .gitignore)."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Conjunto de Utrecht (Visser et al., 2022): https://doi.org/10.5281/zenodo.6906504
METADATA_CSV = RAW_DIR / "metadata.csv"
PV_POWER_CSV = RAW_DIR / "filtered_pv_power_measurements_ac.csv"
