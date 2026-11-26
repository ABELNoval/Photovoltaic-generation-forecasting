# Pronóstico de generación fotovoltaica

Proyecto de la asignatura Machine Learning. Se busca construir un
modelo que, a partir de variables meteorológicas, pronostique la
generación de una planta fotovoltaica.

## Integrantes
- Abel de la Noval Pérez C-411.
- Frank Alberto Piz Torriente C-411.
- Juan Miguel Maestre Rodríguez C-412.
- Daniel Amaranto Mares García C-412.
- Bruno Jesús Pire Ricardo C-412.
- Dayan Cabrera Corvo C-412.

## Estado del proyecto
Fase 1: investigación y estado del arte.

## Estructura
- `docs/`: estado del arte e informes
- `data/`: datos brutos y procesados; solo se versiona el dataset compartido
  `data/processed/dataset_ID003.csv` como excepción acordada en el issue #41.
- `notebooks/`: experimentos
- `src/`: código fuente

El dataset ID003 se regenera desde la raíz con `python -m src.data.build_dataset ID003`
cuando están disponibles los archivos de entrada. Combina y procesa las mediciones de
potencia de Utrecht (Visser et al., 2022, [Zenodo](https://doi.org/10.5281/zenodo.6906504))
y el histórico de [Open-Meteo](https://open-meteo.com/en/licence). Ambas fuentes se
distribuyen bajo CC BY 4.0. Los archivos de entrada y los demás datos generados no se
versionan.

## Entorno de trabajo
Requiere Python 3.11 o superior. Desde la raíz del repositorio:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

La carpeta `.venv/` no se sube al repositorio. Si se añade una
dependencia, se fija su versión en `requirements.txt` y se avisa al
equipo.

## Objetivo
Predecir la generación del día siguiente con resolución horaria
(sujeto a confirmación).
