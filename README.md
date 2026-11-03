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
- `data/`: datasets (no se suben archivos pesados)
- `notebooks/`: experimentos
- `src/`: código fuente

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
