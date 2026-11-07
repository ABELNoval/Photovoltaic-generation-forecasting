# Modelos

Cada algoritmo vive en su propio archivo (`mlp.py`, `svr.py`,
`random_forest.py`, `knn.py`) e implementa la interfaz común
`ModelInterface` de [`base.py`](base.py). Así la evaluación puede entrenar,
comparar y combinar todos los modelos sin conocer su código.

## Cómo crear un modelo

[`example.py`](example.py) es la referencia de la **estructura** que deben
tener todos los modelos. Lo que hay dentro de sus métodos (predecir la
media de cada hora) es solo un ejemplo y se reemplaza por completo con el
algoritmo de cada uno.

**Lo que todo modelo debe tener, igual que en el ejemplo:**

- una clase que hereda de `ModelInterface`;
- un atributo `name` con un identificador corto (`"mlp"`, `"svr"`, ...),
  que se usa para nombrar el archivo de resultados;
- `train(X, y)`, que ajusta el modelo y termina con `return self`;
- `predict(X)`, que devuelve una `Series` con el mismo índice que `X`.

**Lo que cada uno pone por su cuenta dentro de esa estructura:**

- en `__init__`, los hiperparámetros y la semilla (`random_state`);
- en `train`, la elección de columnas, el escalado (ajustado solo con los
  datos recibidos) y el entrenamiento del algoritmo de scikit-learn;
- en `predict`, aplicar las mismas columnas y el mismo escalado y predecir.

```python
from src.models.base import ModelInterface
from src.schema import REQUIRED_FEATURES

class SVRModel(ModelInterface):
    name = "svr"

    def __init__(self, C=10.0, epsilon=0.01):
        ...  # guardar hiperparámetros

    def train(self, X, y):
        ...  # elegir columnas, escalar y ajustar el SVR
        return self

    def predict(self, X):
        ...  # mismas columnas y escalado; Series con el mismo índice que X
```

## Reglas

- **Columnas.** `X` trae todas las columnas de entrada
  ([`src/schema.py`](../schema.py)). Las obligatorias
  (`REQUIRED_FEATURES`: GHI, nubosidad, temperatura y calendario) se usan
  siempre; las secundarias (`SECONDARY_FEATURES`: humedad, viento, presión
  y precipitación) son opcionales y se eligen solo con la validación de
  2014–2015. Las columnas usadas se documentan en el texto final.
- **Noche.** Se entrena solo con horas de luz (`is_day == 1`, es decir,
  GHI > 0). `split_xy` hace ese filtrado.
- **Predicciones.** `predict` no recorta ni pone a cero la noche: lo hace
  `postprocess_predictions` para todos los modelos por igual.
- **Resultados.** Se guardan con `save_predictions` en
  `results/<name>_predictions.csv` (`timestamp_utc`, `prediction`).

## Funciones comunes (`utils.py`)

[`utils.py`](utils.py) reúne lo que es igual para todos los modelos, para
que nadie tenga que reescribirlo ni lo haga de forma distinta:

| Función | Qué hace |
|---|---|
| `split_xy(df)` | Separa el dataset en `X` (columnas de entrada) e `y` (`power_norm`). Por defecto deja solo las horas de luz; con `daytime_only=False` deja las 24 horas, que es lo que se usa para predecir. |
| `postprocess_predictions(pred, is_day)` | Pone en 0 las horas de noche y recorta el resto al rango [0, 1]. Se aplica a las predicciones de todos los modelos por igual. |
| `save_predictions(pred, name)` | Escribe `results/<name>_predictions.csv` con las columnas `timestamp_utc` y `prediction`, el formato que usa la evaluación. |
| `save_model(model, path)` / `load_model(path)` | Guardan un modelo entrenado en un archivo y lo vuelven a cargar (ver abajo). |

### Guardar y cargar un modelo

Un modelo entrenado es un objeto de Python que vive en memoria: existe
mientras el programa está en ejecución y desaparece al terminar el script
o reiniciar el notebook. Para usarlo de nuevo habría que volver a
entrenarlo, lo que en algunos modelos lleva minutos.

`save_model` evita eso: convierte el objeto en un archivo (por ejemplo
`svr.joblib`) con todo lo que el modelo aprendió y necesita para predecir:
los árboles de un Random Forest, los vectores de soporte de un SVR, los
datos de un kNN, y en todos los casos el escalado, las columnas elegidas y
los hiperparámetros. `load_model` lee ese archivo y reconstruye el mismo
objeto, listo para llamar a `predict` sin entrenar otra vez.

Se usa la librería `joblib`, la que recomienda scikit-learn para guardar
modelos. En el proyecto sirve, por ejemplo, para que `forecast.py` cargue
el mejor modelo ya entrenado y para que la evaluación use los modelos de
cada miembro sin repetir su entrenamiento.

Dos precauciones:

- Un modelo se carga con las **mismas versiones de las librerías** con las
  que se guardó; por eso están fijadas en `requirements.txt`.
- Solo se cargan archivos **propios o del equipo**: cargar un archivo de
  este tipo puede ejecutar código.

## Flujo típico

```python
from src.models.example import HourlyMeanModel
from src.models.utils import (
    load_model, postprocess_predictions, save_model, save_predictions, split_xy,
)

X_train, y_train = split_xy(train_df)                   # solo horas de luz
X_all, _ = split_xy(test_df, daytime_only=False)        # las 24 horas

model = HourlyMeanModel().train(X_train, y_train)       # entrenar
save_model(model, "example.joblib")                     # guardar lo aprendido

model = load_model("example.joblib")                    # más tarde: cargar sin reentrenar
raw = model.predict(X_all)
predictions = postprocess_predictions(raw, test_df.set_index("timestamp_utc")["is_day"])
save_predictions(predictions, model.name)
```
