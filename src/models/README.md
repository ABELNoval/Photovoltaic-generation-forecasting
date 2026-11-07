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

## Flujo típico

```python
from src.models.example import HourlyMeanModel
from src.models.utils import postprocess_predictions, save_predictions, split_xy

X_train, y_train = split_xy(train_df)                   # solo horas de luz
X_all, _ = split_xy(test_df, daytime_only=False)        # las 24 horas

model = HourlyMeanModel().train(X_train, y_train)
raw = model.predict(X_all)
predictions = postprocess_predictions(raw, test_df.set_index("timestamp_utc")["is_day"])
save_predictions(predictions, model.name)
```
