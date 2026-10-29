# Estado del arte: pronóstico de generación fotovoltaica con aprendizaje automático

**Proyecto:** photovoltaic-generation-forecasting
**Asignatura:** Machine Learning
**Fase 1:** estado del arte · Octubre de 2026

> *Nota sobre las citas: los números entre corchetes, como [1] o [3, 4], remiten a la lista de **Referencias** del final del documento y señalan la fuente de cada dato o afirmación. Cuando aparecen al inicio de una frase, equivalen al nombre de los autores del trabajo citado. La numeración sigue el orden de primera aparición en el texto.*

---

## 1. Introducción

La generación fotovoltaica (FV) depende de condiciones meteorológicas que cambian de una hora a otra. Como la red eléctrica debe equilibrar en todo momento producción y consumo, conocer con antelación cuánta energía generará una planta permite planificar el despacho de otras fuentes, reducir reservas costosas y evitar penalizaciones por desviaciones entre lo programado y lo producido [1, 2]. Por eso el pronóstico de generación FV es un problema activo de investigación, con un crecimiento sostenido de publicaciones que aplican aprendizaje automático [3, 4, 5].

**Objetivo del proyecto.** Construir y comparar modelos de aprendizaje automático que, a partir de variables meteorológicas y de calendario (hora y fecha), pronostiquen la generación de un sistema fotovoltaico para el **día siguiente, hora a hora** (24 valores).

Este documento revisa los enfoques, variables, datos, algoritmos y métodos de evaluación usados en la literatura reciente, e identifica sus brechas.

## 2. Marco del problema

### 2.1 Horizontes de pronóstico

La literatura no tiene una clasificación única de horizontes:

| Fuente | Clasificación |
|---|---|
| [1] | Muy corto plazo: segundos a minutos. Corto plazo: hasta 48–72 h. Medio plazo: hasta una semana. Largo plazo: meses a un año. |
| [2] | Definición operativa del día siguiente: las ofertas al mercado se envían hasta el mediodía del día T y se pronostica el día T+1, con 12 h de antelación y 36 h de horizonte. |
| [3] | Nowcasting: menos de 15 min. Hora siguiente: 15 min a 5 h. **Día siguiente: 5 h a una semana.** Largo plazo: más de una semana. |
| [6] | Muy corto plazo: 1 min a 1 h. Corto plazo: 1 h a 1 semana. Medio y largo plazo: de 1 mes a años. |
| [7] | Corto plazo: de unas horas hasta siete días. Medio plazo: de una semana a un mes. Largo plazo: más de un mes. |
| [8] | Muy corto plazo: de segundos a menos de 30 min. Corto plazo: 30 a 360 min. Medio plazo: 6 a 24 h. Largo plazo: más de 24 h. |
| [9] | Intra-hora: menos de 1 h. Intra-día: 1 a 6 h. Inter-día: de 24 h a varios días. |
| [10] | Ultracorto plazo: minutos. Corto plazo: de un segundo a una hora. Medio y largo plazo: días y semanas. |

El pronóstico del día siguiente es el horizonte más investigado y el que se vincula con el mercado de energía y el despacho económico [3]. En una revisión de 60 artículos recientes, los horizontes más frecuentes fueron 1 hora y 1 día [4]. El error crece al alargar el horizonte: en uno de los estudios citados por [4], el nRMSE pasa de 2.6 % a 1 h a 11.7 % a 24 h, y [1] indica que la precisión se degrada sobre todo pasadas las 72 h.

Lo anterior convierte al día siguiente en un punto de equilibrio. Es lo bastante lejano como para ser útil en la planificación de la red y de los mercados de energía, donde las decisiones se toman con un día de antelación [2, 4], y lo bastante cercano como para que el error siga siendo manejable, porque la precisión se degrada con rapidez al alejarse [1, 4]. Además, a esa distancia la potencia reciente deja de bastar y el pronóstico del tiempo pasa a ser la entrada principal [3, 6], lo que encaja con un modelo construido a partir de variables meteorológicas.

### 2.2 Enfoques de pronóstico

- **Modelos físicos:** parten de la predicción numérica del tiempo (NWP) o de imágenes satelitales y requieren muchos parámetros del sistema [1].
- **Modelos estadísticos y de aprendizaje automático:** aprenden la relación entre variables y potencia a partir de datos históricos, sin necesidad de describir la planta en detalle [1].
- **Híbridos y ensambles:** combinan modelos físicos con aprendizaje automático, o varios algoritmos entre sí [3].

En los 60 artículos revisados por [4], 40 usan aprendizaje profundo, 15 aprendizaje automático tradicional, 18 modelos estadísticos, 12 ensambles, 5 híbridos y 4 modelos físicos (un artículo puede usar varios). En un benchmark de ocho proveedores comerciales para horizontes de 1 a 6 h, todos usan NWP y cinco usan imágenes satelitales, que resultan clave por debajo de unas 3 h; para más de 3 h basta integrar bien varias fuentes de NWP [9].

En conjunto, la literatura reciente se concentra en los enfoques basados en datos: solo 4 de los 60 artículos revisados usan modelos físicos, que exigen muchos parámetros del sistema [1, 4]. Los híbridos y los ensambles mejoran la precisión, pero a costa de más complejidad e interpretabilidad [3]. Por eso el aprendizaje automático supervisado, que aprende de datos históricos de la propia planta, es el enfoque más práctico para un proyecto como este.

### 2.3 Tipos de entrada

[1] distingue tres enfoques: usar solo la potencia histórica, usar solo variables meteorológicas (medidas o pronosticadas), o combinar ambas. Hasta unas dos horas de horizonte la potencia previa pesa más que la predicción numérica del tiempo [3]; para el día siguiente se necesitan pronósticos meteorológicos como entrada [6].

En resumen, para el día siguiente la potencia pasada por sí sola no alcanza: el modelo necesita variables meteorológicas, idealmente combinadas con algo de potencia histórica [1, 3, 6].

## 3. Variables de entrada

### 3.1 Irradiancia

La irradiancia es la variable dominante. En un estudio con dos plantas, la irradiancia global horizontal (GHI) alcanzó correlaciones de 0.91 (Pearson) y 0.89 (Spearman) con la potencia, y se conservó como única variable de irradiancia para evitar multicolinealidad [11]. En [7], el conjunto de variables sin irradiancia fue el que peor resultado dio, y usar todas las variables redujo el RMSE alrededor de 30 % frente a usar solo la GHI. En [8], la radiación tuvo la correlación más alta con la potencia (0.65), muy por encima de la temperatura (0.23) y la humedad (−0.28). En [12], la intensidad solar fue la variable más relacionada con la potencia (0.95 en el método de amplitud coseno, una medida de relación entre 0 y 1), seguida de temperatura (0.93) y presión (0.92). En los 60 artículos de [4], las entradas más frecuentes fueron irradiancia (39), temperatura (35), viento (27), humedad relativa (19) y presión (11).

En conjunto, la irradiancia es la entrada imprescindible: en los estudios que la compararon con otras variables siempre fue la más relacionada con la potencia, y el conjunto de variables sin ella dio el peor resultado [7, 8, 11, 12]. El resto de las variables se justifica por lo que aportan sobre ella.

### 3.2 Temperatura, nubosidad, humedad y viento

- **Temperatura:** efecto secundario pero habitual, porque afecta la eficiencia de los módulos; [11] la mantiene como entrada con una correlación moderada (Spearman de 0.29 a 0.54).
- **Nubosidad:** la nubosidad cambiante complica el pronóstico y los días nublados son más difíciles que los soleados [1, 3]. Sin embargo, la humedad se usa en más estudios que la nubosidad [3].
- **Viento:** su aporte es dudoso. Tuvo correlación despreciable en [11] (0.05) y casi nula en [8] (−0.011), con la estación en la propia planta. [3] recomienda no usar viento local salvo que se mida en el sitio, porque puede añadir ruido.
- **Humedad:** en [8] se correlaciona negativamente con la temperatura (−0.54) y con la potencia (−0.28).
- **Más variables no siempre mejora el modelo:** [3] documenta casos en que variables redundantes empeoran el resultado y recomienda selección de variables; no existe un conjunto óptimo para todos los casos.

En síntesis, después de la irradiancia, la temperatura es la variable secundaria más habitual y la nubosidad la que explica los días difíciles, mientras que el viento aporta poco y sumar variables no garantiza mejores resultados [3, 8, 11]. Conviene incorporar cada una midiendo qué aporta realmente.

### 3.3 Variables calculadas

- **Geometría solar** (elevación, azimut, ángulo cenital): se calcula con fecha, hora y ubicación. [11] reporta una correlación alta de la elevación solar con la potencia (Spearman de 0.87), y [6] la incluye como entrada junto con el azimut.
- **Hora y época del año** en forma cíclica (seno y coseno), y potencia de horas o días anteriores [1, 3]. En [11], la potencia de la hora anterior y la de 24 h antes tienen correlaciones de 0.91 y 0.82 con la potencia actual, por lo que usa 24 rezagos horarios.

Estas variables no requieren sensores adicionales: se obtienen de la fecha, la hora y la ubicación, y aportan información que la irradiancia sola no recoge, como la posición del sol y el patrón del día anterior [3, 11].

### 3.4 Datos medidos y datos pronosticados

[6] entrenó con irradiancia y temperatura pronosticadas (NWP) más los ángulos solares calculados, y obtuvo resultados ligeramente mejores que entrenando con las medidas: 6.51 % de nRMSE frente a 6.95 %, lo que atribuyen a que el modelo aprende a corregir el sesgo del pronóstico. [2] entrena con pronósticos de archivo del ECMWF (modelo IFS/HRES) tomados con 12 h de antelación y 36 h de horizonte, lo que reproduce las condiciones operativas de un mercado de energía.

Usar datos medidos en la evaluación subestima el error real. En [13], el MAE fue de 0.01 a 0.07 kW/kWp en estimación con datos medidos y de 0.06 a 0.10 kW/kWp en pronóstico. En [11], el MAE aumentó 11 % y 68 % en dos sistemas al usar pronósticos del tiempo en lugar de observaciones satelitales tomadas como pronóstico perfecto. En [14], reemplazar el clima retrospectivo por el valor pronosticado 24 h antes subió el RMSE diurno 13.1 % y 4.2 % según el protocolo de validación.

Por tanto, hay que separar dos decisiones. Entrenar con datos medidos es una práctica habitual y válida, y es la que siguen varios estudios [7, 8, 11]; la evidencia de que entrenar con pronósticos mejore el resultado es limitada: un solo estudio, con una mejora de 6.95 % a 6.51 % de nRMSE [6]. En cambio, evaluar solo con datos medidos da una imagen demasiado optimista del uso real, como muestran varios estudios [11, 13, 14], por lo que conviene medir también el efecto del error de pronóstico.

## 4. Datos

### 4.1 Datos meteorológicos y de irradiancia

| Fuente | Características |
|---|---|
| **Open-Meteo, Historical Weather API** | Reanálisis desde 1940 con resolución espacial de 9 km, gratuita y sin clave para uso no comercial. Ofrece series horarias de radiación de onda corta (equivalente a la GHI, como media de la hora precedente), nubosidad total y por capas (baja, media y alta), temperatura, humedad, viento, presión y precipitación [15]. |
| **Open-Meteo, Historical Forecast y Previous Runs API** | Archivo de pronósticos de los modelos meteorológicos disponible desde 2021 o 2022 según el modelo; la API de ejecuciones previas incluye pronósticos con 1 o más días de antelación [15]. |
| **NASA POWER** | Datos solares y meteorológicos horarios desde 2001, gratuitos y sin cuenta, basados en satélites y modelos de asimilación [16]. |
| **PVGIS (Comisión Europea)** | Series horarias de radiación y simulación de producción fotovoltaica, gratuitas y con API [17]. |

En resumen, estas fuentes cubren el tiempo, no la generación medida (PVGIS la simula) [17]. Open-Meteo destaca porque ofrece, con la misma interfaz, un histórico de reanálisis desde 1940 y archivos de pronósticos recientes [15], de modo que permite construir entradas tanto históricas como de pronóstico.

### 4.2 Datos de generación

- **175 sistemas fotovoltaicos residenciales en Utrecht (Países Bajos).** Potencia medida cada minuto de 2014 a 2017, publicada en Zenodo con licencia CC BY 4.0. Incluye mediciones sin filtrar y dos versiones con control de calidad (3.0 GB cada una), un archivo de metadatos (capacidad DC y AC, inclinación, orientación, rendimiento anual y ubicación aproximada) y el paquete Python `qcpv` de control de calidad. Los tiempos están en UTC y la potencia en vatios [18].
- **PVDAQ (NREL, Estados Unidos).** Base pública de sistemas fotovoltaicos con licencia CC BY 4.0, disponible en AWS (formatos CSV y parquet) y mediante una API en Python; algunos sistemas incluyen sensores de irradiancia, temperatura, viento y precipitación [19]. [11] la usa con datos de 2016 a 2022.
- Otros conjuntos frecuentes en la literatura incluyen el de DKASC (Alice Springs, Australia).

De estos conjuntos, el de Utrecht es el que reúne resolución fina (1 minuto), varios años de datos, metadatos de inclinación y orientación, y una rutina de control de calidad ya publicada [18]; PVDAQ aporta datos más recientes [11, 19].

### 4.3 Cantidad de datos

La mayoría de los estudios usa un año de datos y una sola planta, y [3] recomienda más de un año para cubrir todas las estaciones. En [4], el tamaño de los conjuntos va desde 1 mes hasta 55 años, con 9 artículos de 1 año y 8 de 2 años. [6] muestra que el error baja al aumentar el periodo de entrenamiento: con entrenamiento secuencial, el mejor modelo pasó de 19.14 % de nRMSE con 10 % de los datos a 6.51 % con el primer año completo.

En síntesis, un año de datos es lo habitual, pero más años mejoran el resultado y permiten cubrir todas las estaciones [3, 6].

### 4.4 Preprocesamiento

- **Noche:** la irradiancia es nula y la potencia es cero. [2] pone la noche en cero y la excluye de la evaluación, [13] la descarta con la irradiancia teórica de cielo despejado positiva, [7] conserva solo de 7:30 a 17:00 y [14] define las horas de luz con la elevación solar mayor a 5°.
- **Valores imposibles:** [2] marca los negativos como faltantes, [14] los recorta a cero y [11] elimina registros inválidos extremos y pone en cero los negativos pequeños por consumo en espera. [7] eliminó marcas de tiempo duplicadas, lecturas atascadas de sensores y periodos con el inversor apagado.
- **Faltantes:** [2] interpola linealmente el índice de cielo despejado hasta 3 horas seguidas y elimina el día si faltan más de 3. [14] marca las horas faltantes (9.25 % del año) y entrena solo con las observadas.
- **Normalización:** [2] escala las entradas por su máximo de entrenamiento y la potencia por la capacidad del sistema; [11] usa min-max, [20] estandariza y [7] normaliza para kNN. [6] normaliza la potencia por la capacidad nominal para generalizar entre plantas. [11] y [14] calculan los factores solo con el conjunto de entrenamiento.
- **Agregación temporal:** [7] agregó datos de 1 minuto a 30 minutos y [2] a 1 hora.
- **Alineación temporal:** [14] detectó un desfase aparente de una hora entre potencia e irradiancia por la hora local de registro, comparando con una estimación física simple; corregirlo mejoró esa comprobación (R² de 0.42 a 0.60).
- **Filtro de irradiancia:** [6] mejoró todos los modelos entrenando solo con alta irradiancia (más de 600 W/m²), con ganancias de 1.06 a 1.97 puntos de nRMSE.
- **Recorte de la predicción:** [2] limita las predicciones a la producción máxima posible en cielo despejado.
- **Fallas y nieve:** [9] retiró periodos con datos deteriorados, fallas de inversores y producción reducida por nieve, y perdió al menos 10 % de los datos en todas las plantas.
- **Efecto del preprocesamiento:** comparar seis métodos con tres modelos mostró que la elección puede afectar significativamente la precisión [21].
- **Estacionalidad:** [8] conservó solo las horas de 7:00 a 17:00, promedió a una hora y entrenó modelos separados por estación, con resultados que cambian mucho entre estaciones.

En conjunto, el preprocesamiento sigue un patrón común: descartar la noche y los valores inválidos, tratar los faltantes sin inventar datos, normalizar con estadísticos calculados solo en el entrenamiento y alinear bien los tiempos [2, 11, 14]. Importa porque la elección de estos pasos puede cambiar la precisión del modelo [21].

## 5. Algoritmos

### 5.1 Panorama

- Las redes neuronales son la familia más usada; SVR es el algoritmo individual más popular y LSTM se aplicó 17 veces en las 100 publicaciones revisadas por [3].
- [3] concluye que es difícil señalar un algoritmo o familia superior. Para quien empieza recomienda los **ensambles** (como Random Forest) por su relación entre precisión e interpretabilidad; si hay muchos datos y experiencia, SVR, LSTM o ELM con optimización cuidadosa de parámetros.
- SVR es muy sensible a sus parámetros, lo que explica resultados contradictorios entre estudios; los árboles simples tienden al sobreajuste, que los ensambles corrigen; y la regresión lineal rinde peor que los modelos no lineales pero sirve de referencia [3].
- [4] encuentra que ningún algoritmo supera claramente a los demás y que los ensambles dan los mejores resultados a un costo computacional mayor. Entre los métodos de ML tradicional, los más comunes son Random Forest (8 artículos) y SVM (6); entre los de aprendizaje profundo, las redes recurrentes como LSTM.
- [10] señala que la mayoría de los métodos revisados son híbridos y que estos suelen rendir mejor, que el error crece con el horizonte y que los resultados dependen de la ubicación y del clima.

En resumen, no hay un algoritmo ganador, pero sí un grupo de candidatos habituales: redes neuronales, SVR y métodos de ensamble como Random Forest, con la regresión lineal como referencia [3, 4].

### 5.2 Comparaciones directas relevantes

| Estudio | Algoritmos | Resultado |
|---|---|---|
| [7] (día siguiente, un año, estación solarimétrica) | MLP, SVR, kNN, árbol, regresión lineal | Con todas las variables, RMSE: MLP 45.92, SVR 52.20, regresión lineal 59.03, árbol 65.47, kNN 94.95. kNN fue el peor en general. |
| [6] (día siguiente, dos años) | Red neuronal bayesiana, SVR, árbol de regresión | nRMSE de 4.53 %, 6.37 % y 7.37 %, con diferencias estadísticamente significativas. |
| [2] (día siguiente, 152 sistemas en Utrecht) | Regresión lineal, LASSO, SARIMAX, SVR, Random Forest, Gradient Boosting, red neuronal, LSTM, modelo físico y dos persistencias | En un sistema individual, el MAE (en % de la capacidad) fue 6.13 % para RF, 6.22 % para LSTM y 6.29 % para GB, frente a 11.3 % y 11.4 % de las persistencias. |
| [13] (estimación y día siguiente, 2 plantas) | Regresión lineal y Random Forest con distintos conjuntos de variables | RF supera a la regresión lineal con todos los conjuntos de variables probados. |
| [20] (1 a 3 días, regional) | SVM, RF, Gradient Boosting, red neuronal, ensambles análogos y otros | En Utrecht (150 sistemas), skill score de 42.1 a 43.7 % frente a la persistencia y RMSE de 9.9 a 10.2 % de la capacidad. |
| [8] (1 a 4 h, 20 kW, un año) | RF, DNN, ANN y SVM (referencia) | RF tiene el mejor R² en varios casos a 1 y 2 h; DNN lo supera en varias estaciones a 3 y 4 h. En la estación de monzón todos rinden mal (R² entre 0.19 y 0.40). |
| [12] (día siguiente, datos diarios de un año) | KNN, LSTM, SVM, LSTM-SVM, LSTM-KNN | R² en prueba: LSTM-KNN 0.987, LSTM 0.958, KNN 0.903, LSTM-SVM 0.764, SVM 0.489. |

Resultados de otros estudios de día siguiente: SVR fue el mejor modelo individual (22.49 % de nRMSE) y una mezcla ponderada lo superó levemente (22.24 %) en [22].

Leídas en conjunto, estas comparaciones confirman que el mejor algoritmo cambia con los datos y el ajuste, y que las redes neuronales, SVR y los ensambles suelen quedar entre los primeros [2, 6, 7, 8].

### 5.3 Observaciones sobre los estudios comparados

Las diferencias entre estudios reflejan que el resultado depende de los datos, la métrica y el ajuste de parámetros: kNN fue el peor en [7] pero rindió bien en [12], y SVM quedó último en [12] mientras que SVR fue el mejor individual en [22]. Dos detalles metodológicos conviene tener presentes:

- En [12], los datos son diarios y de un solo año (2022) aunque el resumen habla de pronóstico horario, la "precisión" reportada es el R² del conjunto de prueba, la partición se describe de forma contradictoria y los resultados de SVM presentan inconsistencias numéricas; es un ejemplo útil de híbrido, pero con evidencia limitada.
- En [8], el texto afirma que Random Forest supera siempre a los demás, pero su tabla muestra que DNN tiene mejor R² en varias estaciones a 3 y 4 horas, y el skill score usa SVM como referencia y no la persistencia.

La enseñanza práctica es comparar los algoritmos bajo las mismas condiciones y reportar siempre la partición, la métrica y la referencia utilizadas, para que los resultados puedan interpretarse [8, 12].

### 5.4 Combinaciones y modelos híbridos

Los híbridos aumentan la precisión a costa de complejidad e interpretabilidad [3]. La ganancia observada puede ser pequeña (22.49 % frente a 22.24 % en [22]). El informe de IEA midió el efecto de **promediar** pronósticos para toda Italia: promediar los dos mejores modelos subió el skill score de 46.6 % a casi 50 %, y una mezcla no lineal de seis modelos lo llevó a 51.3 %. El promedio simple solo supera al mejor modelo si la diferencia de RMSE entre el mejor y el peor es menor al 1 % de la capacidad, y los autores lo recomiendan como la primera mezcla a probar [20].

En síntesis, promediar los mejores modelos es una mejora barata de probar, cuya ganancia depende de que sus errores sean parecidos [20, 22].

### 5.5 Configuraciones de hiperparámetros reportadas

| Algoritmo | Configuraciones reportadas | Fuente |
|---|---|---|
| Red neuronal (MLP) | Una capa oculta con 3 a 11 neuronas, activación tanh o ReLU y optimizador Adam; la mejor tuvo 9 neuronas y tanh. En otro estudio, dos capas ocultas con ReLU y 90 y 80 neuronas, con búsqueda en malla y validación cruzada de 5 bloques; un conjunto de 200 redes con distinta inicialización redujo el MAE de 3.65 % a 2.81 %. | [7], [11] |
| SVR | C en {1, 10, 50, 100, 1000}, kernel RBF, polinomial o lineal y epsilon 0.01; C = 1000 fue el mejor en todos los casos. | [7] |
| kNN | k en {3, 5, 7, 9, 11}, distancia euclidiana con variables estandarizadas; los mejores k estuvieron entre 5 y 11. | [7] |
| Árbol de decisión | Mínimo de muestras por hoja de 5 o 10. | [7] |
| Random Forest / Gradient Boosting | Búsqueda en malla con validación cruzada secuencial. En un estudio, profundidad máxima 8, mínimo de 30 muestras por hoja, tasa de aprendizaje 0.04 y parada temprana. | [2], [14] |

Estos rangos sirven como punto de partida para la búsqueda de hiperparámetros de cada algoritmo [2, 7, 11, 14].

## 6. Evaluación

### 6.1 Métricas

- **RMSE** es la más popular, seguida de MAE y sus normalizaciones [3]. En [4]: RMSE (28 de 60 artículos), MAE (23), nRMSE (18) y MAPE (11). Los autores señalan que la falta de un estándar limita la comparación entre estudios.
- Las **versiones normalizadas** por la capacidad instalada permiten comparar plantas de distinto tamaño [3]. Conviene declarar la normalización: [2] normaliza por la capacidad (MAE de unos 6 %), mientras que [9] normaliza por la potencia media del periodo y obtiene nRMSE de 27 a 58 % en un periodo de noviembre a mayo; ambas cifras no son comparables.
- El **skill score** compara con una referencia: 1 menos el cociente entre el RMSE del modelo y el de la referencia; 0 significa igual y valores positivos significan mejora [3, 14].
- [3] recomienda reportar más de una métrica y analizar el tipo de error. Calcular las métricas solo en horas de luz evita inflar el desempeño con ceros nocturnos [14].

En síntesis, conviene reportar varias métricas, normalizarlas por la capacidad instalada y declararlo, y calcularlas solo en horas de luz [3, 9, 14].

### 6.2 Referencias de comparación

La referencia habitual es la **persistencia**: suponer que el futuro repite el pasado. Para el día siguiente se usa la producción del día anterior [6, 14], que es aplicable cuando el tiempo varía poco [6]. [2] explica por qué se usa el día anterior: a las 12 h del día T solo se conoce la producción hasta el mediodía, así que su persistencia diurna toma el día T−1 (MAE de 11.3 %), y añade una persistencia de cielo despejado (11.4 %). [14] agrega una persistencia "inteligente" que rescala la del día anterior con la irradiancia del día objetivo.

Así, un modelo solo demuestra su valor si supera a estas referencias simples; por eso el resultado se expresa como skill score frente a la persistencia del día anterior [2, 14].

### 6.3 Validación

- [7] usa validación **walk-forward** con una ventana fija de unos 30 días: entrena y predice el día siguiente, y avanza un día.
- [6] entrena con un año y prueba con el siguiente completo, y analiza el efecto del periodo y del orden de los datos.
- [2] y [13] entrenan con dos años y prueban con el siguiente, y ajustan hiperparámetros con validación cruzada secuencial de 8 bloques (entrenar con 21 meses y validar en un trimestre), sin tocar el año de prueba. [20] usa 3 bloques de 8 meses.
- [14] comparó dos protocolos con el mismo modelo: con días asignados al azar, el nRMSE diurno fue 12.4 % de la capacidad, y con orden cronológico (ventanas móviles), 18.1 %. Parte de la brecha viene de la cobertura estacional, pero la validación aleatoria es más optimista.
- La proporción habitual de entrenamiento y prueba es de 70–80 % y 20–30 % [10].
- [9] clasifica cada hora con el índice de cielo despejado (KPV, cociente entre irradiancia medida y de cielo despejado): menor a 0.4, cielo cubierto; entre 0.4 y 0.7, mixto; mayor a 0.7, despejado. Permite analizar el error por condición del cielo.

En conclusión, la validación debe respetar el orden del tiempo, porque la validación aleatoria es más optimista y no refleja el uso real [14]; además, clasificar las horas por tipo de cielo permite ver en qué condiciones falla cada modelo [9].

## 7. Trabajos relacionados

| Referencia | Horizonte | Datos | Variables | Algoritmos | Métricas |
|---|---|---|---|---|---|
| [1] | Revisión (varios horizontes) | Publicaciones de 2010–2019 | Varias | ML, DL e híbridos | Varias |
| [2] | Día siguiente (oferta al mediodía de T para T+1) | 152 sistemas residenciales en Utrecht, 1 min promediado a 1 h, ene 2014 a feb 2017 | 12 variables de pronóstico del ECMWF, hora (seno y coseno), irradiancia de cielo despejado y ángulos solares | 12 modelos (regresión lineal, LASSO, SARIMAX, SVR, RF, GB, ANN, LSTM, modelo físico y 2 persistencias) | MAE, RMSE, MBE e ingresos económicos |
| [3] | Revisión (varios horizontes) | 100 publicaciones, 2015–2020 | Varias | Todas las familias de ML | Varias |
| [4] | Revisión de 60 artículos (2020 a 2023), de 1 min a 1 mes | Desde 1 mes hasta 55 años de datos | Irradiancia, temperatura, viento, humedad, presión | 40 con deep learning, 15 ML, 18 estadísticos, 12 ensambles, 5 híbridos, 4 físicos | RMSE, MAE, nRMSE, MAPE |
| [6] | Día siguiente | Sistema de 1.175 kWp, Nicosia (Chipre), 2 años horarios + NWP | GHI y temperatura (medidas y pronosticadas), azimut y elevación solar | Red bayesiana, SVR, árbol | MAE, MAPE, nRMSE, sesgo, skill score |
| [7] | Día siguiente | Estación solarimétrica y planta, Florianópolis (Brasil), 2018, 5 375 observaciones de 30 min | DNI, GHI, DHI, UV, temperatura, viento, humedad | MLP, SVR, kNN, árbol, regresión lineal | RMSE, R² |
| [8] | 1 a 4 h | Sistema de 20 kW y estación meteorológica en el sitio, Godhra (India), oct 2021 a sep 2022, 2 460 muestras horarias de 7:00 a 17:00 | Temperatura, humedad, radiación, velocidad y dirección del viento, hora del día | RF, DNN, ANN, SVM (referencia) | R², MSE, MAPE, skill score relativo |
| [9] | Intra-día (1 a 6 h) | 7 plantas (36 kWp a 7.6 MWp) en 4 países, 6 meses de invierno y 1 año de historial | Potencia e irradiancia en el plano de los paneles; cada proveedor usa NWP y a veces satélite | 8 soluciones comerciales | MAE, RMSE, MBE y versiones normalizadas |
| [10] | Revisión, muy corto a largo plazo | Revisión descriptiva por tipo de red neuronal | Parámetros de entrada y resolución | ANN, RNN y CNN, convencionales e híbridas | MAE, RMSE, MAPE, entre otras |
| [11] | Día siguiente, 24 horas | 2 sistemas (2.4 y 6 kW) de PVDAQ, entrenamiento 2016 a 2021 y prueba en 2022 | GHI, temperatura, elevación solar y 24 rezagos de potencia | Conjunto de 200 redes MLP | MAE, sesgo |
| [12] | Día siguiente (datos diarios) | Planta de 1.17 MWp en Hilvan (Turquía) y datos meteorológicos del servicio nacional, año 2022 | Presión, nubosidad, humedad, temperatura, insolación | KNN, LSTM, SVM, LSTM-KNN, LSTM-SVM | MSE, RMSE, NRMSE, R², PSNR |
| [13] | Estimación y pronóstico del día siguiente | Sistemas de 2.7 kWp (De Bilt, Países Bajos) y 4.2 kWp (Bolzano, Italia), 3 años cada uno | 26 variables (meteorológicas, de irradiancia y calculadas) | Regresión lineal y RF; QR y QRF probabilísticos | MAE, RMSE, sesgo, PICP, PINAW, CRPS |
| [14] | Día siguiente, hora a hora | 1 año de potencia de un cargador solar en Southampton (Reino Unido), 391 días completos; clima de Open-Meteo | 15 variables meteorológicas, geometría solar, índice de claridad y contexto de 3 horas | Ridge, GBM (3 variantes), CNN y su combinación por mínimos cuadrados no negativos | R², MAE, RMSE, normalizados y skill score |
| [20] | 1 a 3 días, regional | Utrecht (150 sistemas), 6 zonas de Italia y toda Italia | Pronósticos NWP (ECMWF en Utrecht) e irradiancia satelital en Italia | SVM, RF, GB, red neuronal, ensambles análogos, QRF, KNN y mezclas | RMSE y MBE por capacidad, skill score |

Los trabajos más cercanos al que se plantea son [2, 6, 7, 11, 13, 14]: todos predicen el día siguiente hora a hora con variables meteorológicas como entrada. El resto aporta contexto sobre horizontes, variables y evaluación. Las coincidencias concretas con el proyecto son:

- **[2]** compara doce modelos sobre los mismos datos, entre ellos regresión lineal, SVR, Random Forest, Gradient Boosting y redes neuronales, con la persistencia como referencia, y usa sistemas residenciales de Utrecht, el mismo tipo de datos que el conjunto descrito en la sección 4.2.
- **[6]** compara una red neuronal, SVR y un árbol de regresión sobre el mismo conjunto de datos, entrena con un año y prueba con el siguiente, y reporta nRMSE y skill score frente a la persistencia.
- **[7]** es el más parecido en algoritmos y entradas: compara MLP, SVR, kNN, árbol de decisión y regresión lineal usando solo variables meteorológicas, con validación en orden cronológico.
- **[11]** predice las 24 horas siguientes con redes neuronales sobre PVDAQ (sección 4.2) y mide cuánto aumenta el error al pasar de observaciones a pronósticos del tiempo.
- **[13]** usa datos de un sistema del mismo conjunto público de Utrecht, compara regresión lineal y Random Forest con pronósticos del ECMWF y mide la diferencia de error entre datos medidos y pronóstico.
- **[14]** usa Open-Meteo como fuente meteorológica (sección 4.1), compara la validación aleatoria con la cronológica, toma la persistencia como referencia y combina varios modelos.

## 8. Síntesis y brechas

**Hallazgos que se repiten en la literatura**
1. La irradiancia y la geometría solar son la base; el resto de las variables aporta menos y depende del caso [3, 7].
2. Ningún algoritmo gana siempre. Los ensambles y las redes suelen quedar arriba, y SVR depende mucho de su ajuste [2, 3, 4].
3. El error crece con el horizonte y con la nubosidad [1, 3, 4].
4. El pronóstico del tiempo es una fuente grande de error: +11 % y +68 % de MAE en [11], +13.1 % y +4.2 % de RMSE en [14], y de 0.01 a 0.07 frente a 0.06 a 0.10 kW/kWp en [13].
5. La forma de evaluar importa tanto como el modelo: validación aleatoria frente a cronológica [14], referencias de persistencia [2] y normalización [9].
6. La calidad y la preparación de los datos pesan tanto o más que la elección del algoritmo [3].

**Brechas identificadas**
- Casi todos los estudios usan una sola planta y uno o pocos años de datos [3]; [2] advierte que no se sabe si mejoras marginales con un único caso son universales.
- No hay estándar de métricas, horizontes ni datasets, así que las cifras de distintos trabajos no son comparables [4].
- Pocos trabajos evalúan con pronósticos reales del tiempo en lugar de datos medidos [2, 11].
- El mejor modelo cambia según se mida el error técnico o el beneficio económico [2].
- Hay pocas implementaciones en hardware para uso en tiempo real: solo 4 de 60 artículos [4].

**Errores de referencia para el día siguiente** (orientativos; cada trabajo usa otra planta, otro periodo y otra normalización):
- MAE de alrededor de 6 % de la capacidad para RF y LSTM en un sistema residencial [2]; RMSE de 10 a 13 % de la capacidad en [13].
- nRMSE de 4.5 a 7.4 % con dos años de datos [6].
- nRMSE diurno de 12 a 18 % de la capacidad con un año de datos [14].

Frente a estas brechas, el proyecto compara varios algoritmos sobre los mismos datos y la misma partición, evalúa de forma cronológica frente a una referencia de persistencia y mide el efecto del error de pronóstico con un conjunto de datos recientes.

## Referencias

[1] Mellit, A.; Massi Pavan, A.; Ogliari, E.; Leva, S.; Lughi, V. (2020). Advanced Methods for Photovoltaic Output Power Forecasting: A Review. *Applied Sciences*, 10(2), 487. https://doi.org/10.3390/app10020487

[2] Visser, L.; AlSkaif, T.; van Sark, W. (2022). Operational day-ahead solar power forecasting for aggregated PV systems with a varying spatial distribution. *Renewable Energy*, 183, 267–282. https://doi.org/10.1016/j.renene.2021.10.102

[3] Alcañiz, A.; Grzebyk, D.; Ziar, H.; Isabella, O. (2023). Trends and gaps in photovoltaic power forecasting with machine learning. *Energy Reports*, 9, 447–471. https://doi.org/10.1016/j.egyr.2022.11.208

[4] Gallardo, I.; Amor, D.; Gutiérrez, Á. (2023). Recent Trends in Real-Time Photovoltaic Prediction Systems. *Energies*, 16(15), 5693. https://doi.org/10.3390/en16155693

[5] Gaboitaolelwe, J.; Zungeru, A. M.; Yahya, A.; Lebekwe, C. K.; Vinod, D. N.; Salau, A. O. (2023). Machine Learning Based Solar Photovoltaic Power Forecasting: A Review and Comparison. *IEEE Access*, 11, 40820–40845. https://doi.org/10.1109/ACCESS.2023.3270041

[6] Theocharides, S.; Theristis, M.; Makrides, G.; Kynigos, M.; Spanias, C.; Georghiou, G. E. (2021). Comparative Analysis of Machine Learning Models for Day-Ahead Photovoltaic Power Production Forecasting. *Energies*, 14(4), 1081. https://doi.org/10.3390/en14041081

[7] Pelisson, A.; Covões, T.; Spengler, A.; Jaskowiak, P. (2020). Comparative Study of Photovoltaic Power Forecasting Methods. *Anais do 17.º Encontro Nacional de Inteligência Artificial e Computacional (ENIAC)*, Sociedade Brasileira de Computação, 555–566. https://doi.org/10.5753/eniac.2020.12159

[8] Jogunuri, S.; Josh, F. T.; Jency Joseph, J.; Meenal, R.; Mohan Das, R.; Kannadhasan, S. (2024). Forecasting hourly short-term solar photovoltaic power using machine learning models. *International Journal of Power Electronics and Drive Systems*, 15(4), 2553–2569. https://doi.org/10.11591/ijpeds.v15.i4.pp2553-2569

[9] Lehmann, J.; Koessler, C.; Ruiz Gomez, L.; Scheerlinck, S. (2023). Benchmark of eight commercial solutions for deterministic intra-day solar forecast. *EPJ Photovoltaics*, 14, 15. https://doi.org/10.1051/epjpv/2023006

[10] Radzi, P. N. L. M.; Akhter, M. N.; Mekhilef, S.; Mohamed Shah, N. (2023). Review on the Application of Photovoltaic Forecasting Using Machine Learning for Very Short- to Long-Term Forecasting. *Sustainability*, 15(4), 2942. https://doi.org/10.3390/su15042942

[11] Danner, P.; de Meer, H. *Two-Stage Photovoltaic Forecasting: Separating Weather Prediction from Plant-Characteristics*. Universidad de Passau. arXiv:2603.04132.

[12] Ağır, T. T. (2025). Estimation of Daily Photovoltaic Power One Day Ahead With Hybrid Deep Learning and Machine Learning Models. *Energy Science & Engineering*, 13(4), 1478–1491. https://doi.org/10.1002/ese3.1994

[13] Visser, L.; AlSkaif, T.; Hu, J.; Louwen, A.; van Sark, W. (2023). On the value of expert knowledge in estimation and forecasting of solar photovoltaic power generation. *Solar Energy*, 251, 86–105. https://doi.org/10.1016/j.solener.2023.01.019

[14] Dehghan, F.; Stein, S.; Yazdanpanah, V.; Gauthier, S.; Nazari, M. (2026). An AI-Based Decision-Support Pipeline for Day-Ahead Photovoltaic Forecasting. *Proceedings of the UK AI Conference (UK-AI 2026)*. arXiv:2608.02088.

[15] Zippenfenig, P. (2023). *Open-Meteo.com Weather API*. Documentación: https://open-meteo.com/en/docs/historical-weather-api y https://open-meteo.com/en/docs/historical-forecast-api

[16] NASA. *Prediction of Worldwide Energy Resources (POWER) Project*. https://power.larc.nasa.gov

[17] Comisión Europea, Joint Research Centre. *Photovoltaic Geographical Information System (PVGIS)*. https://joint-research-centre.ec.europa.eu/pvgis-online-tool_en

[18] Visser, L.; Elsinga, B.; AlSkaif, T.; van Sark, W. (2022). Open-source quality control routine and multi-year power generation data of 175 PV systems. *Journal of Renewable and Sustainable Energy*, 14, 043501. https://doi.org/10.1063/5.0100939. Datos: https://doi.org/10.5281/zenodo.6906504

[19] Deline, C.; Perry, K.; Deceglie, M.; Muller, M.; Sekulic, W.; Jordan, D. (2021). *Photovoltaic Data Acquisition (PVDAQ) Public Datasets*. NREL. https://doi.org/10.25984/1846021

[20] Betti, A.; Pierro, M.; Cornaro, C.; Moser, D.; Moschella, M.; Collino, E.; Ronzio, D.; van der Meer, D.; Visser, L.; Widén, J.; AlSkaif, T.; van Sark, W. (2020). *Regional Solar Power Forecasting*. IEA PVPS Task 16, Report IEA-PVPS T16-01:2020. https://doi.org/10.13140/RG.2.2.32196.24964

[21] Özdemir, G.; Kuzlu, M.; Ozdemir, U.; Catak, F. O. (2024). Optimizing ML-Based Solar PV Forecasting Models in Smart Grids. *IEEE Global Energy Conference (GEC)*. https://doi.org/10.1109/gec61857.2024.10881299

[22] Antonanzas, J.; Urraca, R.; Pernía-Espinoza, A.; Aldama, A.; Fernández-Jiménez, L. A.; Martínez-de-Pisón, F. J. (2017). Single and Blended Models for Day-Ahead Photovoltaic Power Forecasting. En *Hybrid Artificial Intelligent Systems (HAIS 2017)*, Lecture Notes in Computer Science 10334, Springer, 427–434. https://doi.org/10.1007/978-3-319-59650-1_36
