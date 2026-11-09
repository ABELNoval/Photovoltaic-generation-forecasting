# Propuesta del proyecto: pronóstico de generación fotovoltaica con aprendizaje automático

**Proyecto:** photovoltaic-generation-forecasting
**Asignatura:** Machine Learning
**Fase 1:** propuesta de diseño · Octubre de 2026

> *Nota sobre las citas: los números entre corchetes, como [1] o [3, 4], remiten a la lista de **Referencias** del final del documento y señalan la fuente de cada dato o afirmación. La numeración sigue el orden de primera aparición en el texto.*

---

## Objetivo

Construir y comparar modelos de aprendizaje automático que, a partir de variables meteorológicas y de calendario (hora y fecha), pronostiquen la generación de un sistema fotovoltaico para el **día siguiente, hora a hora** (24 valores). Las decisiones de diseño que siguen se derivan de la revisión de la literatura presentada en el documento de estado del arte.

## 1. Tipo de problema y horizonte

**Decisión.** Regresión de la potencia horaria de un sistema fotovoltaico para las 24 horas del día siguiente, evaluada en las horas de luz. Los niveles de producción (baja, media, alta) se derivan del valor numérico solo para presentar resultados.
**Justificación.** El día siguiente es el horizonte más investigado y de mayor relevancia para la red [1, 2, 3]; predecir el valor numérico permite obtener categorías, pero no a la inversa [3, 4]; y calcular las métricas solo de día evita inflar el desempeño con ceros nocturnos [5].

## 2. Datos

**Decisión.**
- **Generación:** el conjunto de 175 sistemas residenciales de Utrecht [6], en su versión con control de calidad. Se trabajará con un sistema individual (o pocos), elegido entre los que cubren 2014–2017 según los metadatos y con una disponibilidad diurna de al menos 95 % calculada sobre las mediciones [7], y se agregarán de 1 minuto a 1 hora.
- **Meteorología:** el archivo histórico de Open-Meteo en las coordenadas del sistema [8], con radiación de onda corta (GHI), nubosidad total y por capas, temperatura, humedad, viento, presión y precipitación, en UTC como la potencia.
- **Periodo:** entrenamiento con 2014–2015, validación y comparación de modelos con 2016, y evaluación final con 2017.
- **Complemento:** para medir el efecto del error de pronóstico se usará un sistema de PVDAQ con datos de 2022 o posteriores [9, 10] junto con el archivo de pronósticos de Open-Meteo, disponible desde 2021 o 2022 [8].

**Justificación.** El conjunto de Utrecht es abierto, tiene resolución de 1 minuto, cuatro años de datos, metadatos de inclinación y orientación, y una rutina de control de calidad ya publicada [6]; es el mismo tipo de datos usado en los estudios de día siguiente de mayor detalle [7, 11]. Más de un año de datos cubre todas las estaciones [1]. Los pronósticos archivados existen desde 2021 o 2022 [8], por lo que la evaluación con pronósticos reales requiere datos recientes, como los de PVDAQ [9].

## 3. Variables

**Decisión.**
- *Núcleo:* radiación de onda corta (GHI), nubosidad y temperatura.
- *Secundarias, sujetas a selección de variables:* humedad, viento y presión.
- *Calendario:* hora del día y día del año en forma cíclica (seno y coseno).

**Justificación.** La irradiancia domina, la temperatura aporta, y las variables adicionales pueden ayudar o perjudicar según el caso [1, 4, 9]. La hora y la fecha en forma cíclica recogen el ciclo diario y estacional sin necesidad de sensores [1, 7].

## 4. Preprocesamiento

**Decisión.** Excluir la noche, es decir, las horas sin radiación en la fuente meteorológica; eliminar valores negativos; interpolar vacíos de hasta 3 horas y descartar días con más faltantes; normalizar la potencia por la capacidad AC y escalar las entradas con estadísticos del entrenamiento; revisar la alineación horaria entre la potencia (promedio horario) y la radiación (media de la hora precedente); y probar el filtro de alta irradiancia en el entrenamiento. Cada paso se documenta en el repositorio.
**Justificación.** Son las prácticas que se repiten en los estudios de referencia [3, 5, 7, 9, 11], y la elección del preprocesamiento afecta la precisión [12].

## 5. Algoritmos

**Decisión.** Referencias: persistencia del día anterior, persistencia con rescalado por irradiancia y regresión lineal. Candidatos de familias distintas: red neuronal (MLP), SVR, Random Forest y kNN, ajustados con los rangos reportados en la literatura (estado del arte, sección 5.5) y comparados sobre los mismos datos y la misma partición.
**Justificación.** No hay un algoritmo superior en la literatura [1, 2]; Random Forest se respalda en la recomendación de ensambles [1, 7]; las redes y SVR son los métodos más usados [1]; y kNN aporta un contraste de familia basada en distancias, con resultados mixtos entre estudios [4, 13].

## 6. Evaluación

**Decisión.** MAE, RMSE, nRMSE normalizado por la capacidad instalada, R² y skill score frente a la persistencia del día anterior. Validación cruzada secuencial para ajustar hiperparámetros dentro del entrenamiento, comparación en 2016 y evaluación final en 2017 sin volver a ajustar. Análisis del error por condición del cielo (despejado, parcial y cubierto) a partir de la nubosidad de la fuente meteorológica.
**Justificación.** Se alinea con las recomendaciones de [1], con el protocolo cronológico de [4], [7] y [5] (la validación aleatoria es más optimista [5]), y con la idea de clasificar el cielo por condiciones de [14], aquí basada en la nubosidad.

## 7. Combinación final

**Decisión.** Promediar los dos o tres mejores modelos y reportar el resultado frente al mejor modelo individual.
**Justificación.** El promedio simple es la primera mezcla recomendada y mejora el skill score cuando los errores de los modelos son parecidos [15, 16].

## Referencias

[1] Alcañiz, A.; Grzebyk, D.; Ziar, H.; Isabella, O. (2023). Trends and gaps in photovoltaic power forecasting with machine learning. *Energy Reports*, 9, 447–471. https://doi.org/10.1016/j.egyr.2022.11.208

[2] Gallardo, I.; Amor, D.; Gutiérrez, Á. (2023). Recent Trends in Real-Time Photovoltaic Prediction Systems. *Energies*, 16(15), 5693. https://doi.org/10.3390/en16155693

[3] Theocharides, S.; Theristis, M.; Makrides, G.; Kynigos, M.; Spanias, C.; Georghiou, G. E. (2021). Comparative Analysis of Machine Learning Models for Day-Ahead Photovoltaic Power Production Forecasting. *Energies*, 14(4), 1081. https://doi.org/10.3390/en14041081

[4] Pelisson, A.; Covões, T.; Spengler, A.; Jaskowiak, P. (2020). Comparative Study of Photovoltaic Power Forecasting Methods. *Anais do 17.º Encontro Nacional de Inteligência Artificial e Computacional (ENIAC)*, Sociedade Brasileira de Computação, 555–566. https://doi.org/10.5753/eniac.2020.12159

[5] Dehghan, F.; Stein, S.; Yazdanpanah, V.; Gauthier, S.; Nazari, M. (2026). An AI-Based Decision-Support Pipeline for Day-Ahead Photovoltaic Forecasting. *Proceedings of the UK AI Conference (UK-AI 2026)*. arXiv:2608.02088.

[6] Visser, L.; Elsinga, B.; AlSkaif, T.; van Sark, W. (2022). Open-source quality control routine and multi-year power generation data of 175 PV systems. *Journal of Renewable and Sustainable Energy*, 14, 043501. https://doi.org/10.1063/5.0100939. Datos: https://doi.org/10.5281/zenodo.6906504

[7] Visser, L.; AlSkaif, T.; van Sark, W. (2022). Operational day-ahead solar power forecasting for aggregated PV systems with a varying spatial distribution. *Renewable Energy*, 183, 267–282. https://doi.org/10.1016/j.renene.2021.10.102

[8] Zippenfenig, P. (2023). *Open-Meteo.com Weather API*. Documentación: https://open-meteo.com/en/docs/historical-weather-api y https://open-meteo.com/en/docs/historical-forecast-api

[9] Danner, P.; de Meer, H. *Two-Stage Photovoltaic Forecasting: Separating Weather Prediction from Plant-Characteristics*. Universidad de Passau. arXiv:2603.04132.

[10] Deline, C.; Perry, K.; Deceglie, M.; Muller, M.; Sekulic, W.; Jordan, D. (2021). *Photovoltaic Data Acquisition (PVDAQ) Public Datasets*. NREL. https://doi.org/10.25984/1846021

[11] Visser, L.; AlSkaif, T.; Hu, J.; Louwen, A.; van Sark, W. (2023). On the value of expert knowledge in estimation and forecasting of solar photovoltaic power generation. *Solar Energy*, 251, 86–105. https://doi.org/10.1016/j.solener.2023.01.019

[12] Özdemir, G.; Kuzlu, M.; Ozdemir, U.; Catak, F. O. (2024). Optimizing ML-Based Solar PV Forecasting Models in Smart Grids. *IEEE Global Energy Conference (GEC)*. https://doi.org/10.1109/gec61857.2024.10881299

[13] Ağır, T. T. (2025). Estimation of Daily Photovoltaic Power One Day Ahead With Hybrid Deep Learning and Machine Learning Models. *Energy Science & Engineering*, 13(4), 1478–1491. https://doi.org/10.1002/ese3.1994

[14] Lehmann, J.; Koessler, C.; Ruiz Gomez, L.; Scheerlinck, S. (2023). Benchmark of eight commercial solutions for deterministic intra-day solar forecast. *EPJ Photovoltaics*, 14, 15. https://doi.org/10.1051/epjpv/2023006

[15] Betti, A.; Pierro, M.; Cornaro, C.; Moser, D.; Moschella, M.; Collino, E.; Ronzio, D.; van der Meer, D.; Visser, L.; Widén, J.; AlSkaif, T.; van Sark, W. (2020). *Regional Solar Power Forecasting*. IEA PVPS Task 16, Report IEA-PVPS T16-01:2020. https://doi.org/10.13140/RG.2.2.32196.24964

[16] Antonanzas, J.; Urraca, R.; Pernía-Espinoza, A.; Aldama, A.; Fernández-Jiménez, L. A.; Martínez-de-Pisón, F. J. (2017). Single and Blended Models for Day-Ahead Photovoltaic Power Forecasting. En *Hybrid Artificial Intelligent Systems (HAIS 2017)*, Lecture Notes in Computer Science 10334, Springer, 427–434. https://doi.org/10.1007/978-3-319-59650-1_36
