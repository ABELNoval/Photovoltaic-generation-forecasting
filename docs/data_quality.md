# Calidad del dataset completo ID003

Este informe describe la cobertura de las fuentes, las correcciones aplicadas
y las filas que se conservaron al construir el dataset horario de ID003 para
2014–2017. Las reglas de limpieza y el esquema compartido no se modifican aquí.

## Reproducibilidad

Desde la raíz del repositorio, con los archivos de entrada disponibles en
`data/raw/` y `data/processed/`, ejecutar:

```bash
python -m src.data.build_dataset ID003
```

El comando regenera `data/processed/dataset_ID003.csv` e imprime las mismas
estadísticas globales y anuales descritas abajo. Para regenerar los datos de
entrada se usan `python -m src.data.pv_power extract ID003` y
`python -m src.data.weather ID003`. Los archivos brutos no se versionan; el
CSV final compartido sí se publica como la excepción acordada en el issue #41.

## Fuentes y cobertura de potencia

Las mediciones de potencia se agregan por hora con etiquetas al final del
intervalo. Una hora cuenta como observada si contiene al menos 45 minutos de
mediciones. Cada medianoche se asigna al año de la hora que termina; por
ejemplo, `2018-01-01 00:00 UTC` representa la última hora de 2017.

| Año de la hora | Horas esperadas | Con potencia | Sin potencia | Cobertura |
|---|---:|---:|---:|---:|
| 2014 | 8 760 | 8 679 | 81 | 99,08 % |
| 2015 | 8 760 | 7 461 | 1 299 | 85,17 % |
| 2016 | 8 784 | 6 166 | 2 618 | 70,20 % |
| 2017 | 8 760 | 8 312 | 448 | 94,89 % |
| **Total** | **35 064** | **30 618** | **4 446** | **87,32 %** |

El archivo meteorológico contiene 35 065 timestamps, incluidos los límites de
la descarga, desde `2014-01-01 00:00 UTC` hasta `2018-01-01 00:00 UTC`. No
tiene timestamps duplicados ni valores meteorológicos faltantes. En las horas
esperadas del periodo hay cero valores meteorológicos faltantes.

## Limpieza aplicada

| Acción o resultado | Horas/días |
|---|---:|
| Valores negativos de potencia corregidos a cero | 0 |
| Potencia nocturna faltante completada con cero | 1 987 |
| Huecos de hasta 3 horas interpolados antes de descartar días | 457 |
| De esas interpolaciones, conservadas en el dataset | 385 |
| Interpolaciones en días que luego se descartaron | 72 |
| Potencia todavía faltante tras las correcciones | 2 002 horas |
| Días con potencia faltante después de limpiar | 208 |

La potencia nocturna faltante se completa cuando `ghi` indica noche (`ghi = 0`).
Los huecos internos de hasta 3 horas se interpolan linealmente en el tiempo.
Los huecos más largos o sin observaciones a ambos lados permanecen faltantes.
Después de esas acciones, si un día conserva alguna hora sin potencia, se
descartan las 24 filas de ese día. Por ello, algunas horas interpoladas también
se descartan junto con el día al que pertenecen.

Los conteos de valores negativos y correcciones nocturnas describen las
acciones realizadas antes de descartar días. Las interpolaciones se separan
entre las que quedaron en días conservados y las que fueron eliminadas al
descartar su día.

## Resultado por año

Los conteos de filas y días se asignan según la hora que representa cada
etiqueta final.

| Año | Valores negativos | Noches completadas | Interpoladas conservadas | Interpoladas en días descartados | Horas aún faltantes | Días descartados |
|---|---:|---:|---:|---:|---:|---:|
| 2014 | 0 | 27 | 33 | 0 | 21 | 2 |
| 2015 | 0 | 664 | 56 | 8 | 571 | 62 |
| 2016 | 0 | 1 078 | 275 | 64 | 1 201 | 124 |
| 2017 | 0 | 218 | 21 | 0 | 209 | 20 |
| **Total** | **0** | **1 987** | **385** | **72** | **2 002** | **208** |

| Año | Días conservados | Filas finales | Filas diurnas | Celdas faltantes |
|---|---:|---:|---:|---:|
| 2014 | 363 | 8 712 | 4 631 | 0 |
| 2015 | 303 | 7 272 | 4 014 | 0 |
| 2016 | 242 | 5 808 | 3 147 | 0 |
| 2017 | 345 | 8 280 | 4 411 | 0 |
| **Total** | **1 253** | **30 072** | **16 203** | **0** |

El resultado tiene 17 columnas, desde `2014-01-01 01:00 UTC` hasta
`2018-01-01 00:00 UTC`, y `power_norm` está entre 0 y 0,981769. No quedan
celdas faltantes en el CSV final.

SHA-256 de `data/processed/dataset_ID003.csv`:

```text
39ae336061aac264dd9c76a1c02d25e38ba17a77b091350f55ca93acb1d68329
```
