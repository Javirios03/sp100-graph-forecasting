# Variables

Tenemos cuatro modelos: Tabular puro (Random Forest), Temporal puro (LSTM) y dos variantes con grafo (difiriendo únicamente en los datos usados, uno usando sólo correlación y sectorial, y otro usando también información geométrica de las distribuciones de retornos).

Por lo tanto, debemos generar dos bloques de variables:

1. Features y targets definidos por acción y por día
2. Aquellas definidas mediante pares de acciones y por día (para el caso de los modelos con grafo).

## Comunes a todos los modelos:

- Identificadores de la acción y del día:
  - Ticker: Identificador único de cada acción.
    - Date: Fecha correspondiente a cada registro de datos.
- Precio y Retornos:
  - Adjusted Close Price: Precio de cierre ajustado.
  - Daily Log Return: Retorno logarítmico diario, calculado como $\log\left(\frac{P_t}{P_{t-1}}\right)$, donde $P_t$ es el precio de cierre ajustado en el día $t$.
  - Weekly Log Return: Retorno logarítmico semanal, calculado como $\log\left(\frac{P_{t+5}}{P_t}\right)$, donde $P_{t+5}$ es el precio de cierre ajustado cinco días después del día $t$.
  - Class: Variable categórica que indica si el retorno semanal es positivo, negativo o neutral, definida en función del signo del Weekly Log Return y un umbral $\epsilon$.
- Volumen y Escala:
  - Volume: Volumen de negociación diario.
  - Volume Normalized: Volumen de negociación normalizado, calculado como $\frac{Volume_t - \mu_{Volume}}{\sigma_{Volume}}$, donde $\mu_{Volume}$ y $\sigma_{Volume}$ son la media y desviación estándar del volumen de negociación para esa acción en el período de entrenamiento.
- Indicadores Técnicos:
  - Moving Average 5: Media móvil de 5 días del precio de cierre ajustado, normalizada por precio actual
  - Moving Average 20: Media móvil de 20 días del precio de cierre ajustado, normalizada por precio actual.
  - RSI: Relative Strength Index, calculado sobre una ventana de 14 días.
  - Rolling Volatility: Desviación estándar de los retornos diarios en la ventana de 20 días
- Información Estructural:
  - Sector: Sector al que pertenece cada acción, codificado como una variable categórica.
  - Market Cap: Capitalización de mercado de la empresa, calculada como el precio de cierre ajustado multiplicado por el número de acciones en circulación, y normalizada por el máximo valor de capitalización de mercado en el período de entrenamiento.

## Específicas para modelos temporales

- Secuencia de features: Para cada acción, se generará una secuencia de las features anteriores (ajustadas para cada día) con una longitud de 20 días, que servirá como entrada para los modelos temporales y con grafo.

## Específicas para modelos con grafo

- Correlaciones: Para cada par de acciones, se calculará la correlación histórica de sus retornos diarios en una ventana de 1 año, y se establecerá una conexión entre las acciones si la correlación supera un cierto umbral $\rho$.
- Conexiones sectoriales: Se establecerá una conexión entre dos acciones si pertenecen al mismo sector. Dicho conjunto de sectores está todavía por definir, pero se podrían usar categorías estándar como Tecnología, Salud, Finanzas, etc.
- Información geométrica de las distribuciones de retornos: Para cada par de acciones, se calculará una medida de similitud entre las distribuciones de sus retornos diarios, como la divergencia de Kullback-Leibler o la divergencia de Jensen-Shannon, y se establecerá una conexión entre las acciones si esta medida indica una similitud significativa (por ejemplo, si la divergencia es menor que un cierto umbral $\delta$).

# Modelos Implementados

- Random Forest: Un modelo de clasificación basado en árboles de decisión, que se entrenará usando las features tabulares definidas anteriormente. Este modelo servirá como baseline para evaluar el desempeño de los modelos más complejos.
- LSTM: Un modelo de red neuronal recurrente, que se entrenará usando las secuencias de features definidas para cada acción. Este modelo permitirá capturar patrones temporales en los datos, lo que podría mejorar su capacidad para predecir el comportamiento futuro de las acciones.
- GGN: De cara a elegir la arquitectura concreta, la primera decisión es qué paradigma de GNN usar. Como hemos ido aprendiendo a lo largo del semestre, tanto Convolutional como Attention-based GNNs se pueden interpretar como casos particulares de un paradigma más general, el de Message-Passing. Por lo tanto, la elección se reduce a comparar el trade-off entre expresividad y eficiencia computacional.
  (1) Por un lado, Convolutional GNNs, como GraphSAGE o GCN, son relativamente eficientes desde el punto de vista computacional, pero asumen que todos los vecinos de un nodo contribuyen de manera similar a su representación, lo que podría no ser el caso en nuestro problema, dado que algunas conexiones (por ejemplo, aquellas basadas en correlación) podrían ser más relevantes que otras (por ejemplo, aquellas basadas en sector).
  (2) Por otro lado, si bien Message-Passing podría aportar una gran expresividad al modelo, especialmente por la presencia de atributos como la correlación o la divergencia de distribuciones de retorno, podría ser excesivamente complejo para una primera aproximación al problema, especialmente teniendo en cuenta que estamos usando un grafo estático con las mismas acciones durante todo el período de tiempo.
  (3) Por lo tanto, decidimos en un principio usar una arquitectura de tipo Attention-based GNN. Específicamente, **GAT + LSTM**, para evitar tener que ajustar manualmente los pesos de las conexiones basadas en correlación o similitud geométrica, y permitir que el modelo aprenda a asignar diferentes pesos a diferentes vecinos en función de su relevancia para la tarea de predicción.

LSTM por nodo -> GAT -> MLP -> Predicción

Centrándonos en el modelo GNN, cabe destacar cómo evolucionan los datos conforme los ingiere el modelo:

1. Input - X: (T, N, W, F) con T = número de días/snapshots (1168), N = número de nodos (88), W = ventana temporal (20), F = número de features por nodo (7). Y: (T, N) con la clase a predecir para cada acción y cada día. Cada elemento de X es, por lo tanto, un grafo completo correspondiente a un día concreto, con 88 nodos (acciones) y 7 features por nodo, que representan la información relevante de cada acción en los últimos 20 días.

## GNN

Para obtener el mejor rendimiento posible (y teniendo en cuenta que, dada la complejidad de la arquitectura, es más importante en este case hacer una búsqueda razonable de hiperparámetros), decidimos hacer una búsqueda de hiperparámetros manual. Para ello, definimos un _grid search_ con los siguientes hiperparámetros y valores:

- Learning Rate: [1e-4, 5e-4, 1e-3]
- Weight Decay: [0, 1e-4, 1e-3]
- Epochs: [20, 40]
- LSTM Hidden Units: [32, 64]
- GAT Hidden Units: [32, 64]

# Rango Temporal

En este caso concreto, tenemos un tradeoff importante entre la cantidad de datos disponibles y la relevancia de los mismos. Por un lado, cuanto más largo sea el período de tiempo considerado, más datos tendremos para entrenar el modelo, lo que podría mejorar su capacidad de generalización. Sin embargo, también es cierto que los datos más antiguos podrían no ser tan relevantes para predecir el comportamiento futuro de las acciones. Adicionalmente, en un plazo mayor habrá un mayor numero de cambios estructurales en el mercado (específicamente, cambios de composición del índice), lo que podría dificultar el aprendizaje de patrones consistentes a lo largo del tiempo. SOTA en forecasting de series temporales financieras suele usar un rango temporal de entre 3 y 5 años con splits típicos de 70/15/15 para entrenamiento, validación y test respectivamente, por lo que se podría considerar un rango temporal de 5 años para este proyecto, aunque se podrían probar también rangos más cortos (por ejemplo, 3 años) para evaluar su impacto en el desempeño del modelo.

Teniendo esto en cuenta, consideraremos un rango temporal de 5 años para este proyecto. En concreto, es importante la elección de la fecha de inicio, ya que esto determinará el período de tiempo para el cual se dispondrá de datos. Sabemos que la pandemia de COVID-19 tuvo un impacto significativo en los mercados financieros, por lo que, para dotar de mayor generalización al modelo, consideraremos como inicio del período de tiempo un punto en el que podamos asumir, razonablemente, que el mercado ya se había adaptado a las condiciones post-pandemia (habiéndose normalizado, relativamente, la volatilidad y los patrones de comportamiento del mercado). Sabemos, además, que la recuperación y normalización fue extremadamente asimétrica entre sectores. Esto puede ser un beneficio en nuestra tarea, ya que incluir la propia variable de sector en el modelo permitirá capturar esta asimetría y su impacto en las relaciones entre acciones. Por lo tanto, se podría considerar como fecha de inicio el 1 de enero de 2021 (siendo por lo tanto el final del período de tiempo el 31 de diciembre de 2025). Esta elección también nos proporcionaría un margen de datos (aquellos pertenecientes al resto de 2026 hasta la fecha actual) para evaluar la capacidad de generalización del modelo a datos futuros, lo que sería un test adicional de su desempeño.

# Universo de Acciones

Hemos desarrollado un pipeline para obtener la composición histórica del índice S&P 100, accediendo al historico de revisiones de su artículo en Wikipedia (puede ser encontrado en `src/eda/sp100_components.ipynb`). Según este pipeline, el número de acciones que componen el índice varía periódicamente. Sin embargo, existen un conjunto de 88 acciones que han formado parte del índice durante todo el período de tiempo considerado (2021-2025). Por lo tanto, optamos en principio por limitar el universo de acciones a estas 88, ya que esto nos evitará tener que lidiar con problemas de datos faltantes o inconsistentes para aquellas acciones que solo formaron parte del índice durante una parte del período de tiempo. Además, estas 88 acciones representan una muestra suficientemente grande y diversa del mercado, lo que debería ser suficiente para entrenar un modelo robusto y generalizable.

# Apuntes Importantes

## Información Sectorial

Disponemos de 88 empresas que pertenecen a 10 sectores diferentes. Dichos sectores, eso sí, no están distribuidos de manera uniforme, sino que algunos sectores tienen una representación mucho mayor que otros:
| Sector | Número de Empresas |
| --- | --- |
| Financial Services | 16 |
| Healthcare | 14 |
| Technology | 12 |
| Industrials | 12 |
| Consumer Defensive | 10 |
| Consumer Cyclical | 9 |
| Communication Services | 7 |
| Energy | 3 |
| Utilities | 3 |
| Real Estate | 2 |

En principio, usaremos esta información sectorial. Sin embargo, cabe preguntarse qué ocurrirá en empresas pertenecientes a sectores poco representados (por ejemplo, Real Estate), que tenderán a ser menos conectada en el grafo (si se opta por usar conexiones sectoriales) y, además, tendrán menos datos disponibles para aprender patrones específicos de su sector. Esto podría afectar negativamente al desempeño del modelo para estas empresas. Por lo tanto, se podrían considerar algunas estrategias para mitigar este problema, como por ejemplo:

- Agrupar sectores poco representados en una categoría "Otros", lo que permitiría aumentar la representación de esta categoría y facilitar el aprendizaje de patrones comunes entre estas empresas.
- Usar técnicas de data augmentation para generar datos sintéticos para estas empresas, lo que podría ayudar a mejorar la capacidad del modelo para generalizar a estas empresas, aunque esto también podría introducir ruido si no se hace de manera cuidadosa

## Creación de Nuevas Features

### Target: Movimiento Semanal

Inicialmente, imponemos un $\epsilon$ de 0.01 para clasificar los retornos semanales como positivos, negativos o neutrales. Sin embargo, este umbral es arbitrario y podría no ser el óptimo para nuestro problema. Obtenemos para dicho valor que tenemos:
| Clase | Porcentaje de Muestras |
| --- | --- |
| Positivo | 41.29% |
| Negativo | 33.86% |
| Neutral | 24.85% |

Estos porcentajes se podrían alinear con una intuición clave sobre el mercado: ligero sesgo alcista en los retornos semanales, lo que es consistente con la literatura financiera que muestra que, a largo plazo, los mercados tienden a subir. Sin embargo, también es importante considerar que este sesgo podría variar dependiendo del período de tiempo considerado y de las condiciones del mercado. En etapas posteriores, usaremos validación cruzada para ajustar este umbral $\epsilon$ y encontrar el valor que maximice el desempeño del modelo, lo que nos permitirá obtener una clasificación más equilibrada entre las clases y mejorar la capacidad de generalización del modelo.

## Splits

Disponemos de 5 años de datos. Decidimos inicialmente usar un split temporal de 3, 1 y 1 años para entrenamiento, validación y test respectivamente. Esto supone, en porcentajes, un split de 60%, 20% y 20%. Podría resultar insuficiente dedicar sólo el 60% de los datos al entrenamiento, pero nos parece que dedicar periodos relativamente similares a cada conjunto de datos (en cuanto a que todos ellos contienen año/s natural/es) es importante para evitar problemas de generalización a datos futuros. Además, el hecho de tener un año completo para validación y otro para test nos permitirá evaluar el desempeño del modelo en condiciones de mercado diferentes, lo que también es importante para garantizar su robustez y capacidad de generalización.

# Resultados

## Baseline: Random Forest

Obtenemos un F1-macro CV (con 5 folds) en train de $0.393 \pm 0.008$, lo que es un resultado bastante decente para un modelo tan simple. Adicionalmente, en el conjunto de validación tenemos las siguientes estadísticas según clase:
| Clase | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| Negativo | 0.333 | 0.31 | 0.32 | 7089 |
| Neutral | 0.31 | 0.5 | 0.38 | 6009 |
| Positivo | 0.43 | 0.27 | 0.33 | 9077 |

Disponemos asimismo de la siguiente matriz de confusión:
| | Pred. Negativo | Pred. Neutral | Pred. Positivo |
| --- | --- | --- | --- |
| Real Negativo | 2194 | 3045 | 1850 |
| Real Neutral | 1531 | 3031 | 1447 |
| Real Positivo | 2827 | 3801 | 2449 |

La principal ventaja de Random Forest, por otro lado, es que nos permite obtener, sencillamente, la importancia de cada feature para la predicción. Las 10 features más importantes, según el modelo, son las siguientes:
| Feature | Importancia |
| --- | --- |
| roll_vol_20_min_20 | 0.0466 |
| roll_vol_20_max_20 | 0.0386 |
| roll_vol_20_mean_20 | 0.0381 |
| roll_vol_20_last | 0.0380 |
| log_ret_1d_min_20 | 0.0376 |
| log_ret_1d_max_20 | 0.0344 |
| ma_5_std_20 | 0.0336 |
| log_ret_1d_std_20 | 0.0333 |
| ma_5_min_20 | 0.0317 |
| ma_20_max_20 | 0.0316 |

El baseline más básico que podríamos considerar es un modelo que asignara, de manera aleatoria, una clase a cada muestra, respetando la distribución de clases en el conjunto de datos. En este caso, el F1-macro CV esperado sería aproximadamente de 0.33, lo que significa que nuestro modelo Random Forest ya está superando este baseline básico, lo cual es un buen indicio de que está aprendiendo patrones útiles en los datos. Sin embargo, también es importante considerar que un F1-macro CV de 0.393 no es un resultado excepcionalmente alto, lo que sugiere que todavía hay margen de mejora, especialmente teniendo en cuenta que estamos usando un modelo relativamente simple. Por lo tanto, es importante seguir explorando diferentes arquitecturas de modelos, así como ajustar los hiperparámetros y las features utilizadas, para intentar mejorar el desempeño del modelo y obtener resultados más robustos y generalizables.

Un par de insights interesantes que se pueden extraer de la importancia de las features son los siguientes:

- La importancia de las features relacionadas con la volatilidad (roll_vol_20_min_20, roll_vol_20_max_20, roll_vol_20_mean_20, roll_vol_20_last) sugiere que la volatilidad reciente de las acciones es un factor clave para predecir su comportamiento futuro. Esto podría indicar, por ejemplo, que se guía especialmente por una baja volatilidad (lo que podría ser un indicio de estabilidad) para predecir retornos neutrales.
- En conjunto, podemos concluir que por un lado la volatilidad confunde la dirección, pero la variabilidad de volatilidad predice la dirección mejor que el precio o los retornos, lo que es un insight interesante sobre la naturaleza del problema y podría guiar la selección de features y la arquitectura del modelo en etapas posteriores.

## Baseline: LSTM

Entrenamos durante 30 epochs un modelo LSTM con una arquitectura relativamente simple (una capa LSTM bidireccional con 128 unidades y dropout del 20%, seguida de 3 capas fully connected con 64, 32 y 3 unidades respectivamente, y activación ReLU en las capas intermedias), usando AdamW como optimizador, una tasa de aprendizaje de 0.0005 y _weight_decay_ de 0.01. Obtenemos un F1-macro CV de $0.323$, lo que es un resultado bastante decepcionante, dado que es incluso inferior al baseline de Random Forest.

# Apéndices

## Interpretación de Resultados

- **F1-score**: Es la media armónica entre precision y recall, lo que nos da una medida equilibrada del desempeño del modelo, especialmente en casos de clases desbalanceadas. Un F1-score alto indica que el modelo tiene tanto una alta precisión (pocos falsos positivos) como un alto recall (pocos falsos negativos). Sin embargo, un F1-score bajo no indica directamente si el problema radica en la precisión o en el recall, por lo que es importante analizar ambas métricas por separado para entender mejor las fortalezas y debilidades del modelo.
- **F1-Macro CV**: Esta métrica se define como la media no-ponderada del F1-score de cada clase:
  $$F1\text{-}clase = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$
  $$F1\text{-}macro = \frac{F1\text{-}negativo + F1\text{-}neutral + F1\text{-}positivo}{3}$$

Macro significa que cada clase, independientemente de su frecuencia, tiene el mismo peso en la métrica final. Por último, el hecho de que esta métrica se calcule mediante validación cruzada (CV) nos da una medida más robusta del desempeño del modelo, ya que promedia los resultados obtenidos en diferentes particiones de los datos, lo que ayuda a mitigar el riesgo de overfitting y proporciona una estimación más generalizable del desempeño del modelo.

## Pipeline de Datos

Los datos originales son extraídos de Yahoo Finance usando la librería `yfinance`. Primero, en el script `src/data/01_load_clean_sp100.py`, se usa la API de `yfinance` para descargar los datos históricos de las acciones (sólo las 88 estables) desde el 1 de enero de 2021 hasta el 31 de diciembre de 2025. El resultado de este script es el archivo `data/raw/prices_raw.parquet`:
| date | ticker | adj_close | close | high | low | open | volume |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-01-04 | AAPL | 125.8566 | 129.41 | 133.6116 | 126.76 | 133.5204 | 143301900 |
Todas estas columnas están en formato float64, excepto `date` (datetime64[ms]), `ticker` (string) y `volume` (int64).

A continuación, en el script `src/data/02_build_metadata.py`, se obtienen los datos específicos de cada acción, guardándose en el archivo `data/raw/ticker_metadata.parquet`:
| ticker | sector | industry | market_cap | short_name | currency |
| --- | --- | --- | --- | --- | --- |
| AAPL | Technology | Consumer Electronics | 3776182222848 | Apple Inc. | USD |
Todas estas columnas están en formato string, excepto `market_cap` (int64).

El tercer paso del pipeline, implementado en el script `src/data/03_build_base_dataset.py`, consiste en combinar los datos de precios con los datos de metadata para generar un dataset base que contenga toda la información relevante para cada acción y cada día. El resultado de este script es el archivo `data/processed/base_panel.parquet`, que tiene la siguiente estructura:
| date | ticker | adj_close | close | high | low | open | volume | sector | industry | market_cap | short_name | currency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Los tipos de las columnas son acordes a los tipos de las columnas en los archivos originales, es decir, `date` (datetime64[ms]), `ticker`, `sector`, `industry`, `short_name` y `currency` (string), `adj_close`, `close`, `high`, `low`, `open` (float64) y `volume` y `market_cap` (int64).

El cuarto paso supone la creación de nuevas features a partir del dataset base (`src/data/04_compute_features.py`), que usa el dataset `data/processed/base_panel.parquet` para generar el dataset intermedio `data/interim/features_panel.parquet`, que contiene un total de
| Column | Type | Example |
| --- | --- | --- |
| date | datetime64[ms] | 2021-01-04 |
| ticker | string | AAPL |
| adj_close | float64 | 125.8566 |
| close | float64 | 129.41 |
| high | float64 | 133.6116 |
| low | float64 | 126.76 |
| open | float64 | 133.5204 |
| volume | int64 | 143301900 |
| sector | string | Technology |
| industry | string | Consumer Electronics |
| market_cap | int64 | 3776182222848 |
| short_name | string | Apple Inc. |
| currency | string | USD | (dropped posteriormente) |
| log_ret_1d | float64 | 0.0288 |
| adj_close_ret_1d | float64 | 0.01236 |
| ma_5 | float64 | 126.45 |
| ma_20 | float64 | 122.34 |
| volume_ma_20 | float64 | 120000000 |
| volume_norm | float64 | 0.5 |
| roll_vol_20 | float64 | 0.02 |
| rsi_14 | float64 | 70.5 |

El quinto paso consiste en añadir al conjunto anterior los targets. Para ello, en el script `src/data/05_compute_targets.py`, se calcula el retorno logarítmico semanal acumulado para cada acción y cada día, y se clasifica en positivo, negativo o neutral en función de su signo y un umbral $\epsilon$ iniciado en 0.01. El resultado de este paso es el dataset final `data/processed/panel_with_targets.parquet`, que tiene la misma estructura que el dataset intermedio, pero con dos columnas adicionales:
| Column | Type | Example |
| --- | --- | --- |
| future_log_ret_5d | float64 | 0.015 |
| target_class | float64 | 1.0 |
donde `future_log_ret_5d` es el retorno logarítmico acumulado de los próximos 5 días, y `target_class` es la clase asignada a dicho retorno (1 para positivo, -1 para negativo y 0 para neutral).

El sexto paso del pipeline consiste en añadir otra columna relativa al conjunto al que pertenece cada muestra (train, validación o test), en función de su fecha. Para ello, se usa el script `src/data/06_build_splits.py`, que toma el dataset `data/processed/panel_with_targets.parquet` y genera el dataset `data/processed/panel_with_splits.parquet`, que tiene la misma estructura que el dataset anterior, pero con una columna adicional:
| Column | Type | Example |
| --- | --- | --- |
| split | string | train |
donde `split` indica el conjunto al que pertenece cada muestra, siendo "train" para muestras con fecha entre el 1 de enero de 2021 y el 31 de diciembre de 2023, "validation" para muestras con fecha entre el 1 de enero de 2024 y el 31 de diciembre de 2024, y "test" para muestras con fecha entre el 1 de enero de 2025 y el 31 de diciembre de 2025.

El séptimo paso consiste en la creación del conjunto de datos tabular (el que alimentaremos al modelo Random Forest), que se obtiene a partir del dataset `data/processed/panel_with_splits.parquet` mediante el script `src/data/07_build_tabular_dataset.py`, que genera el dataset `data/processed/tabular_dataset.parquet`, con 41 columnas: `ticker`, `date`, `split`, `target_class`, `sector`, `market_cap` y el resto siguen el patrón `FEATURE_STATISTIC_20`, donde `FEATURE` es el nombre de la feature original (`adj_close`, `volume_norm`, `log_ret_1d`, `ma_5`, `ma_20`, `rsi_14` o `roll_vol_20`), `STATISTIC` es una estadística calculada sobre una ventana de 20 días (`mean`, `std`, `min`, `max` o `last`), y el número 20 indica la longitud de la ventana usada para calcular dicha estadística. No se usa OHE para la variable `sector`, sino que se codifica como una variable categórica, dado que Random Forest puede manejar variables categóricas de manera eficiente sin necesidad de codificación adicional.

El octavo paso del pipeline consiste en la creación del conjunto de datos para el modelo temporal (el que alimentaremos al modelo LSTM), que se obtiene a partir del dataset `data/processed/panel_with_splits.parquet` mediante el script `src/data/08_build_temporal_dataset.py`, que genera dos archivos NPY: `data/processed/X_temporal.npy` y `data/processed/y_temporal.npy`, así como un archivo de metadata `data/processed/temporal_metadata.parquet`. El archivo parquet tiene la siguiente estructura:
| sample_id | ticker | date | split | target_class |
| --- | --- | --- | --- | --- |
| 0 | AAPL | 2021-03-02 | train | -1.0 |
donde `sample_id` es un identificador único para cada muestra. El archivo `X_temporal.npy` tiene una forma de (N, 20, 7), donde N es el número total de muestras (106438, en este caso), 20 es la longitud de la secuencia temporal (ventana de entrada) y 7 es el número de features originales (adj_close, volume_norm, log_ret_1d, ma_5, ma_20, rsi_14 y roll_vol_20). El archivo `y_temporal.npy` tiene una forma de (N,), donde cada elemento es la clase objetivo correspondiente a cada muestra.

El noveno paso es la creación del grafo, que se obtiene a partir del dataset `data/processed/panel_with_splits.parquet` mediante el script `src/data/09_build_graph.py`, que genera tres archivos: `data/processed/ticker_to_node.parquet`, `data/processed/graph_corr_edges.parquet` y `data/processed/graph_div_edges.parquet`. El primer archivo tiene la siguiente estructura:
| ticker | node_id |
| --- | --- |
| AAPL | 0 |
donde `node_id` es un identificador numérico único para cada acción, que se usará para construir el grafo. El segundo archivo tiene la siguiente estructura:
| src | dst | src_ticker | dst_ticker | weight | edge_type |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 58 | AAPL | MSFT | 0.722713 | correlation |
donde `src` y `dst` son los identificadores numéricos de las acciones conectadas por la arista, `src_ticker` y `dst_ticker` son los tickers correspondientes a dichas acciones, `weight` es el peso de la arista (un número entre 0.3272 y 1 para el caso de las aristas basadas en correlación, y 1 para el caso de las aristas basadas en sector), y `edge_type` indica el tipo de arista (correlation o sector). El tercer archivo tiene la misma estructura que el segundo, pero con una columna adicional `distance`, que indica la medida de similitud geométrica entre las distribuciones de retornos de las acciones conectadas por la arista, calculada mediante la divergencia de Jensen-Shannon. En dicho caso, la columna `weight` tiene rango entre 0.85 y 1 para las aristas basadas en similitud geométrica, y es 1 para las aristas basadas en sector. Por otro lado, la columna `distance` tiene rango entre 0.013 y 0.176 (con media de 0.0458) para las aristas basadas en similitud geométrica, y es NAN (IMPORTANTE) para las aristas basadas en sector, dado que estas no se basan en una medida de similitud geométrica.

El último script, `src/data/10_build_gnn_dataset.py`, lee los archivos `data/processed/panel_with_splits.parquet` y `data/processed/ticker_to_node.parquet` para generar tres archivos: `data/processed/X_gnn.npy`, `data/processed/y_gnn.npy` y `data/processed/gnn_snapshots_index.parquet`. El archivo `X_gnn.npy` tiene una forma de (1168, 88, 20, 7), donde 1168 es el número total de muestras (correspondiente a la cantidad de días únicos en el conjunto de datos), 88 es el número de nodos (acciones) en el grafo, 20 es la longitud de la secuencia temporal (ventana de entrada) y 7 es el número de features originales. El archivo `y_gnn.npy` tiene una forma de (1168, 88), donde cada elemento es la clase objetivo correspondiente a cada muestra y cada nodo. El archivo `gnn_snapshots_index.parquet` tiene la siguiente estructura:
| snapshot_id | date | split | num_nodes |
| --- | --- | --- | --- |
| 20 | 2021-03-02 | train | 88 |
donde `snapshot_id` es un identificador único para cada snapshot del grafo (correspondiente a cada día único en el conjunto de datos), `date` es la fecha correspondiente a cada snapshot, `split` indica el conjunto al que pertenece cada snapshot (train, validación o test) y `num_nodes` es el número de nodos presentes en cada snapshot (que es 88 para todos los snapshots, dado que estamos usando un grafo estático con las mismas acciones durante todo el período de tiempo).

## Estructura de los Grafos

Conceptualmente, el resultado de este proceso de construcción del grafo es un conjunto de 1168 snapshots de un grafo con 88 nodos (correspondientes a las 88 acciones). Dicho grafo (o más bien, sus nodos) no cambia a lo largo del tiempo, dado que estamos usando un grafo estático, pero las aristas sí pueden cambiar, por lo que lo podemos considerar estático en estructura pero dinámico en señales. Para cada fecha _t_, tenemos un snapshot del grafo con señales
$$X_t \in \mathbb{R}^{88 \times 20 \times 7}, y_t \in \mathbb{R}^{88}$$
donde $X_t$ es la matriz de features para cada nodo (acción) en el día _t_, y $y_t$ es el vector de clases objetivo para cada nodo en el día _t_.

**spatio-temporal node classification on a static graph with dynamic node features**
