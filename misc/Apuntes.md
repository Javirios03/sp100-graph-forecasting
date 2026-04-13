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

# Apéndices

## Interpretación de Resultados

- **F1-score**: Es la media armónica entre precision y recall, lo que nos da una medida equilibrada del desempeño del modelo, especialmente en casos de clases desbalanceadas. Un F1-score alto indica que el modelo tiene tanto una alta precisión (pocos falsos positivos) como un alto recall (pocos falsos negativos). Sin embargo, un F1-score bajo no indica directamente si el problema radica en la precisión o en el recall, por lo que es importante analizar ambas métricas por separado para entender mejor las fortalezas y debilidades del modelo.
- **F1-Macro CV**: Esta métrica se define como la media no-ponderada del F1-score de cada clase:
  $$F1\text{-}clase = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$
  $$F1\text{-}macro = \frac{F1\text{-}negativo + F1\text{-}neutral + F1\text{-}positivo}{3}$$

Macro significa que cada clase, independientemente de su frecuencia, tiene el mismo peso en la métrica final. Por último, el hecho de que esta métrica se calcule mediante validación cruzada (CV) nos da una medida más robusta del desempeño del modelo, ya que promedia los resultados obtenidos en diferentes particiones de los datos, lo que ayuda a mitigar el riesgo de overfitting y proporciona una estimación más generalizable del desempeño del modelo.

_Proposal_

# Componentes a Elegir

## Targets

1. El primer objetivo de este proyecto es clasificar el retorno semanal de las acciones en tres categorías: positivo, negativo o neutral. Para ello obtendríamos el retorno logarítmico acumulado de las próximas 5 sesiones de cada acción y lo clasificaríamos en función de su signo: $|ret| < \epsilon \rightarrow \text{neutral}$, $ret > \epsilon \rightarrow \text{positivo}$, $ret < -\epsilon \rightarrow \text{negativo}$, donde $\epsilon$ es un umbral que se determinará a través de la validación cruzada.

2. Si el primer objetivo es alcanzable y se obtiene un buen desempeño, se podría considerar un segundo objetivo más ambicioso: enfocar el problema, en vez desde un punto de vista de clasificación, desde un punto de vista de regresión. En este caso, el objetivo sería predecir la volatilidad futura de las acciones en el mismo horizonte temporal de 5 sesiones. Para ello, se estudiarían diferentes medidas de volatilidad, como la desviación estándar de los retornos diarios o _realized volatility_, y se elegiría la que mejor se adapte al problema.

## Dataset

Para evitar tener demasiados datos, se podría limitar el análisis a un conjunto de acciones representativas del mercado, como las que componen el índice S&P 100. Esto permitiría obtener una muestra suficientemente grande y diversa, pero sin llegar a ser inmanejable. Además, se podrían incluir datos de diferentes períodos de tiempo para capturar distintas condiciones de mercado.

## Tipo de Representación

Quedaría elegir si se usa un grafo estático o dinámico. Un grafo estático representaría las relaciones entre las acciones mediante una matriz de adyacencia constante (para todo el periodo), mientras que un grafo dinámico permitiría que estas relaciones cambien cada $X$ sesiones, capturando así la evolución temporal de las interacciones entre las acciones.

Por otro lado, también se podría considerar la posibilidad de usar un grafo heterogéneo, donde se representen diferentes tipos de relaciones entre las acciones (por ejemplo, correlación, co-movimiento, etc.) mediante diferentes tipos de aristas. Esto permitiría capturar una mayor riqueza de información sobre las interacciones entre las acciones.

En cuanto a la conexión entre nodos, existen distintas reglas que se podrían imponer:

- Conexiones únicamente sectoriales (acciones de la misma industria).
- Conexiones basadas en la correlación histórica de los retornos (por ejemplo, conectando solo aquellas acciones cuya correlación supere un cierto umbral).

## Features incluidas en los Nodos

1. Ventana de entrada: Longitud de historial que alimenta al modelo para predecir el retorno futuro. Se podrían probar diferentes longitudes (por ejemplo, 20, 60, 120 sesiones) para encontrar la que mejor se adapte al problema.
2. Tipo de features: Se podrían incluir tanto features técnicas (como medias móviles, RSI, etc.) como fundamentales (como ratios financieros, datos de balance, etc.) para capturar diferentes aspectos del comportamiento de las acciones.

## Arquitectura del Modelo

_Baselines_:

- **Random Forest** - Tabular sin grafo: Usando features agregadas por ventana de entrada.
- **LSTM** - Puramente Temporal: Usando secuencias de features sin considerar las relaciones entre acciones.

_Modelos con Grafo_:

- **GNN** + GRU/LSTM: Usando un grafo dinámico que capture las relaciones entre acciones a lo largo del tiempo, combinado con una capa recurrente para modelar la evolución temporal de las features.
- **Temporal GNN**: Usando una arquitectura de GNN que integre directamente la dimensión temporal, como T-GCN o EvolveGCN, para capturar tanto las relaciones entre acciones como su evolución a lo largo del tiempo de manera más integrada.

# Decisiones Principales

1. **Tipo de Grafo**: Se optará, en principio, por un grafo estático para simplificar el problema inicial, aunque se explorará la posibilidad de usar un grafo dinámico si el desempeño del modelo lo justifica.
2. **Conexiones entre Nodos**: Se establecerán conexiones tanto basadas en la pertenencia sectorial como en la correlación histórica de los retornos. Otra variante posible será incluir información geométrica de las distribuciones de retornos (por ejemplo, usando divergencias de Kullback-Leibler o Jensen-Shannon para medir la similitud entre las distribuciones de retornos de diferentes acciones).
3. **Ventana Temporal**: En principio se optará por una ventana de entrada de 20 sesiones, ya que teóricamente debería ser capaz de capturar tanto tendencias a corto plazo como patrones de comportamiento más estables.
4. **Features**: En un principio incluiremos unas pocas features para mantener el modelo relativamente simple:

- Precio de cierre ajustado.
- Volumen de negociación.
- Retorno logarítmico diario.
- Medias móviles (por ejemplo, de 5 y 20 sesiones).
- RSI (Relative Strength Index).
- Rolling volatility (desviación estándar de los retornos en la ventana de entrada).
- Sector al que pertenece cada acción (codificado como una variable categórica).
- Market cap (capitalización de mercado) para capturar el tamaño de la empresa.

5. **Arquitectura del Modelo**: Se comenzará con una arquitectura de GNN combinada con una capa recurrente (GRU o LSTM) para modelar tanto las relaciones entre acciones como su evolución temporal:

- GNN espacial sobre cada snapshot del grafo para capturar las relaciones entre acciones en cada punto temporal.
- Capa recurrente (GRU o LSTM) que consume la secuencia de embeddings generados por la GNN para modelar la evolución temporal de las features y las relaciones entre acciones a lo largo del tiempo.
- Capa final de clasificación (MLP) para predecir la categoría del retorno futuro (positivo, negativo o neutral).
