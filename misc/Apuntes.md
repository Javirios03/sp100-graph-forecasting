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
