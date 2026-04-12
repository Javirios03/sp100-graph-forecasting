# Fases del Proyecto

1. Infraestructura y Datos
   - Obtención y preprocesamiento de datos: Recopilación de datos históricos de acciones - Uso de APIs financieras (Yahoo Finance, Alpha Vantage).
   - Definición del universo de acciones a analizar (S&P100).
   - Unificación del calendario, formatos y split de datos.
   - Generación del dataset tabular base sin estructura de grafo.
   * Decisiones importantes:
     - Extensión de los datos: Cuántos años de datos históricos se utilizarán para el entrenamiento y la evaluación.
     - Estabilidad de los tickers: El S&P100 puede cambiar con el tiempo, por lo que se debe decidir si se utilizará una lista fija de tickers o si se actualizará periódicamente.

2. Implementación de Baselines
   - Modelo Tabular: Implementación de un modelo de clasificación tradicional (Random Forest) utilizando solo las características tabulares.
   - Modelo Temporal Puro: Implementación de un modelo temporal (LSTM o GRU) utilizando solo las características temporales.
   - Evaluación de los modelos baselines, verificación de fuentes de aprendizaje y análisis de resultados.
   * Decisiones importantes:
     - Selección de librerías y frameworks para la implementación de los modelos: ¿Usamos PyTorch para todos los modelos o combinamos con scikit-learn para el modelo tabular?
     - Hiperparámetros iniciales: Definir los hiperparámetros iniciales para cada modelo (número de árboles para Random Forest, número de capas y unidades para LSTM/GRU).

3. Construcción de Grafos
   - Grafo estático + sector y correlación: Construcción de un grafo estático basado en la correlación histórica entre las acciones.
   - Grafo estático + divergencias: Construcción de un grafo estático basado en las divergencias entre las acciones.

4. Modelos gráficos
   - Implementación de modelo mínimo viable: GNN + capa temporal (LSTM/GRU) utilizando el grafo estático basado en correlación.
   - Evaluación del modelo mínimo viable, análisis de resultados y comparación con los baselines.

5. Experimentos Adicionales
   - Estudio de la volatilidad: Incorporación de la volatilidad como target principal y análisis de su impacto en el rendimiento del modelo.
   - Grafo dinámico: Exploración de la construcción de grafos dinámicos que evolucionen con el tiempo, capturando las relaciones cambiantes entre las acciones.
   - Análisis de la importancia de las características y las relaciones en el grafo para entender mejor qué factores influyen en el rendimiento del modelo.

# Aislamiento de Fallos

1. Datos:
   - Verificación de _data leakage_: Asegurarse de que no se utilicen datos futuros para predecir el pasado.
   - Análisis del split de datos: Verificar que el split entre entrenamiento, validación y prueba sea adecuado (conjuntos disjuntos) y no introduzca sesgos
   - Análisis de la calidad de los datos: Verificar la presencia de valores faltantes, outliers o inconsistencias en los datos y su impacto en el rendimiento del modelo.
   - Sanity check de los datos: Realizar análisis exploratorios para entender la distribución de las características y su relación con el target.

2. Etiquetas:
   - Confirmar que el horizonte de predicción es correcto y las etiquetas son consistentes con el objetivo del proyecto (predicción de la dirección del precio a 5 días).
   - Comprobar distribución de clases: Verificar que las clases estén balanceadas o aplicar técnicas de balanceo si es necesario.
   - Aplicación del umbral de clasificación: Analizar el impacto de diferentes umbrales de clasificación en las métricas de rendimiento (de cara a clases relativamente desbalanceadas).

3. Grafo:
   - Verificar la construcción del grafo: Asegurarse de que las relaciones entre los nodos (acciones) se construyan correctamente según la metodología definida (correlación, divergencia).
   - Análisis de la estructura del grafo: Evaluar la densidad, el grado de los nodos y otras métricas de grafos para entender su estructura y su impacto en el rendimiento del modelo.
   - Comprobación mediante muestra de vecinos: Comparar con la intuición financiera para verificar que las relaciones entre acciones en el grafo tengan sentido (por ejemplo, acciones del mismo sector deberían estar más conectadas).

4. Modelo Temporal:
   - Verificar la arquitectura del modelo temporal (LSTM/GRU) y su capacidad para capturar las dependencias temporales en los datos.
   - Análisis de la capacidad de generalización: Evaluar el rendimiento del modelo temporal en diferentes períodos de tiempo para verificar su capacidad de generalización.
   - Análisis de la importancia temporal: Evaluar qué períodos de tiempo son más relevantes para la predicción y cómo el modelo utiliza esta información.

5. Modelo GNN:
   - Test de overfitting en dataset pequeño
   - Aislamiento de causas de underfitting: Agregación espacial o temporal insuficiente, falta de capacidad del modelo, etc.
   - Comparación de output frente a identidad o MLP: Verificar que el modelo GNN esté aprendiendo relaciones útiles entre los nodos y no simplemente replicando la información de las características individuales.

6. Evaluación:
   - Verificar la correcta implementación de las métricas de evaluación (accuracy, precision, recall, F1-score) y su interpretación.
   - Verificar la correcta comparación entre baselines y modelos gráficos, asegurando que se utilicen los mismos conjuntos de datos y métricas para una comparación justa.

# Criterios Vitales

1. Si algo falla, sólo puede haber cambiado un componente respecto al experimento anterior. Esto implica que cada variante debe venir acompañada de un test de regresión mínimo que verifique que el nuevo componente no ha introducido errores en el sistema.
2. Si el modelo no mejora respecto a los baselines, se debe analizar exhaustivamente cada componente para identificar posibles causas (datos, etiquetas, grafo, modelo temporal, modelo GNN) y realizar ajustes iterativos hasta lograr una mejora significativa.
