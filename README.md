# Laboratorio 02 de Deep Learning

Clasificación multiclase y ordinal del deterioro cognitivo mediante redes convolucionales unidimensionales y CORAL.

**Autores:** Pablo Cortés y Kevin Silva  
**Profesor:** Dr. Juan Bekios Calfa  
**Universidad Católica del Norte**

## Descripción

El laboratorio evalúa tres configuraciones para predecir la etapa de deterioro cognitivo a partir de 15 respuestas binarias de un test de orientación. Cada registro se representa como una secuencia de un canal y 15 posiciones. Las tres configuraciones comparten un extractor Conv1D seguido de un MLP.

| Configuración            | Cabeza de clasificación     | Función de pérdida                                                                 |
| ------------------------ | --------------------------- | ---------------------------------------------------------------------------------- |
| Conv1D + Softmax         | \(K\) logits, uno por clase | Entropía cruzada categórica                                                        |
| Conv1D + CORAL           | \(K-1\) umbrales ordinales  | Entropía cruzada binaria acumulada por umbral                                      |
| Conv1D + CORAL con pesos | \(K-1\) umbrales ordinales  | Pérdida CORAL ponderada por el número efectivo de muestras, con \(\beta = 0{,}99\) |

Se evaluaron las tres configuraciones en seis objetivos (`GDS` y `GDS_R1` a `GDS_R5`), para un total de **18 experimentos** sobre 1119 registros.

## Instalación

Desde la raíz del repositorio, crea y activa el entorno Conda:

```bash
conda env create -f environment.yml
conda activate lab_pytorch
```

Copia el archivo `.sav` proporcionado en el laboratorio a `dataset/dataset.sav`. Si lo guardas con otro nombre, ajusta el argumento `--data-path` en los comandos.

## Reproducir los experimentos

Ejecuta los comandos desde la raíz del repositorio. Cada variante escribe sus resultados en un directorio distinto.

```bash
# Conv1D + Softmax
python main.py --data-path dataset/dataset.sav --all-targets --algorithm conv1d_softmax --epochs 20 --device cpu --seed 42 --output-dir results/replica_softmax

# Conv1D + CORAL
python main.py --data-path dataset/dataset.sav --all-targets --algorithm conv1d_coral --epochs 20 --device cpu --seed 42 --output-dir results/replica_coral

# Conv1D + CORAL con pesos
python main.py --data-path dataset/dataset.sav --all-targets --algorithm conv1d_coral --use-weights --epochs 20 --device cpu --seed 42 --output-dir results/replica_coral_pesos
```

Para `GDS_R1` a `GDS_R5` se utiliza validación cruzada anidada estratificada con cinco pliegues externos y tres internos. La búsqueda selecciona la configuración con menor MAE ordinal promedio y usa el mayor QWK para desempatar. Para `GDS` original se usan dos pliegues externos y la configuración base, ya que la clase terminal contiene solo dos registros y cada conjunto de entrenamiento externo queda con uno.

## Resultados

Las ejecuciones generan métricas en CSV y Markdown, matrices de confusión y registros de selección de hiperparámetros. Las métricas se resumen como media y desviación estándar de los pliegues externos. El CSV consolidado que respalda las tablas del informe se encuentra en `results/bateria_lab2/resultados_consolidados.csv` dentro del paquete de resultados.

En `GDS_R4`, CORAL con pesos obtuvo la mayor exactitud balanceada (0,6827), mientras que CORAL sin pesos obtuvo el mayor F1 macro (0,5984). En `GDS_R1`, Softmax obtuvo el mayor F1 macro (0,7493). En `GDS` original, CORAL con pesos aumentó la exactitud balanceada de 0,3458 a 0,4422 frente a Softmax, con un aumento de los errores graves de 3,75 % a 10,90 %.

## Estructura del repositorio

| Ruta              | Contenido                                                                 |
| ----------------- | ------------------------------------------------------------------------- |
| `main.py`         | Interfaz de línea de comandos para ejecutar los experimentos              |
| `src/config.py`   | Configuración y espacio de búsqueda de hiperparámetros                    |
| `src/data/`       | Carga de datos, codificación de etiquetas y preparación de las secuencias |
| `src/models/`     | Bloques convolucionales, extractor compartido y cabezas Softmax y CORAL   |
| `src/tasks/`      | Funciones de pérdida, ponderación y predicción                            |
| `src/training/`   | Entrenamiento, búsqueda interna y validación cruzada                      |
| `src/evaluation/` | Métricas multiclase y ordinales                                           |
| `src/reporting/`  | Tablas, gráficos y matrices de confusión                                  |
| `environment.yml` | Definición del entorno Conda                                              |