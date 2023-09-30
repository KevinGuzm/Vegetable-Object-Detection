# Clasificación de Vegetales en Imágenes

Este proyecto tiene como objetivo desarrollar un modelo predictivo para determinar la localización y el tipo de vegetal presente en imágenes. Utilizaremos redes neuronales convolucionales (CNNs) y técnicas de transfer learning para procesar un conjunto de datos que contiene 118 imágenes de tres clases de vegetales: Mushrooms, Eggplant y Cucumber. Este proyecto es parte de una competencia en Kaggle.

## Introducción

El procesamiento de imágenes y la clasificación de objetos en imágenes son problemas desafiantes en el campo de la visión por computadora. En este proyecto, abordamos la clasificación de tres tipos de vegetales en imágenes: Mushrooms, Eggplant y Cucumber. Utilizamos redes neuronales convolucionales (CNNs) y transfer learning para crear un modelo de clasificación preciso.

## Conjunto de Datos

El conjunto de datos consta de 118 imágenes etiquetadas de tres clases de vegetales:

- Mushrooms
- Eggplant
- Cucumber

Cada imagen tiene anotaciones que indican la clase del vegetal y su ubicación en la imagen.

## Metodología

Nuestra metodología consta de los siguientes pasos:

1. **Exploración de Datos**: Comenzamos explorando el conjunto de datos para comprender su estructura y visualizar algunas de las imágenes.

2. **Preprocesamiento de Datos**: Realizamos el preprocesamiento necesario, que incluye la normalización de imágenes y la creación de etiquetas adecuadas para la clasificación.

3. **Entrenamiento de Modelos**: Utilizamos redes neuronales convolucionales (CNNs) y técnicas de transfer learning para entrenar modelos de clasificación.

4. **Evaluación del Modelo**: Medimos el rendimiento de nuestros modelos utilizando métricas como precisión, recall y F1-score.

5. **Generación de Predicciones**: Utilizamos el modelo entrenado para hacer predicciones en un conjunto de datos de prueba.

## Requisitos

Para ejecutar este proyecto, necesitas tener las siguientes bibliotecas de Python instaladas:

- PyTorch
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- Albumentations

## Instrucciones de Uso

1. Clona este repositorio en tu máquina local.
2. Asegúrate de tener todas las bibliotecas requeridas instaladas utilizando `pip install -r requirements.txt`.
3. Sigue el flujo de trabajo descrito en los Jupyter Notebooks proporcionados en la carpeta `notebooks`.
4. Ajusta hiperparámetros y arquitecturas de modelo según sea necesario para obtener los mejores resultados.
5. Una vez que estés satisfecho con el rendimiento del modelo, úsalo para hacer predicciones en el conjunto de datos de prueba.
6. Sube tus predicciones a la competencia de Kaggle y verifica tu puntaje.


## Resultados** 
Iteration #:  250
train_loss = 0.0
train_reg_loss = 0.0
train_cls_loss = 0.0
train_iou = 1.0
train_accuracy = 1.0
val_loss = 0.0
val_reg_loss = 0.0
val_cls_loss = 0.0
val_iou = 1.0
val_accuracy = 1.0


