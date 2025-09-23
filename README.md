# Proyecto de Clasificación del Conjunto de Datos Iris

Este repositorio contiene un proyecto de ejemplo de **Machine Learning** que utiliza un modelo de **Random Forest Classifier** para clasificar el famoso conjunto de datos de **Iris**. El proyecto incluye el código para entrenar el modelo, las dependencias necesarias y un archivo de datos de ejemplo.

## Ficheros del Repositorio

A continuación se describen los archivos principales incluidos en este repositorio:

* **`train.py`**: Este script de Python carga el conjunto de datos Iris, lo divide en conjuntos de entrenamiento y prueba, entrena un modelo **Random Forest** y lo guarda como `model.pkl`.
* **`requirements.txt`**: Contiene todas las bibliotecas de Python necesarias para ejecutar el proyecto, incluyendo `scikit-learn`, `pandas`, `numpy`, `dvc` y `mlflow`.
* **`iris_dataset.csv`**: Un archivo CSV de ejemplo que representa una versión de los datos de Iris.

---

## Requisitos

Para ejecutar este proyecto, necesitas tener **Python 3.x** instalado. Se recomienda usar un entorno virtual para gestionar las dependencias.

Puedes instalar las bibliotecas necesarias usando el siguiente comando:

```bash
pip install -r requirements.txt
