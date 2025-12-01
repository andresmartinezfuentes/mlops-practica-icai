# Proyecto de Clasificación del Conjunto de Datos Breast Cancer 

ste repositorio contiene un proyecto de ejemplo de **Machine Learning** que utiliza un modelo de **Random Forest Classifier** para clasificar el famoso conjunto de datos de **Breast Cancer**. El proyecto incluye el código para entrenar el modelo, las dependencias necesarias y un archivo de datos de ejemplo El objetivo de este proyecto es proporcionar un despliegue de un modelo entrenado bajo el conjunto de datos mencionado. En la página web desplegada se puede comprobar las features más importantes, reentrenar el modelo basandote en ña importancia de las variables y obtener una predicción asi como una explicación local para un ejemplo en concreto propuesto por el usuario usando sliders. 

## Instalación

Para ejecutar este proyecto, necesitas tener **Python 3.x** instalado. El proyecto esta pensado para montarse en docker por lo que instalar las dependencias se hacen de manera interna.

## Uso

La idea general es usar un despliegue en google cloud, pero debido a que es de pago se recomienda montar lel contendor en local con el siguiente comando:

```bash
docker compose up
```

Adicionalmente para ver de forma sencilla el pipeline de explicabilidad hay un notebook en el que se comenat todo el proceso necesario para obtener las diferentes explicaciones en `src/notebooks/explanation.ipynb`

## Ficheros del Repositorio

A continuación se describen los archivos principales incluidos en este repositorio:

* **`src/train.py`**: Este script de Python carga el conjunto de datos Breast Cancer, lo divide en conjuntos de entrenamiento y prueba, entrena un modelo **Random Forest** y lo guarda como `models/model.pkl`.
* **`requirements.txt`**: Contiene todas las bibliotecas de Python necesarias para ejecutar el proyecto, incluyendo `scikit-learn`, `pandas`, `numpy`, `dvc` y `mlflow`.
* **`data/breas_cancer.csv`**: Un archivo CSV de ejemplo que representa una versión de los datos de Breast Cancer.
* **`docker-compose.yml`**: Un archivo yml que se encarga de crear el container.
