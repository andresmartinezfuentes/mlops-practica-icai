import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Explicación Dataset y Modelo")

st.title("Análisis del Dataset Breast Cancer y Modelo Random Forest")

# --- Sección 1: Explicación del dataset ---
st.header("1. Sobre el Dataset Breast Cancer")

CSV_PATH = "../data/breast_cancer.csv"
data = pd.read_csv(CSV_PATH)
feature_names = [c for c in data.columns if c != "target"]
X = data[feature_names].copy()
X = X.to_numpy()
y = data["target"].to_numpy()

st.write("Este dataset proviene de la biblioteca sklearn y contiene datos para diagnosticar cáncer de mama.")
st.write(f"El dataset tiene **{X.shape[0]} muestras** y **{X.shape[1]} características**.")
st.write("Las características son variables numéricas extraídas de imágenes de biopsias, y la variable objetivo indica benigno (0) o maligno (1).")

st.subheader("Primeras 5 filas del dataset:")
st.dataframe(X.head())

st.subheader("Distribución de la variable objetivo:")
st.bar_chart(y.value_counts())

# --- Sección 2: Breve explicación del modelo ---
st.header("2. Modelo: Random Forest Classifier")

st.write("""
El modelo usado es un **Random Forest**, que consiste en un conjunto de árboles de decisión para mejorar la precisión y evitar sobreajuste.

- Es un modelo supervisado para clasificación.
- Cada árbol aprende a partir de diferentes subconjuntos de datos.
- La predicción final es el resultado de la votación de los árboles.
""")

# --- Sección 3: Descripción del proceso de entrenamiento y evaluación ---
st.header("3. Proceso de Entrenamiento y Evaluación")

st.write("""
El proceso típico para entrenar y evaluar el modelo incluye:

- Dividir el dataset en conjunto de entrenamiento (por ejemplo, 75%) y test (25%).
- Entrenar el modelo Random Forest con los datos de entrenamiento.
- Evaluar el modelo con los datos de test, obteniendo métricas como accuracy, matriz de confusión y reporte de clasificación.
- Estos pasos permiten validar el desempeño del modelo antes de usarlo en producción.

*Nota:* En esta página se comenta el proceso; el entrenamiento y evaluación se realizan en un script aparte.
""")