import streamlit as st
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='XAI Interactive — Breast Cancer', layout='wide')
st.title('XAI Interactivo: Entrena, explica y experimenta con features')

# Endpoint config
API_XAI = os.environ.get('API_URL_XAI', 'http://mlops-api-pf:5000/xai_example')
API_EVALUATE = os.environ.get('API_URL_EVALUATE', 'http://mlops-api-pf:5000/evaluate_model')
API_FEATURES = os.environ.get('API_URL_FEATURES', 'http://mlops-api-pf:5000/feature_names')
API_RETRAIN = os.environ.get('API_URL_RETRAIN', 'http://mlops-api-pf:5000/retrain')
API_PREDICT = os.environ.get('API_URL_PREDICT', 'http://mlops-api-pf:5000/predict')

FEATURE_NAMES = [
    'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
    'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension',
    'radius error','texture error','perimeter error','area error','smoothness error',
    'compactness error','concavity error','concave points error','symmetry error','fractal dimension error',
    'worst radius','worst texture','worst perimeter','worst area','worst smoothness',
    'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension'
]

st.markdown("""
Este panel muestra la **exactitud (accuracy)** y la **importancia de las variables** calculadas 
usando `permutation_importance` sobre el conjunto de test.
""")

try:
    if 'active_features' not in st.session_state:
        features_send = requests.get(API_FEATURES).json()['feature_names']
    else:
        features_send = st.session_state['active_features']
    payload = {'feature_names': st.session_state.get('active_features', FEATURE_NAMES)}
    response = requests.post(API_RETRAIN, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    if 'error' in data:
        st.error(f"Error desde la API: {data['error']}")
    else:
        # Mostrar accuracy
        accuracy = data.get('accuracy', None)
        if accuracy is not None:
            st.metric(label="Exactitud (Accuracy) en Test", value=f"{accuracy:.4f}")

        # Mostrar feature importances
        fi = data.get('feature_importances', [])
        fidx = data.get('idx_list', [])
        if fi:
            feature_names = st.session_state['active_features'] if 'active_features' in st.session_state else FEATURE_NAMES
            # Construir DataFrame para graficar
            df_fi = pd.DataFrame({'feature': feature_names, 'importance': fi})
            df_fi = df_fi.sort_values(by='importance', ascending=False)

            st.subheader("Importancia de Características (Permutation Importance)")

            # Gráfico de barras con seaborn y matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=df_fi, palette='viridis', ax=ax)
            ax.set_xlabel("Importancia promedio")
            ax.set_ylabel("Feature")
            ax.set_title("Features ordenadas por Importancia")
            st.pyplot(fig)
        else:
            st.warning("No se recibieron importancias de características.")

except requests.exceptions.RequestException as e:
    st.error(f"Error al conectar con la API: {e}")
except Exception as e:
    st.error(f"Ocurrió un error inesperado: {e}")




st.markdown("### 2) Selecciona features que consideras irrelevantes y reentrena")
st.write("Marca (multi-select) las features que quieres eliminar y pulsa 'Reentrenar sin estas features'. Se entrenará un nuevo modelo en el backend y se te mostrarán accuracy y explicabilidad del nuevo modelo. También podrás comparar con el actual.")

drop_display_features = [fname for fname in FEATURE_NAMES if fname not in st.session_state.get('active_features', FEATURE_NAMES)]
drop = st.multiselect('Selecciona features a eliminar (drop)', options=FEATURE_NAMES, default=drop_display_features)

col_a, col_b = st.columns([1,1])
with col_a:
    n_estimators = st.number_input('n_estimators (RandomForest) para reentrenar', min_value=10, max_value=1000, value=100, step=10)
with col_b:
    retrain_button = st.button('Reentrenar sin las features seleccionadas')

if 'retrain_result' in st.session_state and st.session_state['retrain_result'] is not None:
    d = st.session_state['retrain_result']

    st.success(f"Modelo reentrenado. Nuevo accuracy: **{d['accuracy']:.4f}**")
    st.write(f"Nº features usadas: {len(d['used_features'])}")

    fi = d.get('feature_importances', [])
    used_features = d.get('used_features', [])

    if fi:
        df_fi = pd.DataFrame({'feature': used_features, 'importance': fi}).sort_values('importance', ascending=False)
        st.subheader('Top importancias (nuevo modelo)')
        st.bar_chart(df_fi.head(20).set_index('feature'))

    if st.button("Ocultar resultados del reentrenamiento"):
        st.session_state['retrain_result'] = None
        st.rerun()

if retrain_button:
    try:
        stay = [f for f in FEATURE_NAMES if f not in drop]

        payload = {'feature_names': stay, 'n_estimators': int(n_estimators)}
        feature_names = stay
        resp = requests.post(API_RETRAIN, json=payload, timeout=60)
        if resp.status_code == 200:
            d = resp.json()
            used_features = d.get('used_features')
            st.session_state['active_features'] = used_features
            st.session_state['just_retrained'] = True
            st.session_state['retrain_result'] = d
            st.rerun()
    except Exception as e:
        st.error(f'Error al conectarse a /retrain: {e}')