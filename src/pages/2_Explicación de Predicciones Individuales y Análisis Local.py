import streamlit as st
import requests
import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title='XAI Interactive — Breast Cancer', layout='wide')
st.title('XAI Interactivo: Entrena, explica y experimenta con features')

# Endpoint config
API_XAI = os.environ.get('API_URL_XAI', 'http://mlops-api:5000/xai_example')
API_RANGES = os.environ.get('API_URL_FEATURE_RANGES', 'http://mlops-api:5000/features_ranges')
API_FEATURES = os.environ.get('API_URL_FEATURES', 'http://mlops-api:5000/feature_names')
API_RETRAIN = os.environ.get('API_URL_RETRAIN', 'http://mlops-api:5000/retrain')
API_PREDICT = os.environ.get('API_URL_PREDICT', 'http://mlops-api:5000/predict')


FEATURE_NAMES = [
    'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
    'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension',
    'radius error','texture error','perimeter error','area error','smoothness error',
    'compactness error','concavity error','concave points error','symmetry error','fractal dimension error',
    'worst radius','worst texture','worst perimeter','worst area','worst smoothness',
    'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension'
]

st.markdown("### 1) Ajusta las features (sliders) — todas las features del dataset")
st.write("Puedes modificar valores para simular una observación. Para simplificar la UI los sliders usan rangos amplios, aunque puedes ajustar a rangos reales si quieres.")
if 'active_features' not in st.session_state:
    st.session_state['active_features'] = requests.get(API_FEATURES).json()['feature_names']

feature_names = st.session_state['active_features']
feature_values = {}
cols = st.columns(3)
j = 0
payload = {'feature_names': feature_names}
logging.info(f"Requesting feature ranges for: {feature_names}")
resp = requests.post(API_RANGES, json=payload, timeout=10).json()
for fname in FEATURE_NAMES:
    if fname in feature_names:
        dict_ranges = resp.get(fname, None)
        col = cols[j % 3]
        # default range and default value can be adjusted; here we use 0..100 as a general range
        feature_values[fname] = col.slider(fname, min_value=dict_ranges['min'], max_value=dict_ranges['max'], value=dict_ranges['min'], step=0.1)
        j += 1
    else:
        feature_values[fname] = 0.0

feature_vector = [feature_values[f] for f in st.session_state['active_features']]

st.markdown("### 2) Explicación del modelo actual (predicción + SHAP)")

st.markdown("### Obtener predicción (modelo actual)")
if st.button('Obtener predicción'):
    payload = {'features': feature_vector}
    try:
        resp = requests.post(API_PREDICT, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            pred = data.get('prediction')
            st.success(f"Predicción del modelo actual: **{('Benigno' if pred==0 else 'Maligno')}**")
        else:
            st.error(f'Error API /predict: {resp.status_code} - {resp.text}')
    except Exception as e:
        st.error(f'Error al conectar con API /predict: {e}')

if st.button('Obtener predicción y explicación (modelo actual)'):
    payload = {'features': feature_vector}
    try:
        resp = requests.post(API_XAI, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            pred = data.get('prediction')
            expected_value = data.get('expected_value')
            shap_vals = data.get('shap_values', [])
            st.success(f"Predicción del modelo actual: **{('Benigno' if pred==0 else 'Maligno')}**")
            st.write(f"Expected value (modelo): {expected_value:.4f}")

            if shap_vals:
                df_shap = pd.DataFrame({'feature': feature_names, 'shap': shap_vals})
                df_display = df_shap.head(30).set_index('feature')
                st.subheader('SHAP values (modelo actual) — primeras 30 features')
                st.bar_chart(df_display['shap'])

                top_pos = df_shap.sort_values('shap', ascending=False).head(3)
                top_neg = df_shap.sort_values('shap', ascending=True).head(3)

                st.write("Top 3 que empujan hacia la clase predicha:")
                for idx, row in top_pos.iterrows():
                    st.write(f"- **{row['feature']}**: SHAP={row['shap']:.4f}")
                st.write("Top 3 que empujan en contra:")
                for idx, row in top_neg.iterrows():
                    st.write(f"- **{row['feature']}**: SHAP={row['shap']:.4f}")
            else:
                st.warning('La API no devolvió SHAP values.')
        else:
            st.error(f'Error API /xai: {resp.status_code} - {resp.text}')
    except Exception as e:
        st.error(f'Error al conectar con API /xai: {e}')
