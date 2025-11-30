import joblib
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import pandas as pd
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST 

PREDICTION_COUNTER = Counter( 
   'cancer_prediction_count', 
   'Contador de predicciones del modelo breast cancer por tipo', 
   ['type'] 
) 

CSV_PATH = "../data/breast_cancer.csv"
MODEL_PATH = '../models/model.pkl'

feature_names = [
    'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
    'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension',
    'radius error','texture error','perimeter error','area error','smoothness error',
    'compactness error','concavity error','concave points error','symmetry error','fractal dimension error',
    'worst radius','worst texture','worst perimeter','worst area','worst smoothness',
    'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension'
]

app = Flask(__name__)
CORS(app)

try: 
   model = joblib.load(MODEL_PATH) 
except FileNotFoundError: 
   print("Error: 'model.pkl' no encontrado. Por favor, asegúrate de haber ejecutado el script de entrenamiento.") 
   model = None 
 
# Inicializar la aplicación Flask 
app = Flask(__name__)  
@app.route('/metrics') 
def metrics(): 
   return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST) 

@app.route("/feature_names", methods=["GET"])
def feature():
    return jsonify({"feature_names": feature_names})

@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load(MODEL_PATH)
    if model is None:
        return jsonify({'error': 'Modelo no cargado.'}), 500
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        pred = model.predict(features)
        prediction_int = int(pred[0])

        type_map = {0: 'Benigno', 1: 'Maligno'} 
        predicted_type = type_map.get(prediction_int, 'unknown')   
        PREDICTION_COUNTER.labels(type=predicted_type).inc()

        return jsonify({'prediction': prediction_int})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route("/features_ranges", methods=["POST"])
def features_ranges():
    try:
        body = request.get_json(force=True)
        feature_names = body.get('feature_names') or feature_names
        data = pd.read_csv(CSV_PATH)
        all_feature_names = [c for c in data.columns if c != "target"]
        X = data[all_feature_names].copy()
        ranges = {}
        for col in feature_names:
            ranges[col] = {
                'min': float(X[col].min()),
                'max': float(X[col].max()),
            }
        return jsonify(ranges)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route("/evaluate_model", methods=["POST"])
def evaluate_model():
    model = joblib.load(MODEL_PATH)
    if model is None:
        return jsonify({'error': 'Modelo no cargado.'}), 500
    try:
        body = request.get_json(force=True)
        feature_names = body.get('feature_names') or feature_names
        data = pd.read_csv(CSV_PATH)
        all_feature_names = [c for c in data.columns if c != "target"]
        X = data[all_feature_names].copy()
        y = data["target"].to_numpy()

        X_sub = X[feature_names]
        X_sub = X_sub.to_numpy()

        _, X_test, _, y_test = train_test_split(X_sub, y, test_size=0.25, random_state=42)

        y_pred = model.predict()
        acc = accuracy_score(y_test, y_pred)

        perm_imp = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0)
        return jsonify({'accuracy': float(acc),
                        'feature_importances': perm_imp.importances_mean.tolist(),
                        'idx_list': perm_imp.importances_mean.argsort().tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/xai_example', methods=['POST'])
def xai_example():
    model = joblib.load(MODEL_PATH)
    if model is None:
        return jsonify({'error': 'Modelo no cargado.'}), 500
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)


        explainer = shap.TreeExplainer(model)
        prediction = model.predict(features)
        pred_class = int(prediction[0])

        type_map = {0: 'Benigno', 1: 'Maligno'} 
        predicted_type = type_map.get(pred_class, 'unknown')   
        PREDICTION_COUNTER.labels(type=predicted_type).inc()

        shap_values = explainer.shap_values(features)

        if isinstance(shap_values, list) and len(shap_values) == 1:
        # binario/one-output
            shap_for_pred_class = shap_values[0]
            expected_value = explainer.expected_value[0]
        else:
            shap_for_pred_class = shap_values[0][:, pred_class]
            expected_value = explainer.expected_value[pred_class]

        return jsonify({
        'prediction': pred_class,
        'shap_values': shap_for_pred_class.tolist(),
        'expected_value': float(expected_value)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Request JSON:
    {
      "drop_features": ["mean texture", "worst symmetry", ...],  # list of feature names to drop
      "n_estimators": 100  # optional
    }

    Response:
    {
      "accuracy": 0.97,
      "used_features": [...],
      "feature_importances": [...],   # same length as used_features
      "shap_example_values": [...],   # shap values for a sample (first test sample)
      "expected_value": float,
      "message": "Modelo reentrenado y guardado en model.pkl"
    }
    """
    try:
        body = request.get_json(force=True)
        feature_names = body.get('feature_names') or feature_names
        n_estimators = int(body.get('n_estimators', 100))

        data = pd.read_csv(CSV_PATH)
        all_feature_names = [c for c in data.columns if c != "target"]
        X = data[all_feature_names].copy()
        y = data["target"].to_numpy()

        if len(feature_names) == 0:
            return jsonify({'error': 'No se pueden usar 0 features.'}), 400

        X_sub = X[feature_names]
        X_sub = X_sub.to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X_sub, y, test_size=0.25, random_state=42)
        model_new = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model_new.fit(X_train, y_train)

        y_pred = model_new.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        joblib.dump(model_new, MODEL_PATH)
    

        # Feature importances
        fi = model_new.feature_importances_.tolist()

        return jsonify({
            'accuracy': float(acc),
            'used_features': feature_names,
            'feature_importances': fi,
            'message': 'Modelo reentrenado y guardado en model.pkl'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
 
if __name__ == '__main__': 
   print("Iniciando API en puerto 5000...") 
   app.run(host='0.0.0.0', port=5000) 