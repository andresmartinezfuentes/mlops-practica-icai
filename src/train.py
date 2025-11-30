# train.py
import os
import json
import joblib
import argparse
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") 
mlflow.set_tracking_uri(tracking_uri) 

MODEL_PATH = '../models/model.pkl'
CSV_PATH = "../data/breast_cancer.csv"

def train_model(n_estimators=100):
    data = pd.read_csv(CSV_PATH)
    feature_names = [c for c in data.columns if c != "target"]
    X = data[feature_names].copy()
    X = X.to_numpy()

    y = data["target"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Guardar modelo
        joblib.dump(model, MODEL_PATH)
        mlflow.sklearn.log_model(model, 'random-forest-model')

        # Log params & metrics
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_metric('accuracy', acc)

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.savefig('confusion_matrix.png', bbox_inches='tight')
        mlflow.log_artifact('confusion_matrix.png')

        metrics = {
            'accuracy': acc
        }
        with open("mlflow_metrics.json", "w") as f: 
           json.dump(metrics, f) 
 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    args = parser.parse_args()
    train_model(args.n_estimators)
