import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_DIR = "../data"
CSV_PATH = os.path.join(DATA_DIR, "breast_cancer.csv")

def save_sklearn_dataset_to_csv():

    if os.path.exists(CSV_PATH):
        print("CSV ya existe, no se vuelve a generar.")
        return CSV_PATH

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    df.to_csv(CSV_PATH, index=False)
    print(f"Dataset guardado como CSV en {CSV_PATH}")


save_sklearn_dataset_to_csv()