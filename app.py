from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

import joblib
import numpy as np
import pandas as pd

# =========================
# LOAD MODEL
# =========================
model = joblib.load('models/fruit_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# =========================
# APP INIT + TITLE SWAGGER
# =========================
app = FastAPI(
    title="Fruit Clustering API",
    description="""
    API de Machine Learning pour regrouper des fruits en clusters.

    Modèle : KMeans / Clustering  
    Entrées : 2 features numériques  
    Sorties : cluster ID  

    Cette API fait partie d’un pipeline MLOps avec FastAPI + MLflow + Docker.
    """,
    version="1.0.0"
)

# =========================
# INPUT MODEL
# =========================
class FruitData(BaseModel):
    Feature1: float
    Feature2: float

    class Config:
        json_schema_extra = {
            "example": {
                "Feature1": 43.7,
                "Feature2": 38.57
            }
        }

# =========================
# HOME
# =========================
@app.get(
    "/",
    tags=["Health Check"],
    summary="API status",
    description="Vérifie que l'API fonctionne correctement."
)
def home():
    return {"message": "Fruit Clustering API is running 🍏"}

# =========================
# SINGLE PREDICTION
# =========================
@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict single fruit cluster",
    description="Prédit le cluster d’un seul point de données (Feature1, Feature2)."
)
def predict(data: FruitData):

    features = np.array([[data.Feature1, data.Feature2]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    return {
        "cluster": int(prediction[0]),
        "message": f"Ce fruit appartient au groupe {prediction[0]}"
    }

# =========================
# BATCH PREDICTION
# =========================
@app.post(
    "/predict_batch",
    tags=["Prediction"],
    summary="Predict multiple fruits",
    description="Prend une liste de fruits et retourne leurs clusters."
)
def predict_batch(data: List[FruitData]):

    features = np.array([[item.Feature1, item.Feature2] for item in data])
    scaled = scaler.transform(features)
    predictions = model.predict(scaled)

    return {
        "clusters": [int(p) for p in predictions]
    }

# =========================
# CSV PREDICTION
# =========================
@app.post(
    "/predict_csv",
    tags=["Prediction"],
    summary="Predict from CSV file",
    description="Upload un fichier CSV contenant Feature1 et Feature2."
)
def predict_csv(file: UploadFile = File(...)):

    df = pd.read_csv(file.file, header=None, names=["Feature1", "Feature2"])

    features = df.values
    scaled = scaler.transform(features)

    predictions = model.predict(scaled)

    return {
        "clusters": [int(p) for p in predictions]
    }