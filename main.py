from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"


# ==========================================
# Load model files once when API starts
# ==========================================

model = tf.keras.models.load_model(
    MODEL_DIR / "autoimmune_model.keras",
    compile=False
)

scaler = joblib.load(
    MODEL_DIR / "scaler.joblib"
)

pca = joblib.load(
    MODEL_DIR / "pca.joblib"
)

with open(
    MODEL_DIR / "feature_names.json",
    "r",
    encoding="utf-8"
) as file:
    feature_names = json.load(file)

with open(
    MODEL_DIR / "metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)


# ==========================================
# Disease mapping
# ==========================================

CLASS_NAMES = {
    0: "Ankylosing Spondylitis",
    1: "Normal",
    2: "Psoriatic Arthritis",
    3: "Reactive Arthritis",
    4: "Sjogren's Syndrome",
    5: "Systemic Lupus Erythematosus"
}


# ==========================================
# Request model
# ==========================================

class PatientInput(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True
    )

    Gender: int = Field(ge=0, le=1)

    ESR: float
    CRP: float
    RF: float

    Anti_CCP: float = Field(
        alias="Anti-CCP"
    )

    HLA_B27: int = Field(
        ge=0,
        le=1,
        alias="HLA-B27"
    )

    ANA: int = Field(ge=0, le=1)

    Anti_Ro: int = Field(
        ge=0,
        le=1,
        alias="Anti-Ro"
    )

    Anti_La: int = Field(
        ge=0,
        le=1,
        alias="Anti-La"
    )

    Anti_dsDNA: int = Field(
        ge=0,
        le=1,
        alias="Anti-dsDNA"
    )

    Anti_Sm: int = Field(
        ge=0,
        le=1,
        alias="Anti-Sm"
    )

    C3: float
    C4: float


# ==========================================
# FastAPI
# ==========================================

app = FastAPI(
    title="Autoimmune Disease Prediction API",
    description="Prediction API using StandardScaler, PCA and ANN",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Autoimmune Disease Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "pca_components": int(pca.n_components_),
        "expected_features": feature_names
    }


@app.post("/predict")
def predict(patient: PatientInput):
    try:
        # تحويل طلب Pydantic إلى dictionary
        input_data = patient.model_dump(
            by_alias=True
        )

        # التأكد من ترتيب الأعمدة كما في التدريب
        ordered_values = [
            input_data[feature]
            for feature in feature_names
        ]

        input_dataframe = pd.DataFrame(
            [ordered_values],
            columns=feature_names
        )

        # نفس خطوات preprocessing المستخدمة بالتدريب
        scaled_input = scaler.transform(
            input_dataframe
        )

        pca_input = pca.transform(
            scaled_input
        )

        # Prediction
        probabilities = model.predict(
            pca_input,
            verbose=0
        )[0]

        predicted_class = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_class]
        )

        probability_results = {
            CLASS_NAMES[index]: round(
                float(probability),
                6
            )
            for index, probability
            in enumerate(probabilities)
        }

        return {
            "predicted_class": predicted_class,
            "predicted_disease": CLASS_NAMES[
                predicted_class
            ],
            "confidence": round(confidence, 6),
            "probabilities": probability_results
        }

    except KeyError as error:
        raise HTTPException(
            status_code=422,
            detail=f"Missing feature: {str(error)}"
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )