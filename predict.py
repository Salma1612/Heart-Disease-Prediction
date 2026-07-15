"""
Heart Disease Prediction - Inference Script
=============================================

Loads the trained Random Forest model and fitted StandardScaler, then
runs a prediction on a sample patient record.

Usage:
    python predict.py
"""

import os

import joblib
import numpy as np

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Feature order must exactly match the training data columns
# (excluding the "target" column).
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def load_artifacts():
    """Load the trained model and scaler from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Model or scaler not found. Please run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_patient(model, scaler, patient_data):
    """
    Predict heart disease presence for a single patient record.

    Args:
        model: Trained classifier with predict/predict_proba methods.
        scaler: Fitted StandardScaler used during training.
        patient_data (dict): Mapping of feature name to value.

    Returns:
        tuple: (prediction (int), probability of disease (float))
    """
    # Ensure feature order matches training data
    features = np.array([[patient_data[feature] for feature in FEATURE_NAMES]])

    # Apply the same scaling used during training
    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]

    return prediction, probability


def main():
    """Run a sample prediction using a hardcoded patient record."""
    model, scaler = load_artifacts()

    # Sample patient record (feature values for demonstration purposes)
    sample_patient = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }

    prediction, probability = predict_patient(model, scaler, sample_patient)

    print("===== Heart Disease Prediction =====")
    print("Patient Data:", sample_patient)
    print("Prediction:", "Heart Disease Detected" if prediction == 1 else "No Heart Disease")
    print(f"Probability of Heart Disease: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
