import joblib
import numpy as np

def predict_stroke(age, avg_glucose_level, hypertension, heart_disease, bmi):
    model = joblib.load('stroke_risk_model.pkl')
    # Example: expand inputs to match model features (dummy values used here)
    input_data = np.array([[0, age, hypertension, heart_disease, avg_glucose_level, 1, 0, 0, bmi, 1]])
    prediction = model.predict_proba(input_data)[0][1]
    return round(prediction * 100, 2)