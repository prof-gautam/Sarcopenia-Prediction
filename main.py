from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import numpy as np

app = FastAPI(title="Sarcopenia Prediction API")

models = {
    'model1_gradient_boosting_female': load(f'model1_gradient_boosting_female.joblib'),
    'model1_gradient_boosting_male': load(f'model1_gradient_boosting_male.joblib'),
    'model1_logistic_regression_female': load(f'model1_logistic_regression_female.joblib'),
    'model1_logistic_regression_male': load(f'model1_logistic_regression_male.joblib'),
    'model1_random_forest_female': load(f'model1_random_forest_female.joblib'),
    'model1_random_forest_male': load(f'model1_random_forest_male.joblib'),
    
    'model2_gb_female': load(f'model2_gb_female.joblib'),
    'model2_gb_male': load(f'model2_gb_male.joblib'),
    'model2_logreg_female': load(f'model2_logreg_female.joblib'),
    'model2_logreg_male': load(f'model2_logreg_male.joblib'),
    'model2_rf_female': load(f'model2_rf_female.joblib'),
    'model2_rf_male': load(f'model2_rf_male.joblib'),
    
    'model3_gb_female': load(f'model3_gb_female.joblib'),
    'model3_gb_male': load(f'model3_gb_male.joblib'),
    'model3_logreg_female': load(f'model3_logreg_female.joblib'),
    'model3_logreg_male': load(f'model3_logreg_male.joblib'),
    'model3_rf_female': load(f'model3_rf_female.joblib'),
    'model3_rf_male': load(f'model3_rf_male.joblib'),
}

# Pydantic models for expected request and response data.
class PredictionInput(BaseModel):
    Age: float
    Weight: float
    Height: float
    GripStrength: float
    Gender: str  # Expect 'male' or 'female'

class PredictionOutput(BaseModel):
    model: str
    prediction_probability: float

# Prediction endpoint which takes inputs from the user and returns predictions.
@app.post("/predict/{model_id}", response_model=PredictionOutput)
async def predict(model_id: str, input_data: PredictionInput):
    # Check if the model exists
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found.")

    # Preprocess the input data as required before making a prediction.
    # This is just a placeholder; you need to replace it with your own preprocessing steps.
    features = np.array([[input_data.Age, input_data.Weight, input_data.Height, input_data.GripStrength]])

    # Make a prediction
    model = models[model_id]
    prediction_probability = model.predict_proba(features)[0, 1]

    # Return the prediction probability
    return {
        "model": model_id,
        "prediction_probability": prediction_probability
    }

# Run this command in your terminal to start the server:
# uvicorn script_name:app --reload
# Replace script_name with the actual name of this Python script.
