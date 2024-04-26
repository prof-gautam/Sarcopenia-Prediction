# Sarcopenia Prediction API

This API provides an interface for predicting sarcopenia using various machine learning models. Sarcopenia, an age-related muscle degeneration disease, can significantly impact the elderly with severe health outcomes like increased falls, frailty, and even premature mortality. This API utilizes models trained to understand and predict the risk of sarcopenia based on clinical inputs.

## Models

The API supports multiple models for both male and female patients:

- Gradient Boosting
- Logistic Regression
- Random Forest

Each model has been trained separately for male and female data to account for biological differences affecting sarcopenia.

## Usage

### Endpoint: `/predict/{model_id}`

Make a POST request to this endpoint with the required input parameters to get the prediction probability of sarcopenia.

### Input

The input should be a JSON object with the following fields:

- `Age`: Patient's age in years (float)
- `Weight`: Patient's weight in kilograms (float)
- `Height`: Patient's height in centimeters (float)
- `GripStrength`: Measured grip strength of the patient (float)
- `Gender`: Patient's gender ('male' or 'female')

### Output

The output will be a JSON object with the following fields:

- `model`: Identifier of the model used for the prediction
- `prediction_probability`: Probability of the patient having sarcopenia (float)

## Example

```json
POST /predict/model1_gradient_boosting_female
Content-Type: application/json

{
    "Age": 65,
    "Weight": 70,
    "Height": 165,
    "GripStrength": 30,
    "Gender": "female"
}

Response:
{
    "model": "model1_gradient_boosting_female",
    "prediction_probability": 0.85
}
```


## Running the Server

To run the server, use the following command:

uvicorn main:app --reload

As the data can be cofidential, I have ignored dataset
