# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

class ModelInput(BaseModel):
    NMAIST_D: int
    NMAIST_B: int
    NMAIST_C: int
    CDAC_14: int

class ModelOutput(BaseModel):
    output_x: float
    output_y: float

# Load model
def load_model():
    model_path = os.getenv("MODEL_PATH", "random_forest_model.pkl")
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")

wifi_model = load_model()

@app.post('/predict', response_model=ModelOutput)
def wifi_pred(input_parameters: ModelInput):
    try:
        prediction = wifi_model.predict([[input_parameters.NMAIST_D, 
                                          input_parameters.NMAIST_B, 
                                          input_parameters.NMAIST_C, 
                                          input_parameters.CDAC_14]])
        output_x, output_y = prediction[0]
        return {"output_x": output_x, "output_y": output_y}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# The Uvicorn command should be specified in the Procfile for Heroku
