from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

# Combined startup event
@app.on_event("startup")
async def startup_event():
    global wifi_model
    wifi_model = get_model()
    print(f"Loaded model type: {type(wifi_model)}")  # Debug print

class ModelInput(BaseModel):
    NMAIST_D: int
    NMAIST_B: int
    NMAIST_C: int
    CDAC_14: int

class ModelOutput(BaseModel):
    output_x: float
    output_y: float


import joblib

def load_model():
    model_path = os.getenv("MODEL_PATH", "random_forest_model.pkl")
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")


def get_model():
    try:
        return load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/', response_model=ModelOutput)
def wifi_pred(input_parameters: ModelInput, model=Depends(get_model)):
    try:
        prediction = model.predict([[input_parameters.NMAIST_D, 
                                     input_parameters.NMAIST_B, 
                                     input_parameters.NMAIST_C, 
                                     input_parameters.CDAC_14]])
        output_x, output_y = prediction[0]
        return {"output_x": output_x, "output_y": output_y}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
