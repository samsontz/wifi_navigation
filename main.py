from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

class ModelInput(BaseModel):
    NMAIST_D: int
    NMAIST_B: int
    NMAIST_C: int
    CDAC_14: int

# loading the saved model
wifi_model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.post('/predict')
def wifi_pred(input_parameters: ModelInput):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    a = input_dictionary['NMAIST_D']
    b = input_dictionary['NMAIST_B']
    c = input_dictionary['NMAIST_C']
    dac = input_dictionary['CDAC_14']

    input_list = [a, b, c, dac]
    prediction = wifi_model.predict([input_list])

    # Extracting X and Y predictions
    output_x, output_y = prediction[0]

    return {"output_x": output_x, "output_y": output_y}

    
