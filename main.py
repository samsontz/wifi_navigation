from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    NM_AIST_D: int
    NM_AIST_B: int
    NM_AIST_C: int
    CDAC_14: int
       

# loading the saved model
wifi_model = pickle.load(open('random_forest_model.pkl','rb'))


@app.post('/predict')
def wifi_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    a = input_dictionary['NM_AIST_D']
    b = input_dictionary['NM_AIST_B']
    c = input_dictionary['NM_AIST_C']
    dac = input_dictionary['CDAC_14']


    input_list = [a, b, c, dac]
    
    prediction = wifi_model.predict([input_list])
    

