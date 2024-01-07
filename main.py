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
    
    NM-AIST_D: int
    NM-AIST_B: int
    NM-AIST_C: int
    CDAC_14: int
       

# loading the saved model
diabetes_model = pickle.load(open('random_forest_model.pkl','rb'))


@app.post('/predict')
def diabetes_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    d = input_dictionary['NM-AIST_D']
    b = input_dictionary['NM-AIST_B']
    c = input_dictionary['NM-AIST_C']
    dac = input_dictionary['CDAC_14']


    input_list = [a, b, c, dac]
    
    prediction = diabetes_model.predict([input_list])
    

