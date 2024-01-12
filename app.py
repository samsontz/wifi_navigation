from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("random_forest_model.pkl","rb")
random_forest_model = pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    NMAIST_D=data['NMAIST_D']
    NMAIST_B=data['NMAIST_B']
    NMAIST_C=data['NMAIST_C']
    CDAC_14=data['CDAC_14']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = random_forest_model.predict([[NMAIST_D,NMAIST_B,NMAIST_C,CDAC_14]])
        output_x, output_y = prediction[0]
        return {"output_x": output_x, "output_y": output_y}
