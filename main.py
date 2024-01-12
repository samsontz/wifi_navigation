from fastapi import FastAPI

# 2. Create the app object
app = FastAPI()


@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}




