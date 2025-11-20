from fastapi import FastAPI
from pydantic import BaseModel

from inference import predict

app = FastAPI()

class Input(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: Input):
    return predict(data.text)
