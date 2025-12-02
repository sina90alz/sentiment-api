from fastapi import FastAPI
from pydantic import BaseModel

from inference import SentimentInference

app = FastAPI()
inference = SentimentInference()

class Input(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: Input):
    return inference.predict(data.text)
