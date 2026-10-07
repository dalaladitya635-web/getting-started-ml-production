from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="Getting Started with ML in Production API",
    description="A minimal text classification prediction API built for the workshop.",
    version="1.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: int


@app.get("/")
def root():
    return {"message": "Workshop ML API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )

    try:
        pred = int(model.predict([request.text])[0])
        return PredictionResponse(prediction=pred)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )