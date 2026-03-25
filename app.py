import joblib
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from model import MLPRegressor


app = FastAPI(
    title="Glucose Predictor API",
    description="Predicts plasma glucose concentration in mg/dL",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

feat_scaler = joblib.load("feat_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

model = MLPRegressor(in_features=7, hidden_sizes=[64, 32, 16])
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()


class InputData(BaseModel):
    Pregnancies: float = Field(..., example=2.0)
    BloodPressure: float = Field(..., example=70.0)
    SkinThickness: float = Field(..., example=30.0)
    Insulin: float = Field(..., example=120.0)
    BMI: float = Field(..., example=33.5)
    DiabetesPedigree: float = Field(..., example=0.5)
    Age: float = Field(..., example=28.0)


class PredictionResponse(BaseModel):
    predicted_glucose_mg_dL: float
    message: str


@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    features = np.array([[
        data.Pregnancies,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigree,
        data.Age
    ]], dtype=np.float32)

    scaled = feat_scaler.transform(features).astype(np.float32)
    tensor = torch.tensor(scaled)

    with torch.no_grad():
        pred_scaled = model(tensor).numpy()

    pred_orig = target_scaler.inverse_transform(pred_scaled)
    glucose = round(float(pred_orig[0][0]), 2)

    if glucose < 100:
        message = "Normal fasting glucose range"
    elif glucose < 126:
        message = "Pre-diabetic range (consult a doctor)"
    else:
        message = "Diabetic range (consult a doctor)"

    return PredictionResponse(
        predicted_glucose_mg_dL=glucose,
        message=message
    )
