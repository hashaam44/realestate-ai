from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.config import (
    MODEL_PATH,
    FRONTEND_ORIGINS,
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION
)

from api.schemas import PropertyInput, PredictionResponse


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health_check():
    return {
        "api_status": "running",
        "model_status": "loaded"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_price(property_data: PropertyInput):
    try:
        property_dict = property_data.model_dump()

        input_df = pd.DataFrame([property_dict])

        prediction = model.predict(input_df)[0]

        return PredictionResponse(
            predicted_price=round(float(prediction), 2),
            currency="USD"
        )

    except Exception as error:
        print(f"Prediction error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs."
        ) from error