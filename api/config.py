from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "real_estate_price_model.pkl"

FRONTEND_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

API_TITLE = "RealEstate AI API"
API_DESCRIPTION = "API for predicting real estate prices"
API_VERSION = "1.0.0"