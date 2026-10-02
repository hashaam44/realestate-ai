from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "RealEstate AI API is running!"


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["api_status"] == "running"
    assert response.json()["model_status"] == "loaded"


def test_prediction():
    property_data = {
        "bed": 3,
        "bath": 2,
        "acre_lot": 0.25,
        "house_size": 1800,
        "prev_sold_year": 2020,
        "status": "for_sale",
        "state": "California"
    }

    response = client.post(
        "/predict",
        json=property_data
    )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_price" in result
    assert isinstance(result["predicted_price"], float)
    assert result["currency"] == "USD"


def test_invalid_prediction():
    property_data = {
        "bed": -2,
        "bath": 2,
        "acre_lot": 0.25,
        "house_size": 1800,
        "prev_sold_year": 2020,
        "status": "for_sale",
        "state": "California"
    }

    response = client.post(
        "/predict",
        json=property_data
    )

    assert response.status_code == 422