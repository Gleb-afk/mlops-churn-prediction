from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


sample_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 255.0,
}


def test_root_endpoint():
    """
    Проверяет главную страницу API.
    """

    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    """
    Проверяет endpoint /health.
    """

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    """
    Проверяет endpoint /model-info.
    """

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "Gradient Boosting"
    assert data["threshold"] == 0.30


def test_predict_endpoint():
    """
    Проверяет endpoint /predict.
    """

    response = client.post("/predict", json=sample_customer)

    assert response.status_code == 200

    data = response.json()

    assert "input_data" in data
    assert "prediction" in data

    prediction = data["prediction"]

    assert "churn_probability" in prediction
    assert "churn_probability_percent" in prediction
    assert "churn_prediction" in prediction
    assert "risk_level" in prediction
    assert "threshold" in prediction

    assert 0 <= prediction["churn_probability"] <= 1
    assert 0 <= prediction["churn_probability_percent"] <= 100
    assert prediction["churn_prediction"] in [0, 1]
    assert prediction["risk_level"] in ["low", "medium", "high"]