from fastapi import FastAPI
from pydantic import BaseModel

from src.config import CHURN_THRESHOLD
from src.predict import predict_churn


app = FastAPI(
    title="Churn Prediction API",
    description="API для прогнозирования оттока клиентов на основе ML-модели.",
    version="1.0.0",
)


class CustomerData(BaseModel):
    """
    Схема входных данных клиента для прогноза оттока.
    """

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    """
    Главная страница API.
    """

    return {
        "message": "Churn Prediction API работает",
        "description": "Сервис прогнозирует вероятность оттока клиента.",
    }


@app.get("/health")
def health_check():
    """
    Проверка работоспособности API.
    """

    return {
        "status": "ok",
        "message": "API доступен",
    }


@app.get("/model-info")
def model_info():
    """
    Возвращает краткую информацию о модели и используемом пороге классификации.
    """

    return {
        "model": "Gradient Boosting",
        "task": "customer churn prediction",
        "threshold": CHURN_THRESHOLD,
        "description": "Модель прогнозирует вероятность оттока клиента.",
    }


@app.post("/predict")
def predict(customer: CustomerData):
    """
    Выполняет прогноз оттока клиента.
    """

    customer_dict = customer.model_dump()
    prediction = predict_churn(customer_dict)

    return {
        "input_data": customer_dict,
        "prediction": prediction,
    }