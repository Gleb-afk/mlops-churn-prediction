import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH, CHURN_THRESHOLD


def load_model():
    """
    Загружает обученную лучшую модель из папки models.
    """

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {BEST_MODEL_PATH}. "
            "Сначала запустите обучение: python -m src.train"
        )

    return joblib.load(BEST_MODEL_PATH)


def get_risk_level(churn_probability: float) -> str:
    """
    Преобразует вероятность оттока в уровень риска.
    """

    if churn_probability >= 0.60:
        return "high"
    if churn_probability >= CHURN_THRESHOLD:
        return "medium"
    return "low"


def predict_churn(customer_data: dict) -> dict:
    """
    Выполняет прогноз вероятности оттока для одного клиента.
    """

    model = load_model()

    input_df = pd.DataFrame([customer_data])

    churn_probability = float(model.predict_proba(input_df)[:, 1][0])
    churn_prediction = int(churn_probability >= CHURN_THRESHOLD)
    risk_level = get_risk_level(churn_probability)

    result = {
        "churn_probability": round(churn_probability, 4),
        "churn_probability_percent": round(churn_probability * 100, 2),
        "churn_prediction": churn_prediction,
        "risk_level": risk_level,
        "threshold": CHURN_THRESHOLD,
    }

    return result


def main() -> None:
    """
    Запускает пример предсказания для тестового клиента.
    """

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

    prediction = predict_churn(sample_customer)

    print("Результат предсказания:")
    print(prediction)

    if prediction["churn_prediction"] == 1:
        print("Вывод: у клиента есть риск оттока.")
    else:
        print("Вывод: у клиента низкий риск оттока.")


if __name__ == "__main__":
    main()