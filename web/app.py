import os
import sys
from pathlib import Path

# Добавляем корневую папку проекта в путь импортов,
# чтобы Streamlit видел папки src, api, reports и другие модули проекта
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import requests
import streamlit as st

from src.config import (
    MODEL_COMPARISON_PATH,
    METRICS_PATH,
    REPORTS_DIR,
    THRESHOLD_METRICS_PATH,
)


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Прогнозирование оттока клиентов",
    page_icon="📊",
    layout="wide",
)


def get_risk_text(risk_level: str) -> str:
    """
    Возвращает русское описание уровня риска.
    """

    risk_map = {
        "low": "низкий риск",
        "medium": "средний риск",
        "high": "высокий риск",
    }

    return risk_map.get(risk_level, risk_level)


def get_prediction(customer_data: dict) -> dict:
    """
    Отправляет данные клиента в FastAPI и получает прогноз модели.
    """

    response = requests.post(
        f"{API_URL}/predict",
        json=customer_data,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def show_prediction_page() -> None:
    """
    Отображает страницу с формой для прогноза оттока клиента.
    """

    st.header("Прогноз оттока клиента")

    st.write(
        "Заполните данные клиента. Модель рассчитает вероятность оттока "
        "и определит уровень риска."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Пол", ["Female", "Male"])
            senior_citizen = st.selectbox("Пожилой клиент", [0, 1])
            partner = st.selectbox("Есть партнёр", ["Yes", "No"])
            dependents = st.selectbox("Есть иждивенцы", ["Yes", "No"])
            tenure = st.number_input(
                "Срок обслуживания, месяцев",
                min_value=0,
                max_value=100,
                value=3,
            )
            phone_service = st.selectbox("Телефонная связь", ["Yes", "No"])

        with col2:
            multiple_lines = st.selectbox(
                "Несколько телефонных линий",
                ["No", "Yes", "No phone service"],
            )
            internet_service = st.selectbox(
                "Тип интернета",
                ["DSL", "Fiber optic", "No"],
            )
            online_security = st.selectbox(
                "Онлайн-безопасность",
                ["No", "Yes", "No internet service"],
            )
            online_backup = st.selectbox(
                "Онлайн-резервное копирование",
                ["No", "Yes", "No internet service"],
            )
            device_protection = st.selectbox(
                "Защита устройства",
                ["No", "Yes", "No internet service"],
            )
            tech_support = st.selectbox(
                "Техническая поддержка",
                ["No", "Yes", "No internet service"],
            )

        with col3:
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
            )
            contract = st.selectbox(
                "Тип договора",
                ["Month-to-month", "One year", "Two year"],
            )
            paperless_billing = st.selectbox("Электронный счёт", ["Yes", "No"])
            payment_method = st.selectbox(
                "Способ оплаты",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Ежемесячный платёж",
                min_value=0.0,
                value=85.0,
            )
            total_charges = st.number_input(
                "Общая сумма платежей",
                min_value=0.0,
                value=255.0,
            )

        submitted = st.form_submit_button("Рассчитать вероятность оттока")

    if submitted:
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            result = get_prediction(customer_data)
            prediction = result["prediction"]

            probability_percent = prediction["churn_probability_percent"]
            churn_prediction = prediction["churn_prediction"]
            risk_level = prediction["risk_level"]
            threshold = prediction["threshold"]

            st.subheader("Результат прогноза")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            metric_col1.metric(
                "Вероятность оттока",
                f"{probability_percent}%",
            )

            metric_col2.metric(
                "Порог классификации",
                threshold,
            )

            metric_col3.metric(
                "Уровень риска",
                get_risk_text(risk_level),
            )

            if churn_prediction == 1:
                st.error("Клиент относится к группе риска оттока.")
            else:
                st.success("Клиент не относится к группе риска оттока.")

            st.write("Данные, отправленные в модель:")
            st.json(customer_data)

        except requests.exceptions.ConnectionError:
            st.error(
                "Не удалось подключиться к API. "
                "Проверьте, что FastAPI запущен командой: "
                "uvicorn api.main:app --reload"
            )
        except requests.exceptions.RequestException as error:
            st.error(f"Ошибка при запросе к API: {error}")


def show_model_comparison_page() -> None:
    """
    Отображает страницу сравнения моделей.
    """

    st.header("Сравнение моделей")

    if not MODEL_COMPARISON_PATH.exists():
        st.warning("Файл сравнения моделей не найден. Сначала запустите обучение.")
        return

    comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)

    st.write(
        "В таблице показаны метрики качества моделей, обученных в рамках "
        "MLOps-пайплайна."
    )

    st.dataframe(comparison_df, width="stretch")

    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    st.subheader("Графическое сравнение метрик")

    chart_df = comparison_df.set_index("model")[metric_columns]
    st.bar_chart(chart_df)

    st.subheader("Краткий вывод")

    best_model = comparison_df.iloc[0]["model"]
    best_roc_auc = comparison_df.iloc[0]["roc_auc"]

    st.info(
        f"Лучшая модель по ROC-AUC: {best_model}. "
        f"Значение ROC-AUC: {best_roc_auc:.4f}."
    )


def show_analysis_page() -> None:
    """
    Отображает страницу с графиками и результатами анализа модели.
    """

    st.header("Анализ лучшей модели")

    metrics_path = Path(METRICS_PATH)

    if metrics_path.exists():
        metrics_data = pd.read_json(metrics_path)
        st.subheader("Файл metrics.json")
        st.json(metrics_data.to_dict())

    st.subheader("Матрица ошибок")

    confusion_matrix_path = REPORTS_DIR / "confusion_matrix.png"

    if confusion_matrix_path.exists():
        st.image(str(confusion_matrix_path), caption="Confusion Matrix")
    else:
        st.warning("Файл confusion_matrix.png не найден.")

    st.subheader("ROC-кривая")

    roc_curve_path = REPORTS_DIR / "roc_curve.png"

    if roc_curve_path.exists():
        st.image(str(roc_curve_path), caption="ROC Curve")
    else:
        st.warning("Файл roc_curve.png не найден.")

    st.subheader("Важность признаков")

    feature_importance_path = REPORTS_DIR / "feature_importance.png"

    if feature_importance_path.exists():
        st.image(str(feature_importance_path), caption="Top 15 Feature Importances")
    else:
        st.warning("Файл feature_importance.png не найден.")

    st.subheader("Анализ порога классификации")

    threshold_plot_path = REPORTS_DIR / "threshold_analysis.png"

    if threshold_plot_path.exists():
        st.image(str(threshold_plot_path), caption="Threshold Analysis")
    else:
        st.warning("Файл threshold_analysis.png не найден.")

    if THRESHOLD_METRICS_PATH.exists():
        threshold_df = pd.read_csv(THRESHOLD_METRICS_PATH)
        st.write("Таблица метрик при разных порогах классификации:")
        st.dataframe(threshold_df, width="stretch")


def main() -> None:
    """
    Запускает веб-интерфейс Streamlit.
    """

    st.title("MLOps-пайплайн для прогнозирования оттока клиентов")

    st.write(
        "Веб-интерфейс демонстрирует работу обученной ML-модели, "
        "результаты сравнения моделей и анализ качества классификации."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Прогноз оттока",
            "Сравнение моделей",
            "Анализ модели",
        ]
    )

    with tab1:
        show_prediction_page()

    with tab2:
        show_model_comparison_page()

    with tab3:
        show_analysis_page()


if __name__ == "__main__":
    main()