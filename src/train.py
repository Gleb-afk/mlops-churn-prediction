import json

import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    BEST_MODEL_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.preprocessing import prepare_train_test_data


def get_models() -> dict:
    """
    Возвращает набор моделей для сравнения.
    """

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

def log_model_to_mlflow(model_name: str, model_pipeline, metrics: dict) -> None:
    """
    Логирует модель, параметры и метрики в MLflow.
    """

    model = model_pipeline.named_steps["model"]

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)

        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            name="model",
        )

def evaluate_model(model_pipeline, X_test, y_test) -> dict:
    """
    Считает метрики качества модели.
    """

    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


def save_metrics_json(best_model_name: str, best_metrics: dict) -> None:
    """
    Сохраняет метрики лучшей модели в JSON-файл.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_data = {
        "best_model": best_model_name,
        "selection_metric": "roc_auc",
        "metrics": best_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics_data, file, indent=4, ensure_ascii=False)

    print(f"Метрики лучшей модели сохранены: {METRICS_PATH}")


def save_confusion_matrix(best_pipeline, X_test, y_test) -> None:
    """
    Сохраняет матрицу ошибок лучшей модели.
    """

    y_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"],
    )

    display.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()

    output_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Матрица ошибок сохранена: {output_path}")


def save_roc_curve(best_pipeline, X_test, y_test) -> None:
    """
    Сохраняет ROC-кривую лучшей модели.
    """

    display = RocCurveDisplay.from_estimator(best_pipeline, X_test, y_test)

    plt.title("ROC Curve")
    plt.tight_layout()

    output_path = REPORTS_DIR / "roc_curve.png"
    plt.savefig(output_path)
    plt.close()

    print(f"ROC-кривая сохранена: {output_path}")


def translate_feature_name(feature_name: str) -> str:
    """
    Переводит технические названия признаков после OneHotEncoder
    в понятные русские подписи для графика важности признаков.
    """

    column_translations = {
        "gender": "Пол",
        "SeniorCitizen": "Пожилой клиент",
        "Partner": "Наличие партнёра",
        "Dependents": "Наличие иждивенцев",
        "tenure": "Срок обслуживания",
        "PhoneService": "Телефонная связь",
        "MultipleLines": "Несколько телефонных линий",
        "InternetService": "Тип интернета",
        "OnlineSecurity": "Онлайн-безопасность",
        "OnlineBackup": "Онлайн-резервное копирование",
        "DeviceProtection": "Защита устройства",
        "TechSupport": "Техническая поддержка",
        "StreamingTV": "Стриминговое ТВ",
        "StreamingMovies": "Стриминговые фильмы",
        "Contract": "Тип договора",
        "PaperlessBilling": "Электронный счёт",
        "PaymentMethod": "Способ оплаты",
        "MonthlyCharges": "Ежемесячный платёж",
        "TotalCharges": "Общая сумма платежей",
    }

    value_translations = {
        "Female": "женский",
        "Male": "мужской",
        "Yes": "да",
        "No": "нет",
        "No internet service": "нет интернет-услуг",
        "No phone service": "нет телефонной связи",
        "DSL": "DSL",
        "Fiber optic": "оптоволокно",
        "Month-to-month": "ежемесячный",
        "One year": "один год",
        "Two year": "два года",
        "Electronic check": "электронный чек",
        "Mailed check": "чек по почте",
        "Bank transfer (automatic)": "банковский перевод",
        "Credit card (automatic)": "кредитная карта",
    }

    clean_name = feature_name

    if clean_name.startswith("numeric__"):
        clean_name = clean_name.replace("numeric__", "", 1)
        return column_translations.get(clean_name, clean_name)

    if clean_name.startswith("categorical__"):
        clean_name = clean_name.replace("categorical__", "", 1)

        if "_" in clean_name:
            column_name, value_name = clean_name.split("_", 1)
            column_ru = column_translations.get(column_name, column_name)
            value_ru = value_translations.get(value_name, value_name)
            return f"{column_ru}: {value_ru}"

        return column_translations.get(clean_name, clean_name)

    return column_translations.get(clean_name, clean_name)


def save_feature_importance(best_pipeline) -> None:
    """
    Сохраняет важность признаков для лучшей модели.
    Работает для моделей с feature_importances_ или coef_.
    """

    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        print("Для выбранной модели невозможно получить важность признаков.")
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_ru": [translate_feature_name(name) for name in feature_names],
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)

    csv_path = REPORTS_DIR / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    top_features = importance_df.head(15).sort_values(by="importance")

    plt.rcParams["font.family"] = "DejaVu Sans"

    plt.figure(figsize=(12, 8))
    plt.barh(top_features["feature_ru"], top_features["importance"])
    plt.title("Топ-15 наиболее важных признаков")
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")
    plt.tight_layout()

    png_path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Важность признаков сохранена: {csv_path}")
    print(f"График важности признаков сохранён: {png_path}")


def train_and_compare_models() -> None:
    """
    Обучает несколько моделей, сравнивает их и сохраняет лучшую.
    """

    mlflow.set_experiment("churn_prediction_experiment")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data()

    models = get_models()
    results = []
    trained_pipelines = {}

    print("\nНачинаем обучение моделей...\n")

    for model_name, model in models.items():
        print(f"Обучение модели: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        log_model_to_mlflow(model_name, pipeline, metrics)

        results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

        trained_pipelines[model_name] = pipeline

        print(
            f"{model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )
        print("-" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="roc_auc", ascending=False)

    results_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    print("\nСравнение моделей:")
    print(results_df)

    print(f"\nТаблица сравнения моделей сохранена: {MODEL_COMPARISON_PATH}")

    best_model_name = results_df.iloc[0]["model"]
    best_metrics = results_df.iloc[0].drop(labels=["model"]).to_dict()
    best_pipeline = trained_pipelines[best_model_name]

    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    print(f"\nЛучшая модель: {best_model_name}")
    print(f"Лучшая модель сохранена: {BEST_MODEL_PATH}")

    save_metrics_json(best_model_name, best_metrics)
    save_confusion_matrix(best_pipeline, X_test, y_test)
    save_roc_curve(best_pipeline, X_test, y_test)
    save_feature_importance(best_pipeline)


def main() -> None:
    train_and_compare_models()


if __name__ == "__main__":
    main()