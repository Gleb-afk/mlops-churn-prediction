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
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)

    csv_path = REPORTS_DIR / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    top_features = importance_df.head(15).sort_values(by="importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()

    png_path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(png_path)
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