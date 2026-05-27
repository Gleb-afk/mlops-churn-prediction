import json

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import BEST_MODEL_PATH, CHURN_THRESHOLD, REPORTS_DIR
from src.preprocessing import prepare_train_test_data


EVALUATION_PATH = REPORTS_DIR / "evaluation.json"


def calculate_metrics(y_true, y_pred, y_proba) -> dict:
    """
    Рассчитывает основные метрики качества классификации.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def evaluate_model(model, X_test, y_test, threshold: float) -> dict:
    """
    Оценивает модель на тестовой выборке при заданном пороге классификации.
    """

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = calculate_metrics(y_test, y_pred, y_proba)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Клиент остался", "Клиент ушёл"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "threshold": threshold,
        "metrics": metrics,
        "classification_report": report,
    }


def save_evaluation_report(report: dict) -> None:
    """
    Сохраняет отчёт об оценке модели в JSON-файл.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVALUATION_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print(f"Отчёт об оценке модели сохранён: {EVALUATION_PATH}")


def main() -> None:
    """
    Загружает лучшую модель, оценивает её качество на тестовой выборке
    и сохраняет результат в reports/evaluation.json.
    """

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {BEST_MODEL_PATH}. "
            "Сначала запустите обучение командой: python -m src.train"
        )

    print("Загрузка лучшей модели...")
    model = joblib.load(BEST_MODEL_PATH)

    print("Подготовка тестовых данных...")
    _, X_test, _, y_test, _ = prepare_train_test_data()

    print("Оценка модели при стандартном пороге 0.5...")
    default_threshold_report = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=0.5,
    )

    print(f"Оценка модели при бизнес-пороге {CHURN_THRESHOLD}...")
    business_threshold_report = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=CHURN_THRESHOLD,
    )

    evaluation_report = {
        "model_path": str(BEST_MODEL_PATH),
        "default_threshold": default_threshold_report,
        "business_threshold": business_threshold_report,
    }

    save_evaluation_report(evaluation_report)

    print("\nМетрики при стандартном пороге 0.5:")
    print(default_threshold_report["metrics"])

    print(f"\nМетрики при бизнес-пороге {CHURN_THRESHOLD}:")
    print(business_threshold_report["metrics"])


if __name__ == "__main__":
    main()