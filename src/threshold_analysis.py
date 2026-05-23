import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.config import (
    BEST_MODEL_PATH,
    REPORTS_DIR,
    THRESHOLD_METRICS_PATH,
    THRESHOLD_PLOT_PATH,
)
from src.preprocessing import prepare_train_test_data


def calculate_metrics_for_thresholds(model, X_test, y_test) -> pd.DataFrame:
    """
    Считает метрики качества модели для разных порогов классификации.
    """

    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = [round(x / 100, 2) for x in range(10, 91, 5)]
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            }
        )

    return pd.DataFrame(results)


def save_threshold_plot(results_df: pd.DataFrame) -> None:
    """
    Сохраняет график зависимости метрик от порога классификации.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(results_df["threshold"], results_df["precision"], marker="o", label="Precision")
    plt.plot(results_df["threshold"], results_df["recall"], marker="o", label="Recall")
    plt.plot(results_df["threshold"], results_df["f1"], marker="o", label="F1-score")

    plt.title("Threshold Analysis")
    plt.xlabel("Classification Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(THRESHOLD_PLOT_PATH)
    plt.close()

    print(f"График анализа порога сохранён: {THRESHOLD_PLOT_PATH}")


def print_recommendations(results_df: pd.DataFrame) -> None:
    """
    Выводит рекомендации по выбору порога.
    """

    best_f1_row = results_df.loc[results_df["f1"].idxmax()]

    high_recall_df = results_df[results_df["recall"] >= 0.75]

    print("\nЛучший порог по F1-score:")
    print(best_f1_row)

    if not high_recall_df.empty:
        business_row = high_recall_df.sort_values(
            by=["precision", "f1"],
            ascending=False,
        ).iloc[0]

        print("\nРекомендуемый бизнес-порог при recall >= 0.75:")
        print(business_row)
    else:
        print("\nНе найден порог, при котором recall >= 0.75")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка лучшей модели...")
    model = joblib.load(BEST_MODEL_PATH)

    print("Подготовка тестовых данных...")
    _, X_test, _, y_test, _ = prepare_train_test_data()

    print("Расчёт метрик для разных порогов...")
    results_df = calculate_metrics_for_thresholds(model, X_test, y_test)

    results_df.to_csv(THRESHOLD_METRICS_PATH, index=False)
    print(f"Таблица анализа порогов сохранена: {THRESHOLD_METRICS_PATH}")

    print("\nТаблица анализа порогов:")
    print(results_df)

    save_threshold_plot(results_df)
    print_recommendations(results_df)


if __name__ == "__main__":
    main()