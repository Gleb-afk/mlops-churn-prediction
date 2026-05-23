from src.config import (
    BEST_MODEL_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    REPORTS_DIR,
)
from src.train import get_models


def test_get_models_returns_expected_models():
    """
    Проверяет, что в проекте определены основные модели для сравнения.
    """

    models = get_models()

    assert "Logistic Regression" in models
    assert "Decision Tree" in models
    assert "Random Forest" in models
    assert "Gradient Boosting" in models


def test_best_model_file_exists():
    """
    Проверяет, что после обучения сохранён файл лучшей модели.
    """

    assert BEST_MODEL_PATH.exists()


def test_model_comparison_file_exists():
    """
    Проверяет, что после обучения сохранена таблица сравнения моделей.
    """

    assert MODEL_COMPARISON_PATH.exists()


def test_metrics_file_exists():
    """
    Проверяет, что после обучения сохранён файл с метриками лучшей модели.
    """

    assert METRICS_PATH.exists()


def test_report_images_exist():
    """
    Проверяет, что после обучения сохранены основные графики.
    """

    assert (REPORTS_DIR / "confusion_matrix.png").exists()
    assert (REPORTS_DIR / "roc_curve.png").exists()
    assert (REPORTS_DIR / "feature_importance.png").exists()
    assert (REPORTS_DIR / "threshold_analysis.png").exists()