from pathlib import Path

# Корневая папка проекта
ROOT_DIR = Path(__file__).resolve().parents[1]

# Папки проекта
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Пути к файлам
RAW_DATA_PATH = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_churn.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
THRESHOLD_METRICS_PATH = REPORTS_DIR / "threshold_metrics.csv"
THRESHOLD_PLOT_PATH = REPORTS_DIR / "threshold_analysis.png"

# Ссылка на датасет
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

# Настройки ML

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Порог классификации для бизнес-сценария оттока клиентов
CHURN_THRESHOLD = 0.30