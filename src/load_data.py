import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    RAW_DATA_PATH,
    DATA_URL,
    TARGET_COLUMN,
)


def download_dataset(force: bool = False) -> None:
    """
    Загружает датасет Telco Customer Churn и сохраняет его в data/raw.
    Если файл уже существует, повторная загрузка не выполняется.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists() and not force:
        print(f"Датасет уже существует: {RAW_DATA_PATH}")
        return

    print("Загрузка датасета...")
    df = pd.read_csv(DATA_URL)
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Датасет сохранён: {RAW_DATA_PATH}")


def load_raw_data() -> pd.DataFrame:
    """
    Загружает сырой датасет из data/raw.
    Если файла нет, сначала скачивает его.
    """

    if not RAW_DATA_PATH.exists():
        download_dataset()

    df = pd.read_csv(RAW_DATA_PATH)
    return df


def print_dataset_info(df: pd.DataFrame) -> None:
    """
    Выводит краткую информацию о датасете.
    """

    print("\nИнформация о датасете:")
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

    print("\nПервые 5 строк:")
    print(df.head())

    print("\nСписок столбцов:")
    print(df.columns.tolist())

    if TARGET_COLUMN in df.columns:
        print(f"\nРаспределение целевой переменной '{TARGET_COLUMN}':")
        print(df[TARGET_COLUMN].value_counts())


def main() -> None:
    download_dataset()
    df = load_raw_data()
    print_dataset_info(df)


if __name__ == "__main__":
    main()