import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
)

from src.load_data import load_raw_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет базовую очистку датасета:
    - удаляет customerID;
    - преобразует TotalCharges в числовой формат;
    - удаляет строки с пропусками в TotalCharges;
    - преобразует Churn в 0 и 1.
    """

    df = df.copy()

    # Удаляем идентификатор клиента, так как он не несёт полезной информации для модели
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges в датасете хранится как текст, поэтому переводим его в число
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # После преобразования могут появиться пропуски, удаляем такие строки
    df = df.dropna(subset=["TotalCharges"])

    # Целевая переменная: No -> 0, Yes -> 1
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    return df


def save_processed_data(df: pd.DataFrame) -> None:
    """
    Сохраняет очищенный датасет в data/processed.
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Очищенный датасет сохранён: {PROCESSED_DATA_PATH}")


def get_features_and_target(df: pd.DataFrame):
    """
    Разделяет датасет на признаки X и целевую переменную y.
    """

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Создаёт обработчик признаков:
    - числовые признаки заполняются медианой и масштабируются;
    - категориальные признаки заполняются самым частым значением и кодируются OneHotEncoder.
    """

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def prepare_train_test_data():
    """
    Полный цикл подготовки данных:
    - загрузка;
    - очистка;
    - сохранение очищенного датасета;
    - разделение на train/test;
    - создание preprocessor.
    """

    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)

    save_processed_data(clean_df)

    X, y = get_features_and_target(clean_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor


def main() -> None:
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)

    X, y = get_features_and_target(clean_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    save_processed_data(clean_df)

    print("\nПредобработка данных завершена.")
    print(f"Размер исходного датасета: {raw_df.shape[0]} строк, {raw_df.shape[1]} столбцов")
    print(f"Размер очищенного датасета: {clean_df.shape[0]} строк, {clean_df.shape[1]} столбцов")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Размер обучающей выборки: {X_train.shape[0]} строк")
    print(f"Размер тестовой выборки: {X_test.shape[0]} строк")

    print("\nЧисловые признаки:")
    print(numeric_features)

    print("\nКатегориальные признаки:")
    print(categorical_features)

    print(f"\nРаспределение целевой переменной '{TARGET_COLUMN}':")
    print(y.value_counts())


if __name__ == "__main__":
    main()