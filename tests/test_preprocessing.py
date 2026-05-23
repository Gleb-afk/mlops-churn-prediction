from src.config import TARGET_COLUMN
from src.load_data import load_raw_data
from src.preprocessing import clean_data, get_features_and_target, prepare_train_test_data


def test_clean_data_removes_customer_id():
    """
    Проверяет, что после очистки удаляется столбец customerID.
    """

    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)

    assert "customerID" not in clean_df.columns


def test_clean_data_converts_target_to_numbers():
    """
    Проверяет, что целевая переменная Churn преобразуется в 0 и 1.
    """

    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)

    unique_values = set(clean_df[TARGET_COLUMN].unique())

    assert unique_values.issubset({0, 1})


def test_total_charges_has_no_missing_values_after_cleaning():
    """
    Проверяет, что после очистки в TotalCharges нет пропущенных значений.
    """

    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)

    assert clean_df["TotalCharges"].isna().sum() == 0


def test_get_features_and_target():
    """
    Проверяет разделение данных на признаки и целевую переменную.
    """

    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)

    X, y = get_features_and_target(clean_df)

    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y)


def test_prepare_train_test_data():
    """
    Проверяет подготовку train/test-выборок.
    """

    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert preprocessor is not None