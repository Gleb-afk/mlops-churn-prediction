# MLOps-пайплайн для прогнозирования оттока клиентов

Курсовой проект на тему: **«Разработка MLOps-пайплайна для прогнозирования оттока клиентов с веб-интерфейсом»**.

Проект демонстрирует полный цикл работы с ML-моделью: загрузку и предобработку данных, обучение нескольких моделей, сравнение метрик, выбор лучшей модели, логирование экспериментов, создание API, веб-интерфейса, тестов, Docker-контейнеров и CI/CD-пайплайна.

## Цель проекта

Цель проекта — разработать практический MLOps-пайплайн, который автоматизирует основные этапы жизненного цикла модели машинного обучения:

- подготовку данных;
- обучение моделей;
- сравнение качества моделей;
- выбор лучшей модели;
- сохранение модели;
- предоставление REST API для предсказаний;
- создание веб-интерфейса;
- автоматическое тестирование;
- контейнеризацию;
- запуск CI/CD-пайплайна через GitHub Actions.

## Используемый датасет

В проекте используется датасет **Telco Customer Churn**.

Задача модели — предсказать, уйдёт клиент из компании или останется.

Целевая переменная:

```text
Churn
```

Возможные значения:

```text
No  — клиент не ушёл
Yes — клиент ушёл
```

После предобработки значения преобразуются:

```text
No  → 0
Yes → 1
```

## Стек технологий

| Инструмент | Назначение |
|---|---|
| Python | основной язык проекта |
| pandas, numpy | обработка данных |
| scikit-learn | обучение ML-моделей |
| MLflow | логирование экспериментов |
| FastAPI | REST API для предсказаний |
| Streamlit | веб-интерфейс |
| pytest | автоматические тесты |
| Docker | контейнеризация |
| Docker Compose | запуск нескольких сервисов |
| GitHub Actions | CI/CD-пайплайн |

## Структура проекта

```text
mlops-churn-prediction/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── threshold_metrics.csv
│   └── threshold_analysis.png
│
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── threshold_analysis.py
│   └── predict.py
│
├── tests/
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_training.py
│
├── web/
│   └── app.py
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── Dockerfile.api
├── Dockerfile.web
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── README.md
```

## Обучение моделей

В проекте сравниваются следующие модели:

- Logistic Regression;
- Decision Tree;
- Random Forest;
- Gradient Boosting.

Для сравнения используются метрики:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC.

Лучшая модель выбирается по метрике **ROC-AUC**.

По результатам экспериментов лучшей моделью стала:

```text
Gradient Boosting
```

## Анализ порога классификации

Помимо стандартного порога классификации `0.5`, в проекте проводится анализ разных порогов.

Для бизнес-задачи прогнозирования оттока клиентов был выбран порог:

```text
0.30
```

Такой порог позволяет повысить полноту `recall`, то есть находить больше клиентов, которые действительно могут уйти.

## Запуск проекта локально

### 1. Создание виртуального окружения

```powershell
python -m venv .venv
```

### 2. Активация окружения

```powershell
.\.venv\Scripts\activate
```

### 3. Установка зависимостей

```powershell
pip install -r requirements.txt
```

### 4. Загрузка данных

```powershell
python -m src.load_data
```

### 5. Предобработка данных

```powershell
python -m src.preprocessing
```

### 6. Обучение моделей

```powershell
python -m src.train
```

### 7. Анализ порогов классификации

```powershell
python -m src.threshold_analysis
```

### 8. Проверка предсказания

```powershell
python -m src.predict
```

## Запуск FastAPI

```powershell
uvicorn api.main:app --reload
```

После запуска API доступно по адресу:

```text
http://127.0.0.1:8000
```

Swagger-документация:

```text
http://127.0.0.1:8000/docs
```

Основные endpoints:

```text
GET  /
GET  /health
GET  /model-info
POST /predict
```

## Запуск веб-интерфейса Streamlit

Во втором терминале:

```powershell
streamlit run web/app.py
```

Веб-интерфейс доступен по адресу:

```text
http://localhost:8501
```

В интерфейсе доступны вкладки:

- прогноз оттока клиента;
- сравнение моделей;
- анализ лучшей модели.

## Запуск через Docker Compose

Сборка и запуск контейнеров:

```powershell
docker compose up --build
```

После запуска доступны:

```text
FastAPI:   http://localhost:8000/docs
Streamlit: http://localhost:8501
```

Остановка контейнеров:

```powershell
docker compose down
```

## Тестирование

Для запуска тестов:

```powershell
pytest
```

В проекте реализованы тесты для:

- предобработки данных;
- проверки сохранённых моделей и отчётов;
- проверки FastAPI endpoints;
- проверки endpoint `/predict`.

Пример успешного результата:

```text
14 passed
```

## MLflow

MLflow используется для логирования экспериментов:

- названия модели;
- параметров модели;
- метрик качества;
- сохранённой модели.

Запуск MLflow UI:

```powershell
mlflow server --port 5000
```

Интерфейс доступен по адресу:

```text
http://127.0.0.1:5000
```

Название эксперимента:

```text
churn_prediction_experiment
```

## CI/CD

В проекте настроен GitHub Actions workflow:

```text
.github/workflows/ci-cd.yml
```

CI/CD-пайплайн выполняет:

1. загрузку кода из репозитория;
2. установку Python;
3. установку зависимостей;
4. обучение моделей;
5. анализ порогов;
6. запуск тестов;
7. сборку Docker-образов API и Web;
8. сохранение модели и отчётов как artifacts.

Workflow запускается при `push` или `pull request` в ветки:

```text
main
master
```

## Итог

В результате был разработан MLOps-пайплайн, который автоматизирует жизненный цикл ML-модели для прогнозирования оттока клиентов. Проект включает обучение и сравнение моделей, логирование экспериментов, API, веб-интерфейс, тестирование, контейнеризацию и CI/CD-пайплайн.
