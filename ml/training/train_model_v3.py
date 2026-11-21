from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "ml" / "data"

# Загрузка открытого датасета
data = pd.read_csv(
    DATA_DIR / "prepared_dataset.csv",
    encoding="utf-8"
)

# Сохранение условий второго эксперимента
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=data["sentiment"]
)

# Загрузка синтетических обращений
synthetic = pd.read_csv(
    DATA_DIR / "synthetic_appeals.csv",
    encoding="utf-8"
)

# Объединение обучающих данных
X_train = pd.concat(
    [X_train, synthetic["text"]],
    ignore_index=True
)

y_train = pd.concat(
    [y_train, synthetic["sentiment"]],
    ignore_index=True
)

# Создание модели
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Обучение модели
model.fit(X_train, y_train)

# Оценка на тестовой выборке RuSentiTweet
predictions = model.predict(X_test)

print("CiviMind — Experiment 3")
print("Обучающая выборка:", len(X_train))
print("Тестовая выборка:", len(X_test))
print("Accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("\nМетрики классификации:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Проверка на обращениях граждан
appeals = pd.read_csv(
    DATA_DIR / "citizen_appeals.csv",
    encoding="utf-8"
)

appeal_predictions = model.predict(appeals["text"])

print("\nПроверка на обращениях граждан:")
print("Accuracy:", round(
    accuracy_score(appeals["sentiment"], appeal_predictions), 4
))

print(classification_report(
    appeals["sentiment"],
    appeal_predictions,
    zero_division=0
))

# Демонстрационные примеры
examples = [
    "Спасибо за прекрасную работу",
    "Прошу сообщить время приема",
    "Возмущена качеством обслуживания"
]

print("\nПримеры предсказаний:")

for text in examples:
    sentiment = model.predict([text])[0]
    confidence = model.predict_proba([text])[0].max()

    print(f"{text} -> {sentiment} ({confidence:.2%})")
