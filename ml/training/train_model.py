from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


# Загрузка набора данных
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "ml" / "data" / "citizen_appeals.csv"

data = pd.read_csv(DATA_PATH, encoding="utf-8")

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=data["sentiment"]
)

# Создание модели машинного обучения
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Обучение модели
model.fit(X_train, y_train)

# Оценка качества модели
predictions = model.predict(X_test)

print("CiviMind — Sentiment Analysis")
print("Всего обращений:", len(data))
print("Обучающая выборка:", len(X_train))
print("Тестовая выборка:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nМетрики классификации:")
print(classification_report(y_test, predictions, zero_division=0))

# Проверка модели на новых обращениях
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
