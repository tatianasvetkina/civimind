import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "ml" / "data"

MODEL_PATH = (
    BASE_DIR / "src" / "main" /
    "resources" / "ml" / "sentiment-model.json"
)

# Загрузка данных
data = pd.read_csv(DATA_DIR / "prepared_dataset.csv")
synthetic = pd.read_csv(DATA_DIR / "synthetic_appeals.csv")

# Воспроизведение третьего эксперимента
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=data["sentiment"]
)

X_train = pd.concat(
    [X_train, synthetic["text"]],
    ignore_index=True
)

y_train = pd.concat(
    [y_train, synthetic["sentiment"]],
    ignore_index=True
)

# Создание и обучение модели
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

model.fit(X_train, y_train)

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

# Подготовка параметров для Java
export_data = {
    "name": "CiviMind",
    "version": "3.0",
    "algorithm": "TF-IDF + Logistic Regression",
    "trainingSamples": len(X_train),
    "classes": classifier.classes_.tolist(),
    "vocabulary": {
        word: int(index)
        for word, index in vectorizer.vocabulary_.items()
    },
    "idf": vectorizer.idf_.tolist(),
    "coefficients": classifier.coef_.tolist(),
    "intercept": classifier.intercept_.tolist(),
    "lowercase": True,
    "ngramRange": [1, 2],
    "sublinearTf": True,
    "tokenPattern": vectorizer.token_pattern,
    "norm": "l2"
}

# Сохранение модели
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(MODEL_PATH, "w", encoding="utf-8") as file:
    json.dump(
        export_data,
        file,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False
    )

print("CiviMind — Model Export")
print("Версия модели: 3.0")
print("Обучающих текстов:", len(X_train))
print("Количество признаков:", len(vectorizer.vocabulary_))
print("Классы:", classifier.classes_.tolist())
print("Размер модели:", round(
    MODEL_PATH.stat().st_size / 1024 / 1024, 2
), "MB")

print("\nМодель успешно экспортирована.")
print("Путь:", MODEL_PATH)
