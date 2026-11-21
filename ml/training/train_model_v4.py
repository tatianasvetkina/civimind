from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline, FeatureUnion


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "ml" / "data"

# Загрузка данных
data = pd.read_csv(DATA_DIR / "prepared_dataset.csv")
synthetic = pd.read_csv(DATA_DIR / "synthetic_appeals.csv")
independent = pd.read_csv(DATA_DIR / "independent_test.csv")

# Разделение открытого набора данных
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=data["sentiment"]
)

# Добавление синтетических обращений
X_train = pd.concat(
    [X_train, synthetic["text"]],
    ignore_index=True
)

y_train = pd.concat(
    [y_train, synthetic["sentiment"]],
    ignore_index=True
)

# Объединение словесных и символьных признаков
features = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        sublinear_tf=True
    )),
    ("char_tfidf", TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(3, 5),
        max_features=40000,
        min_df=2,
        sublinear_tf=True
    ))
])

# Создание модели
model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Обучение
print("CiviMind — Experiment 4")
print("Обучающая выборка:", len(X_train))
print("Обучение модели...")

model.fit(X_train, y_train)


# Оценка качества
def evaluate(name, texts, labels):
    predictions = model.predict(texts)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(labels, predictions):.2%}")
    print(f"Macro F1: {f1_score(labels, predictions, average='macro'):.4f}")

    print(classification_report(
        labels,
        predictions,
        zero_division=0
    ))

    print("Матрица ошибок:")
    print(confusion_matrix(
        labels,
        predictions,
        labels=["NEGATIVE", "NEUTRAL", "POSITIVE"]
    ))


evaluate("RuSentiTweet", X_test, y_test)

evaluate(
    "Проверка на новых обращениях",
    independent["text"],
    independent["sentiment"]
)

print("\nЭксперимент завершен.")
