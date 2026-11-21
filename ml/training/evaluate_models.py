from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "ml" / "data"

# Загрузка данных
data = pd.read_csv(DATA_DIR / "prepared_dataset.csv")
synthetic = pd.read_csv(DATA_DIR / "synthetic_appeals.csv")
test_data = pd.read_csv(DATA_DIR / "independent_test.csv")

# Воспроизведение условий предыдущих экспериментов
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=data["sentiment"]
)


# Создание модели
def create_model():
    return Pipeline([
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


# Обучение второй модели
model_v2 = create_model()
model_v2.fit(X_train, y_train)

# Обучение третьей модели
model_v3 = create_model()

X_combined = pd.concat(
    [X_train, synthetic["text"]],
    ignore_index=True
)

y_combined = pd.concat(
    [y_train, synthetic["sentiment"]],
    ignore_index=True
)

model_v3.fit(X_combined, y_combined)

# Независимое тестирование
print("CiviMind — Independent Evaluation")
print("Количество тестовых обращений:", len(test_data))

models = {
    "Model V2": model_v2,
    "Model V3": model_v3
}

results = []

for name, model in models.items():

    predictions = model.predict(test_data["text"])

    accuracy = accuracy_score(
        test_data["sentiment"], predictions
    )

    macro_f1 = f1_score(
        test_data["sentiment"],
        predictions,
        average="macro"
    )

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro F1: {macro_f1:.4f}")

    print(classification_report(
        test_data["sentiment"],
        predictions,
        zero_division=0
    ))

    print("Матрица ошибок:")
    print("Порядок классов: NEGATIVE, NEUTRAL, POSITIVE")

    print(confusion_matrix(
        test_data["sentiment"],
        predictions,
        labels=["NEGATIVE", "NEUTRAL", "POSITIVE"]
    ))

    results.append({
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

# Сохранение результатов эксперимента
pd.DataFrame(results).to_csv(
    DATA_DIR / "evaluation_results.csv",
    index=False
)

print("\nНезависимое тестирование завершено.")
