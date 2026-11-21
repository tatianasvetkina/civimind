from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "ml" / "data"

# Загрузка открытого набора данных
data = pd.read_csv(
    DATA_DIR / "rusentitweet_full.csv",
    encoding="utf-8"
)

print("Исходное количество текстов:", len(data))

# Отбор необходимых классов
data = data[data["label"].isin([
    "positive",
    "neutral",
    "negative"
])].copy()

# Удаление лишних столбцов
data = data[["text", "label"]]

# Приведение меток к формату CiviMind
data = data.rename(columns={"label": "sentiment"})
data["sentiment"] = data["sentiment"].str.upper()

# Удаление пропусков и дубликатов
data = data.dropna(subset=["text"])
data["text"] = data["text"].str.strip()
data = data[data["text"] != ""]
data = data.drop_duplicates(subset=["text"])

# Сохранение обработанного набора данных
data.to_csv(
    DATA_DIR / "prepared_dataset.csv",
    index=False,
    encoding="utf-8"
)

print("Количество текстов после обработки:", len(data))
print("\nРаспределение классов:")
print(data["sentiment"].value_counts())

print("\nПодготовка данных завершена.")
