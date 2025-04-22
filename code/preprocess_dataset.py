import pandas as pd
import os

# === Определение пути до корня проекта ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "raw", "option_dataset.csv")

# === Загрузка исходного датасета ===
df = pd.read_csv(data_path)

# === One-hot encoding для типа опциона ===
df_encoded = pd.get_dummies(df, columns=["type"])

# === Формирование X и y ===
X = df_encoded.drop(columns=["price"]).fillna(0)  # заполняем NaN в barrier нулями
y = df_encoded["price"]

# === Сохранение X и y ===
processed_path = os.path.join(project_root, "data", "processed")
os.makedirs(processed_path, exist_ok=True)

X.to_csv(os.path.join(processed_path, "X.csv"), index=False)
y.to_csv(os.path.join(processed_path, "y.csv"), index=False)

print("✅ Features and target saved to data/processed/")
