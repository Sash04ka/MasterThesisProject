import pandas as pd
import os

# === Define project root path ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "raw", "option_dataset.csv")

# === Load raw dataset ===
df = pd.read_csv(data_path)

# === One-hot encoding for option type ===
df_encoded = pd.get_dummies(df, columns=["type"])

# === Create features X and target y ===
X = df_encoded.drop(columns=["price"]).fillna(0)  # fill NaN in barrier with 0
y = df_encoded["price"]

# === Save X and y ===
processed_path = os.path.join(project_root, "data", "processed")
os.makedirs(processed_path, exist_ok=True)

X.to_csv(os.path.join(processed_path, "X.csv"), index=False)
y.to_csv(os.path.join(processed_path, "y.csv"), index=False)

print("✅ Features and target saved to data/processed/")
