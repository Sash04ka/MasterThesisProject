import pandas as pd
import os

# Compute project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "raw", "option_dataset.csv")

# Load dataset
df = pd.read_csv(data_path)

# One-hot encode the 'type' column
df_encoded = pd.get_dummies(df, columns=["type"])

# Define X and y
X = df_encoded.drop(columns=["price"])
y = df_encoded["price"]

# Save to processed folder
processed_path = os.path.join(project_root, "data", "processed")
os.makedirs(processed_path, exist_ok=True)

X.to_csv(os.path.join(processed_path, "X.csv"), index=False)
y.to_csv(os.path.join(processed_path, "y.csv"), index=False)

print("✅ Features and target saved to data/processed/")
