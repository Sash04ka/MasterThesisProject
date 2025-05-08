import pandas as pd
import os

# Load dataset
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))  # Current project directory
dataset_path = os.path.join(project_root, "data", "raw", "option_dataset.csv")

df = pd.read_csv(dataset_path)

# Print statistics
print("--- Volatility Feature Statistics ---")
print("vol_mean:")
print(df["vol_mean"].describe())

print("\nvol_T:")
print(df["vol_T"].describe())

print("\nvol_std:")
print(df["vol_std"].describe())
