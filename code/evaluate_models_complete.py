import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Setup paths ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
visual_dir = os.path.join(project_root, "results", "visualizations")
os.makedirs(visual_dir, exist_ok=True)

option_types = ["asian", "barrier", "lookback"]

def dataset_overview(option_type):
    print(f"Generating dataset overview for {option_type}...")

    # Smart dataset file resolution
    path_random = os.path.join(data_dir, f"option_dataset_{option_type}_random.csv")
    path_filtered = os.path.join(data_dir, f"option_dataset_{option_type}_filtered.csv")

    if os.path.exists(path_random):
        df = pd.read_csv(path_random)
        print(f"✅ Loaded random dataset: {path_random}")
    elif os.path.exists(path_filtered):
        df = pd.read_csv(path_filtered)
        print(f"✅ Loaded filtered dataset: {path_filtered}")
    else:
        raise FileNotFoundError(f"❌ No dataset found for {option_type}. Checked: _random and _filtered.")

    # Derived features if needed
    df["log_moneyness"] = np.log(df["K"] / 100)
    df["sqrt_T"] = np.sqrt(df["T"])

    # === Pair plot of key variables
    cols = ["K", "T", "r", "log_moneyness", "sqrt_T", "price"]
    if option_type == "barrier":
        cols.append("barrier")

    plt.figure(figsize=(10, 6))
    sns.histplot(df["price"], bins=40, kde=True)
    plt.title(f"{option_type.capitalize()} Option Price Distribution")
    plt.xlabel("Price")
    plt.tight_layout()
    plt.savefig(os.path.join(visual_dir, f"{option_type}_price_distribution.png"))
    plt.close()

    sns.pairplot(df[cols].sample(n=min(500, len(df)))).fig.suptitle(f"{option_type.capitalize()} – Pairwise Overview", y=1.02)
    plt.savefig(os.path.join(visual_dir, f"{option_type}_pairplot.png"))
    plt.close()

# === Run all
if __name__ == "__main__":
    for opt_type in option_types:
        dataset_overview(opt_type)
