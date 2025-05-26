import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = r"/data/option_dataset_asian_filtered.csv"
df = pd.read_csv(file_path)

moneyness_levels = [0.8, 0.9, 1.0, 1.1, 1.2]
maturity_months = [1, 3, 6, 9, 12]

def has_complete_smile(sample):
    for t in maturity_months:
        for m in moneyness_levels:
            col_name = f"sigma_{int(m * 100):03d}_{t}m"
            if pd.isna(sample.get(col_name)):
                return False
    return True

def plot_smile_for_sample(sample_idx):
    sample = df.iloc[sample_idx]
    vols = []

    for t in maturity_months:
        vol_row = []
        for m in moneyness_levels:
            col_name = f"sigma_{int(m * 100):03d}_{t}m"
            vol = sample.get(col_name, np.nan)
            vol_row.append(vol)
        vols.append(vol_row)

    price = sample['price']
    plt.figure(figsize=(10, 6))
    for i, t in enumerate(maturity_months):
        y = vols[i]
        plt.plot(moneyness_levels, y, marker='o', label=f"{t} months", linestyle='-')

    plt.title(f"Volatility Smile for Sample #{sample_idx + 1} — Price: {price:.4f}")
    plt.xlabel("Moneyness (S/K)")
    plt.ylabel("Implied Volatility")
    plt.xticks(moneyness_levels)
    plt.xlim(min(moneyness_levels), max(moneyness_levels))
    plt.ylim(0, max(np.nanmax(vols), 0.1))
    plt.grid(True)
    plt.legend()
    plt.show()

def interactive_plot():
    n_samples = len(df)
    idx = 0

    while idx < n_samples:
        if has_complete_smile(df.iloc[idx]):
            print(f"Plotting sample {idx + 1} of {n_samples} with price {df.iloc[idx]['price']:.4f}")
            plot_smile_for_sample(idx)
            cont = input("Show next sample? (y/n): ").strip().lower()
            if cont != 'y':
                print("Exiting plot viewer.")
                break
        idx += 1

if __name__ == "__main__":
    interactive_plot()
