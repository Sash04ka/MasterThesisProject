from utils.bates_model import simulate_bates_paths
import matplotlib.pyplot as plt
import os

S, v = simulate_bates_paths(M=3, N=252, seed=None)

# Plot the paths
for i in range(3):
    plt.plot(S[i], label=f"Path {i+1}")
plt.title("Bates Simulated Price Paths")
plt.xlabel("Time Step")
plt.ylabel("Asset Price")
plt.legend()
plt.grid(True)

# Save BEFORE showing
os.makedirs("results/figs", exist_ok=True)
plt.savefig("results/figs/bates_paths.png", dpi=300)

# Then show
plt.show()

from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff, barrier_call_payoff, lookback_call_payoff
import numpy as np

# Simulate paths
S, _ = simulate_bates_paths(M=1000, N=252, T=1.0, seed=42)

# Payoffs
asian = asian_call_payoff(S, strike=100, r=0.0, T=1.0)
barrier = barrier_call_payoff(S, strike=100, barrier=130, r=0.0, T=1.0)
lookback = lookback_call_payoff(S, r=0.0, T=1.0)

# Output results
print("Asian option price:", np.mean(asian))
print("Barrier option price:", np.mean(barrier))
print("Lookback option price:", np.mean(lookback))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/raw/option_dataset.csv")

# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(14, 6))

# Plot 1: Price vs Strike
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x="K", y="price", hue="type", alpha=0.7)
plt.title("Option Price vs Strike (K)")
plt.xlabel("Strike Price (K)")
plt.ylabel("Option Price")

# Plot 2: Price vs Time to Maturity
plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x="T", y="price", hue="type", alpha=0.7)
plt.title("Option Price vs Time to Maturity (T)")
plt.xlabel("Time to Maturity (T)")
plt.ylabel("Option Price")

plt.tight_layout()

# Save figure
os.makedirs("results/figs", exist_ok=True)
plt.savefig("results/figs/price_vs_K_T.png", dpi=300)

plt.show()

# Load dataset
df = pd.read_csv("data/raw/option_dataset.csv")

# Filter only barrier options
df_barrier = df[df["type"] == "barrier"].copy()

# Create barrier bins
df_barrier["barrier_bin"] = pd.cut(df_barrier["barrier"], bins=np.arange(110, 152, 2))

# Group by bin and calculate mean price
avg_prices = df_barrier.groupby("barrier_bin")["price"].mean().reset_index()
avg_prices["barrier_center"] = avg_prices["barrier_bin"].apply(lambda x: x.mid)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(avg_prices["barrier_center"], avg_prices["price"], marker="o", color="darkorange")
plt.title("Average Barrier Option Price vs Barrier Level")
plt.xlabel("Barrier Level (binned)")
plt.ylabel("Average Option Price")
plt.grid(True)
plt.tight_layout()

# Save the plot
os.makedirs("results/figs", exist_ok=True)
plt.savefig("results/figs/avg_price_vs_barrier.png", dpi=300)
plt.show()