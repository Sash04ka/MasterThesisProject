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
