import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# === Bates Model ===
def simulate_bates_paths(
    S0=100, v0=0.04, r=0.0,
    kappa=2.0, theta=0.04, sigma_v=0.3,
    rho=-0.7, lambda_jump=0.1, mu_jump=-0.05, sigma_jump=0.2,
    T=1.0, N=252, M=1000, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    S = np.zeros((M, N + 1))
    v = np.zeros((M, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(1, N + 1):
        z1 = np.random.normal(size=M)
        z2 = np.random.normal(size=M)
        dW_S = z1
        dW_v = rho * z1 + np.sqrt(1 - rho**2) * z2

        v[:, t] = np.abs(
            v[:, t - 1]
            + kappa * (theta - v[:, t - 1]) * dt
            + sigma_v * np.sqrt(np.maximum(v[:, t - 1], 0) * dt) * dW_v
        )

        jumps = np.random.poisson(lambda_jump * dt, M)
        jump_sizes = np.random.normal(mu_jump, sigma_jump, M)

        S[:, t] = S[:, t - 1] * np.exp(
            (r - 0.5 * v[:, t - 1]) * dt
            + np.sqrt(np.maximum(v[:, t - 1], 0) * dt) * dW_S
            + jumps * jump_sizes
        )

    return S

# === Payoff functions ===
def asian_call_payoff(price_paths, strike, r, T):
    avg = np.mean(price_paths[:, 1:], axis=1)
    return np.exp(-r * T) * np.maximum(avg - strike, 0)

def barrier_call_payoff(price_paths, strike, barrier, r, T):
    S_T = price_paths[:, -1]
    max_price = np.max(price_paths, axis=1)
    mask = max_price < barrier
    payoffs = np.where(mask, np.maximum(S_T - strike, 0), 0.0)
    return np.exp(-r * T) * payoffs

def lookback_call_payoff(price_paths, r, T):
    S_T = price_paths[:, -1]
    S_min = np.min(price_paths, axis=1)
    return np.exp(-r * T) * np.maximum(S_T - S_min, 0)

# === Dataset Generation ===
records = []
N_SAMPLES = 500  # можно увеличить позже

for _ in tqdm(range(N_SAMPLES)):
    K = np.random.uniform(80, 120)
    T = np.random.uniform(0.25, 1.5)
    barrier = np.random.uniform(110, 150)
    S = simulate_bates_paths(M=1000, N=252, T=T)

    price_asian = np.mean(asian_call_payoff(S, strike=K, r=0.0, T=T))
    records.append({'K': K, 'T': T, 'barrier': np.nan, 'type': 'asian', 'price': price_asian})

    price_barrier = np.mean(barrier_call_payoff(S, strike=K, barrier=barrier, r=0.0, T=T))
    records.append({'K': K, 'T': T, 'barrier': barrier, 'type': 'barrier', 'price': price_barrier})

    price_lookback = np.mean(lookback_call_payoff(S, r=0.0, T=T))
    records.append({'K': K, 'T': T, 'barrier': np.nan, 'type': 'lookback', 'price': price_lookback})

df = pd.DataFrame(records)

# Save to data/raw
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_path = os.path.join(project_root, 'data', 'raw', 'option_dataset.csv')

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Dataset saved to {output_path}")
