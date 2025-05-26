import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff
from utils.implied_volatility import implied_volatility_call  # Assuming this is your util

# === Constants ===
S0 = 100
r = 0.02
M_MC = 1000

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

# === Discrete grids ===
K_choices = np.arange(80, 125, 5)
T_choices = [1/12, 3/12, 6/12, 9/12, 1.0]
moneyness_grid = [0.8, 0.9, 1.0, 1.1, 1.2]

# === Filtering criteria ===
price_threshold = 1.0
vol_min = 0.05
vol_max = 1.0
min_filtered_samples = 100

records = []
filtered_count = 0
total_attempts = 0
max_attempts = 10000  # safety cap to avoid infinite loops

pbar = tqdm(total=min_filtered_samples, desc="Generating filtered Asian samples")

while filtered_count < min_filtered_samples and total_attempts < max_attempts:
    total_attempts += 1

    K = np.random.choice(K_choices)
    T = np.random.choice(T_choices)
    N = int(252 * T)

    # Bates params randomized
    v0 = np.random.uniform(0.01, 0.09)
    theta = np.random.uniform(0.01, 0.09)
    sigma_v = np.random.uniform(0.1, 0.6)
    kappa = np.random.uniform(0.5, 4.0)
    rho = np.random.uniform(-0.9, -0.1)
    lambda_jump = np.random.uniform(0.01, 0.2)
    mu_jump = np.random.uniform(-0.1, 0.0)
    sigma_jump = np.random.uniform(0.1, 0.4)

    S, v = simulate_bates_paths(
        M=M_MC, N=N, T=T, S0=S0, r=r,
        v0=v0, theta=theta, sigma_v=sigma_v, kappa=kappa,
        rho=rho, lambda_jump=lambda_jump,
        mu_jump=mu_jump, sigma_jump=sigma_jump
    )

    price = np.mean(asian_call_payoff(S, strike=K, r=r, T=T))

    # Smile features: implied vols at fixed moneyness & maturities
    smile = {}
    for m in moneyness_grid:
        for t in T_choices:
            K_m = S0 / m
            payoff_m = asian_call_payoff(S, strike=K_m, r=r, T=t)
            price_m = np.mean(payoff_m)
            iv = implied_volatility_call(price_m, S0, K_m, t, r)
            smile[f"sigma_{int(m * 100):03d}_{int(t * 12)}m"] = iv

    # Filter criteria check
    if price <= price_threshold:
        continue  # discard low price

    # Check for NaN in smile features
    if any(np.isnan(list(smile.values()))):
        continue

    # Check smile vol range
    if not all(vol_min <= v <= vol_max for v in smile.values()):
        continue

    # Passed all filters - save sample
    row = {
        "type": "asian",
        "K": K,
        "T": T,
        "r": r,
        "price": price,
        **smile
    }
    records.append(row)
    filtered_count += 1
    pbar.update(1)

pbar.close()

if filtered_count < min_filtered_samples:
    print(f"Warning: only {filtered_count} samples generated after {total_attempts} attempts.")

df_filtered = pd.DataFrame(records)
output_path = os.path.join(data_dir, "option_dataset_asian_filtered.csv")
df_filtered.to_csv(output_path, index=False)
print(f"✅ Filtered dataset saved to {output_path} with {filtered_count} samples.")
