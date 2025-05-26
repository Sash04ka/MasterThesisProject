import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from utils.bates_model import simulate_bates_paths
from utils.payoffs import lookback_call_payoff
from utils.implied_volatility import implied_volatility_call

# Constants
S0 = 100
r = 0.02
M_MC = 1000

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

K_choices = np.arange(80, 125, 5)
T_choices = [1/12, 3/12, 6/12, 9/12, 1.0]
moneyness_grid = [0.8, 0.9, 1.0, 1.1, 1.2]

# Loosened filters for lookback
price_threshold = 0.5    # lower than before
vol_min = 0.03
vol_max = 1.5

target_filtered_samples = 3000
existing_samples_path = os.path.join(data_dir, "option_dataset_lookback_filtered.csv")

# Load existing filtered lookback dataset
if os.path.exists(existing_samples_path):
    df_existing = pd.read_csv(existing_samples_path)
    existing_count = len(df_existing)
else:
    df_existing = pd.DataFrame()
    existing_count = 0

samples_needed = max(target_filtered_samples - existing_count, 0)
print(f"Need to generate {samples_needed} more filtered lookback samples.")

if samples_needed == 0:
    print("No additional samples needed.")
    exit(0)

records = []

pbar = tqdm(total=samples_needed, desc="Generating additional filtered lookback samples")

while len(records) < samples_needed:
    K = np.random.choice(K_choices)
    T = np.random.choice(T_choices)
    N = int(252 * T)

    # Bates parameters randomized
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

    price = np.mean(lookback_call_payoff(S, strike=K, r=r, T=T))

    smile = {}
    for m in moneyness_grid:
        for t in T_choices:
            K_m = S0 / m
            payoff_m = lookback_call_payoff(S, strike=K_m, r=r, T=t)
            price_m = np.mean(payoff_m)
            iv = implied_volatility_call(price_m, S0, K_m, t, r)
            smile[f"sigma_{int(m * 100):03d}_{int(t * 12)}m"] = iv

    # Apply loosened filtering
    if price <= price_threshold:
        continue
    if any(np.isnan(list(smile.values()))):
        continue
    if not all(vol_min <= v <= vol_max for v in smile.values()):
        continue

    row = {
        "type": "lookback",
        "K": K,
        "T": T,
        "r": r,
        "price": price,
        **smile
    }

    records.append(row)
    pbar.update(1)

pbar.close()

df_new = pd.DataFrame(records)

# Combine with existing dataset if any
if existing_count > 0:
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_combined = df_new

output_path = os.path.join(data_dir, "option_dataset_lookback_filtered.csv")
df_combined.to_csv(output_path, index=False)
print(f"✅ Saved combined lookback dataset with {len(df_combined)} samples to {output_path}")
