import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff, barrier_call_payoff, lookback_call_payoff
from utils.implied_volatility import implied_volatility_call

# === Constants ===
S0 = 100
r = 0.02
M_MC = 1000
sample_target = 200
max_attempts_per_type = 10000

K_choices = np.arange(80, 125, 5)
T_choices = [1/12, 3/12, 6/12, 9/12, 1.0]
moneyness_grid = [0.8, 0.9, 1.0, 1.1, 1.2]
price_threshold = 1.0
vol_min = 0.05
vol_max = 1.0

option_types = ['asian', 'barrier', 'lookback']

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

def compute_payoff(option_type, S, K, r, T, barrier=None):
    if option_type == 'asian':
        return asian_call_payoff(S, strike=K, r=r, T=T)
    elif option_type == 'barrier':
        return barrier_call_payoff(S, strike=K, barrier=barrier, r=r, T=T)
    elif option_type == 'lookback':
        return lookback_call_payoff(S, strike=K, r=r, T=T)
    else:
        raise ValueError("Unknown option type")

for option_type in option_types:
    records = []
    attempts = 0
    pbar = tqdm(total=sample_target, desc=f"Generating {option_type}_random")

    while len(records) < sample_target and attempts < max_attempts_per_type:
        attempts += 1
        K = np.random.choice(K_choices)
        T = np.random.choice(T_choices)
        N = int(252 * T)

        # Random Bates parameters
        v0 = np.random.uniform(0.01, 0.09)
        theta = np.random.uniform(0.01, 0.09)
        sigma_v = np.random.uniform(0.1, 0.6)
        kappa = np.random.uniform(0.5, 4.0)
        rho = np.random.uniform(-0.9, -0.1)
        lambda_jump = np.random.uniform(0.01, 0.2)
        mu_jump = np.random.uniform(-0.1, 0.0)
        sigma_jump = np.random.uniform(0.1, 0.4)

        S, _ = simulate_bates_paths(
            M=M_MC, N=N, T=T, S0=S0, r=r,
            v0=v0, theta=theta, sigma_v=sigma_v, kappa=kappa,
            rho=rho, lambda_jump=lambda_jump,
            mu_jump=mu_jump, sigma_jump=sigma_jump
        )

        barrier = None
        if option_type == 'barrier':
            raw_barrier = np.random.uniform(1.1, 1.4) * max(S0, K)
            barrier = np.clip(raw_barrier, 105, 180)

        price = np.mean(compute_payoff(option_type, S, K, r, T, barrier))

        smile = {}
        for m in moneyness_grid:
            for t in T_choices:
                K_m = S0 / m
                payoff_m = compute_payoff(option_type, S, K_m, r, t, barrier if option_type == 'barrier' else None)
                price_m = np.mean(payoff_m)
                iv = implied_volatility_call(price_m, S0, K_m, t, r)
                smile[f"sigma_{int(m * 100):03d}_{int(t * 12)}m"] = iv

        if price <= price_threshold:
            continue
        if any(np.isnan(list(smile.values()))):
            continue
        if not all(vol_min <= v <= vol_max for v in smile.values()):
            continue

        row = {
            "type": option_type,
            "K": K,
            "T": T,
            "r": r,
            "price": price,
            **smile
        }
        if option_type == 'barrier':
            row['barrier'] = barrier

        records.append(row)
        pbar.update(1)

    pbar.close()
    df = pd.DataFrame(records)
    filename = f"option_dataset_{option_type}_random.csv"
    df.to_csv(os.path.join(data_dir, filename), index=False)
    print(f"✅ Saved {len(records)} samples to {filename}")
