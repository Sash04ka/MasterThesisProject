import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff, barrier_call_payoff, lookback_call_payoff
from config import N_SAMPLES, K_MIN, K_MAX, T_MIN, T_MAX, BARRIER_MIN, BARRIER_MAX

# === Dataset Generation ===
records = []

for _ in tqdm(range(N_SAMPLES)):
    K = np.random.uniform(K_MIN, K_MAX)
    T = np.random.uniform(T_MIN, T_MAX)
    barrier = np.random.uniform(BARRIER_MIN, BARRIER_MAX)
    S, v = simulate_bates_paths(M=1000, N=252, T=T)

    # Volatility features
    vol_mean = np.mean(v[:, 1:], axis=1)
    vol_T = v[:, -1]
    vol_std = np.std(v[:, 1:], axis=1)

    # Asian
    price_asian = np.mean(asian_call_payoff(S, strike=K, r=0.0, T=T))
    records.append({'K': K, 'T': T, 'barrier': np.nan, 'type': 'asian', 'price': price_asian,
                    'vol_mean': np.mean(vol_mean), 'vol_T': np.mean(vol_T), 'vol_std': np.mean(vol_std)})

    # Barrier
    price_barrier = np.mean(barrier_call_payoff(S, strike=K, barrier=barrier, r=0.0, T=T))
    records.append({'K': K, 'T': T, 'barrier': barrier, 'type': 'barrier', 'price': price_barrier,
                    'vol_mean': np.mean(vol_mean), 'vol_T': np.mean(vol_T), 'vol_std': np.mean(vol_std)})

    # Lookback
    price_lookback = np.mean(lookback_call_payoff(S, r=0.0, T=T))
    records.append({'K': K, 'T': T, 'barrier': np.nan, 'type': 'lookback', 'price': price_lookback,
                    'vol_mean': np.mean(vol_mean), 'vol_T': np.mean(vol_T), 'vol_std': np.mean(vol_std)})

# === Save ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_path = os.path.join(project_root, 'data', 'raw', 'option_dataset.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pd.DataFrame(records).to_csv(output_path, index=False)

print(f"✅ Dataset with volatility features saved to {output_path}")
