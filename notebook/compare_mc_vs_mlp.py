import os
import pandas as pd
import numpy as np
import torch
from joblib import load
from tqdm import tqdm
from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff, barrier_call_payoff, lookback_call_payoff

S0 = 100
r = 0.02
M_MC = 1000
T_values = [1/12, 3/12, 6/12, 9/12, 1.0]
K_values = [80, 90, 100, 110, 120]
option_types = ['asian', 'barrier', 'lookback']
payoff_funcs = {
    "asian": asian_call_payoff,
    "barrier": barrier_call_payoff,
    "lookback": lookback_call_payoff
}
sample_sizes = {"asian": 200, "barrier": 200, "lookback": 200}

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
model_dir = os.path.join(project_root, "models")

class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

def run_mc_on_structured(option_type):
    print(f"🔍 Structured test set comparison: {option_type}")
    X = pd.read_csv(os.path.join(data_dir, f"X_test_{option_type}.csv"))
    y = pd.read_csv(os.path.join(data_dir, f"y_test_{option_type}.csv")).squeeze()

    scaler = load(os.path.join(model_dir, f"scaler_{option_type}.pkl"))
    model = MLP(input_dim=X.shape[1])
    model.load_state_dict(torch.load(os.path.join(model_dir, f"mlp_{option_type}.pth")))
    model.eval()

    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        mlp_preds = model(X_tensor).squeeze().numpy()

    records = []
    for i, row in tqdm(X.iterrows(), total=len(X)):
        T = row["T"]
        K = row["K"]
        N = int(252 * T)
        barrier = row["barrier"] if "barrier" in row else None

        S, _ = simulate_bates_paths(M=M_MC, N=N, T=T, S0=S0, r=row["r"])
        mc = payoff_funcs[option_type](S, K, r=row["r"], T=T) if option_type != "barrier" else \
             payoff_funcs[option_type](S, K, barrier, row["r"], T)

        records.append({
            "K": K,
            "T": T,
            "MC": np.mean(mc),
            "MLP": mlp_preds[i],
            "abs_error": abs(np.mean(mc) - mlp_preds[i])
        })
    return pd.DataFrame(records)

def run_mc_on_random(option_type):
    print(f"🎲 Generating and testing new random samples: {option_type}")
    scaler = load(os.path.join(model_dir, f"scaler_{option_type}.pkl"))
    model = MLP(input_dim=len(scaler.feature_names_in_))
    model.load_state_dict(torch.load(os.path.join(model_dir, f"mlp_{option_type}.pth")))
    model.eval()

    records = []
    for _ in tqdm(range(sample_sizes[option_type])):
        T = np.random.choice(T_values)
        K = np.random.choice(K_values)
        barrier = np.random.uniform(100, 200) if option_type == "barrier" else None
        N = int(252 * T)

        # Randomized Bates parameters
        params = {
            'v0': np.random.uniform(0.01, 0.09),
            'theta': np.random.uniform(0.01, 0.09),
            'sigma_v': np.random.uniform(0.1, 0.6),
            'kappa': np.random.uniform(0.5, 4.0),
            'rho': np.random.uniform(-0.9, -0.1),
            'lambda_jump': np.random.uniform(0.01, 0.2),
            'mu_jump': np.random.uniform(-0.1, 0.0),
            'sigma_jump': np.random.uniform(0.1, 0.4)
        }

        S_mc, _ = simulate_bates_paths(M=M_MC, N=N, T=T, S0=S0, r=r, **params)
        mc_price = payoff_funcs[option_type](S_mc, K, r, T) if option_type != "barrier" else \
                   payoff_funcs[option_type](S_mc, K, barrier, r, T)

        row = {
            "K": K,
            "T": T,
            "r": r,
            "moneyness": S0 / K,
            "log_T": np.log(T),
        }
        if option_type == "barrier":
            row["barrier"] = barrier
            row["barrier_ratio"] = barrier / S0

        # Placeholder smile features
        for i in range(model.model[0].in_features - len(row)):
            row[f"dummy_{i}"] = 0.0

        df_row = pd.DataFrame([row])
        df_row = df_row[scaler.feature_names_in_]
        X_scaled = scaler.transform(df_row)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            mlp_pred = model(X_tensor).item()

        records.append({
            "K": K,
            "T": T,
            "MC": np.mean(mc_price),
            "MLP": mlp_pred,
            "abs_error": abs(np.mean(mc_price) - mlp_pred)
        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    for opt_type in option_types:
        df_struct = run_mc_on_structured(opt_type)
        df_rand = run_mc_on_random(opt_type)

        out_dir = os.path.join(project_root, "results", "fresh_comparisons")
        os.makedirs(out_dir, exist_ok=True)
        df_struct.to_csv(os.path.join(out_dir, f"{opt_type}_structured_vs_mc.csv"), index=False)
        df_rand.to_csv(os.path.join(out_dir, f"{opt_type}_random_vs_mc.csv"), index=False)
        print(f"✅ Saved comparisons for {opt_type}")
