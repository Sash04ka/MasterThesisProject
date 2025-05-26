import os
import numpy as np
import pandas as pd
import torch
from joblib import load
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff, barrier_call_payoff, lookback_call_payoff
from utils.implied_volatility import implied_volatility_call

# === Constants ===
S0 = 100
r = 0.02
M_MC = 1000
K_values = [80, 90, 100, 110, 120]
T_values = [1/12, 3/12, 6/12, 9/12, 1.0]
MONEINESS_LEVELS = [0.8, 0.9, 1.0, 1.1, 1.2]
MATURITY_LEVELS = T_values
sample_sizes = {"asian": 200, "barrier": 200, "lookback": 200}

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
models_dir = os.path.join(project_root, "models")
data_dir = os.path.join(project_root, "data")
option_types = ["asian", "barrier", "lookback"]

payoff_funcs = {
    "asian": asian_call_payoff,
    "barrier": barrier_call_payoff,
    "lookback": lookback_call_payoff
}

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

def compute_smile_features(v0, theta, sigma_v, kappa, rho, lambda_jump, mu_jump, sigma_jump):
    smile = {}
    for m in MONEINESS_LEVELS:
        for t in MATURITY_LEVELS:
            K_smile = S0 / m
            N_smile = int(252 * t)
            S_smile, _ = simulate_bates_paths(
                M=M_MC, N=N_smile, T=t, S0=S0, r=r,
                v0=v0, theta=theta, sigma_v=sigma_v, kappa=kappa,
                rho=rho, lambda_jump=lambda_jump,
                mu_jump=mu_jump, sigma_jump=sigma_jump
            )
            vanilla_price = np.mean(asian_call_payoff(S_smile, K_smile, r, t))
            iv = implied_volatility_call(vanilla_price, S0, K_smile, t, r)
            col = f"sigma_{int(m*100):03d}_{int(t*12)}m"
            smile[col] = iv
    return smile

def add_smile_derivatives_and_stats(smile):
    ivs = list(smile.values())
    smile["avg_iv"] = np.mean(ivs)
    smile["std_iv"] = np.std(ivs)
    for t_month in [1, 3, 6, 12]:
        iv_line = [smile[f"sigma_{int(m*100):03d}_{t_month}m"] for m in MONEINESS_LEVELS]
        slope = (iv_line[3] - iv_line[1]) / 0.2
        curv = (iv_line[0] - 2 * iv_line[2] + iv_line[4]) / (0.2 ** 2)
        smile[f"slope_{t_month}m"] = slope
        smile[f"curv_{t_month}m"] = curv
    return smile

def run_comparison(opt_type, structured=True):
    print(f"\n=== {opt_type.capitalize()} - {'Structured' if structured else 'Random'} ===")

    scaler = load(os.path.join(models_dir, f"scaler_{opt_type}.pkl"))
    input_dim = scaler.mean_.shape[0]

    model = MLP(input_dim=input_dim)
    model.load_state_dict(torch.load(os.path.join(models_dir, f"mlp_{opt_type}.pth")))
    model.eval()

    records = []

    if structured:
        X = pd.read_csv(os.path.join(data_dir, f"X_test_{opt_type}.csv"))
        y_true = pd.read_csv(os.path.join(data_dir, f"y_test_{opt_type}.csv")).squeeze()
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            mlp_preds = model(X_tensor).squeeze().numpy()

        for i, row in tqdm(X.iterrows(), total=X.shape[0]):
            K, T, r_val = row["K"], row["T"], row["r"]
            barrier = row["barrier"] if "barrier" in row else None
            N = int(252 * T)
            S, _ = simulate_bates_paths(M=M_MC, N=N, T=T, S0=S0, r=r_val)
            mc_price = payoff_funcs[opt_type](S, K, r_val, T) if opt_type != "barrier" else \
                       payoff_funcs[opt_type](S, K, barrier, r_val, T)
            mc_val = np.mean(mc_price)
            mlp_pred = mlp_preds[i]

            if not np.isfinite(mc_val) or not np.isfinite(mlp_pred):
                continue

            records.append({
                "type": opt_type, "K": K, "T": T, "r": r_val,
                "MC": mc_val,
                "MLP": mlp_pred,
                "abs_error": abs(mlp_pred - mc_val),
                "note": "⚠️ NEGATIVE!" if mlp_pred < 0 else ""
            })

    else:
        for _ in tqdm(range(sample_sizes[opt_type]), desc=f"{opt_type.capitalize()} Random"):
            K = np.random.choice(K_values)
            T = np.random.choice(T_values)
            N = int(252 * T)
            barrier = np.random.uniform(100, 200) if opt_type == "barrier" else None

            v0 = np.random.uniform(0.01, 0.09)
            theta = np.random.uniform(0.01, 0.09)
            sigma_v = np.random.uniform(0.1, 0.6)
            kappa = np.random.uniform(0.5, 4.0)
            rho = np.random.uniform(-0.9, -0.1)
            lambda_jump = np.random.uniform(0.01, 0.2)
            mu_jump = np.random.uniform(-0.1, 0.0)
            sigma_jump = np.random.uniform(0.1, 0.4)

            smile = compute_smile_features(v0, theta, sigma_v, kappa, rho, lambda_jump, mu_jump, sigma_jump)
            smile = add_smile_derivatives_and_stats(smile)

            row = {
                "K": K, "T": T, "r": r, "S0": S0,
                "moneyness": S0 / K,
                "log_T": np.log(T),
                **smile
            }
            if opt_type == "barrier":
                row["barrier"] = barrier
                row["barrier_ratio"] = barrier / S0

            df_row = pd.DataFrame([row])
            df_row = df_row[scaler.feature_names_in_]

            X_scaled = scaler.transform(df_row)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                mlp_pred = model(X_tensor).item()

            S_mc, _ = simulate_bates_paths(M=M_MC, N=N, T=T, S0=S0, r=r,
                                           v0=v0, theta=theta, sigma_v=sigma_v, kappa=kappa,
                                           rho=rho, lambda_jump=lambda_jump,
                                           mu_jump=mu_jump, sigma_jump=sigma_jump)

            mc_price = payoff_funcs[opt_type](S_mc, K, r, T) if opt_type != "barrier" else \
                       payoff_funcs[opt_type](S_mc, K, barrier, r, T)

            price_val = np.mean(mc_price)

            if not np.isfinite(price_val) or not np.isfinite(mlp_pred):
                continue

            records.append({
                "type": opt_type, "K": K, "T": T, "r": r,
                "MC": price_val,
                "MLP": mlp_pred,
                "abs_error": abs(mlp_pred - price_val),
                "note": "⚠️ NEGATIVE!" if mlp_pred < 0 else ""
            })

    df = pd.DataFrame(records)
    mode = "structured" if structured else "random"
    out_dir = os.path.join(project_root, "results", f"benchmark_comparison_{mode}")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{opt_type}_comparison.csv"), index=False)

    if len(df) == 0:
        print(f"❌ No valid records collected for {opt_type} ({mode}).")
        return

    rmse = np.sqrt(mean_squared_error(df["MC"], df["MLP"]))
    mae = mean_absolute_error(df["MC"], df["MLP"])
    r2 = r2_score(df["MC"], df["MLP"])
    rel_error = np.mean(np.abs(df["MLP"] - df["MC"]) / (np.abs(df["MC"]) + 1e-8)) * 100

    with open(os.path.join(out_dir, f"{opt_type}_metrics.txt"), "w") as f:
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"R2: {r2:.6f}\n")
        f.write(f"Mean Relative Error (%): {rel_error:.2f}%\n")

    print(f"✅ {opt_type.capitalize()} {mode} comparison complete. RMSE: {rmse:.4f}")

if __name__ == "__main__":
    for opt in option_types:
        run_comparison(opt, structured=True)
        run_comparison(opt, structured=False)
