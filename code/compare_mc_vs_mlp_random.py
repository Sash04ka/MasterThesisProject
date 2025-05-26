import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from mlp_model import MLP
from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff, barrier_call_payoff, lookback_call_payoff

# === Constants ===
S0 = 100
r = 0.02
M_MC = 5000
option_types = ["asian", "barrier", "lookback"]

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
model_dir = os.path.join(project_root, "models")
result_dir = os.path.join(project_root, "results", "basic_random_eval")
os.makedirs(result_dir, exist_ok=True)

payoff_funcs = {
    "asian": asian_call_payoff,
    "barrier": barrier_call_payoff,
    "lookback": lookback_call_payoff,
}

def sample_bates_params():
    return {
        "v0": np.random.uniform(0.01, 0.09),
        "theta": np.random.uniform(0.01, 0.09),
        "sigma_v": np.random.uniform(0.1, 0.6),
        "kappa": np.random.uniform(0.5, 4.0),
        "rho": np.random.uniform(-0.9, -0.1),
        "lambda_jump": np.random.uniform(0.01, 0.2),
        "mu_jump": np.random.uniform(-0.1, 0.0),
        "sigma_jump": np.random.uniform(0.1, 0.4),
    }

def run_generated_comparison(option_type):
    print(f"\n📥 {option_type.upper()}")

    df_path = os.path.join(data_dir, f"option_dataset_{option_type}_random.csv")
    df = pd.read_csv(df_path)
    print(f"✅ Loaded {len(df)} samples.")

    df["sqrt_T"] = np.sqrt(df["T"])
    df["log_moneyness"] = np.log(df["K"] / S0)

    feature_scaler = load(os.path.join(model_dir, f"scaler_{option_type}.pkl"))
    target_scaler = load(os.path.join(model_dir, f"target_scaler_{option_type}.pkl"))

    model = MLP(input_dim=len(feature_scaler.mean_))
    model.load_state_dict(torch.load(os.path.join(model_dir, f"mlp_{option_type}.pth")))
    model.eval()

    X = df[feature_scaler.feature_names_in_]
    X_scaled = feature_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).squeeze().numpy()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).squeeze()
    df["MLP"] = np.maximum(0.0, y_pred)

    # === Monte Carlo benchmark with random Bates parameters
    MC_prices = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"MC {option_type}"):
        K, T = row["K"], row["T"]
        barrier = row.get("barrier", None)
        N = int(252 * T)

        params = sample_bates_params()
        S, _ = simulate_bates_paths(
            M=M_MC, N=N, T=T, S0=S0, r=r,
            v0=params["v0"], theta=params["theta"], sigma_v=params["sigma_v"],
            kappa=params["kappa"], rho=params["rho"],
            lambda_jump=params["lambda_jump"], mu_jump=params["mu_jump"], sigma_jump=params["sigma_jump"]
        )

        mc = payoff_funcs[option_type](S, K, r, T) if option_type != "barrier" else \
             payoff_funcs[option_type](S, K, barrier, r, T)
        MC_prices.append(np.mean(mc))

    df["MC"] = MC_prices
    df["abs_error"] = np.abs(df["MC"] - df["MLP"])
    df.to_csv(os.path.join(result_dir, f"{option_type}_generated_comparison.csv"), index=False)

    # === Metrics
    rmse = np.sqrt(mean_squared_error(df["MC"], df["MLP"]))
    mae = mean_absolute_error(df["MC"], df["MLP"])
    r2 = r2_score(df["MC"], df["MLP"])
    rel_error = np.mean(np.abs(df["MLP"] - df["MC"]) / (np.abs(df["MC"]) + 1e-8)) * 100

    print(f"📊 {option_type.upper()} Metrics:")
    print(f"  RMSE    = {rmse:.4f}")
    print(f"  MAE     = {mae:.4f}")
    print(f"  R²      = {r2:.4f}")
    print(f"  RelErr  = {rel_error:.2f}%")

    with open(os.path.join(result_dir, f"{option_type}_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R²: {r2:.4f}\n")
        f.write(f"Relative Error (%): {rel_error:.2f}\n")

    # === Visualizations
    plt.figure(figsize=(8, 6))
    sns.histplot(df["abs_error"], bins=40, kde=True)
    plt.xlabel("Absolute Error")
    plt.title(f"{option_type.capitalize()} - Absolute Error Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{option_type}_abs_error_hist.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df["MC"], y=df["MLP"], alpha=0.6)
    plt.plot([df["MC"].min(), df["MC"].max()], [df["MC"].min(), df["MC"].max()], 'k--')
    plt.xlabel("True Monte Carlo Price")
    plt.ylabel("MLP Predicted Price")
    plt.title(f"{option_type.capitalize()} - True vs Predicted Prices")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{option_type}_true_vs_predicted.png"))
    plt.close()

    cutoffs = np.linspace(0, df["MC"].max(), 30)
    mae_cutoffs = [mean_absolute_error(df.loc[df["MC"] >= c, "MC"], df.loc[df["MC"] >= c, "MLP"])
                   if np.sum(df["MC"] >= c) > 0 else np.nan for c in cutoffs]

    plt.figure(figsize=(8, 6))
    plt.plot(cutoffs, mae_cutoffs, marker="o")
    plt.xlabel("Minimum MC Price (cutoff)")
    plt.ylabel("MAE on Filtered Subset")
    plt.title(f"{option_type.capitalize()} - MAE vs Cutoff Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{option_type}_mae_vs_cutoff.png"))
    plt.close()

# === Run All ===
if __name__ == "__main__":
    for opt in option_types:
        run_generated_comparison(opt)
