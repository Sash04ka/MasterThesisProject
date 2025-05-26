import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from mlp_model import MLP

# === Train one model ===
def train_model(option_type, extra_features):
    print(f"\n🚀 Training model for {option_type.upper()} options")

    # Absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", f"option_dataset_{option_type}_filtered.csv")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)
    y = df["price"]
    X = df.drop(columns=["type", "price"])

    # Feature columns
    smile_cols = [col for col in X.columns if col.startswith("sigma_")]
    feature_cols = smile_cols + ["K", "T", "r"] + extra_features
    X = X[feature_cols]

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))

    X_train_scaled = torch.tensor(scaler_X.transform(X_train), dtype=torch.float32)
    X_test_scaled = torch.tensor(scaler_X.transform(X_test), dtype=torch.float32)
    y_train_scaled = torch.tensor(scaler_y.transform(y_train.values.reshape(-1, 1)), dtype=torch.float32)
    y_test_scaled = torch.tensor(scaler_y.transform(y_test.values.reshape(-1, 1)), dtype=torch.float32)

    # Initialize model
    model = MLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_scaled), y_train_scaled)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            val_loss = loss_fn(model(X_test_scaled), y_test_scaled).item()
            print(f"Epoch {epoch:03d} | Val Loss: {val_loss:.4f}")

    # === Evaluation ===
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_scaled).squeeze().numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).squeeze()
        y_true = y_test.values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"📊 {option_type.upper()} | RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # === Save model and scalers ===
    torch.save(model.state_dict(), os.path.join(model_dir, f"mlp_{option_type}.pth"))
    joblib.dump(scaler_X, os.path.join(model_dir, f"scaler_{option_type}.pkl"))
    joblib.dump(scaler_y, os.path.join(model_dir, f"target_scaler_{option_type}.pkl"))

    # === Save test set and predictions ===
    X_test.to_csv(os.path.join(project_root, "data", f"X_test_{option_type}.csv"), index=False)
    y_test.to_csv(os.path.join(project_root, "data", f"y_test_{option_type}.csv"), index=False)
    pd.DataFrame({"MLP_pred": y_pred}).to_csv(os.path.join(project_root, "data", f"y_pred_{option_type}.csv"), index=False)

    print(f"💾 Saved model, scalers, test set and predictions for {option_type.upper()}.")

# === Run all 3 models ===
if __name__ == "__main__":
    train_model("asian", [])
    train_model("barrier", ["barrier"])
    train_model("lookback", [])
