# === Shared configuration ===
N_SAMPLES = 3000              # Total base simulations (produces 3x this many rows)
M_MC = 5000                   # Monte Carlo simulations per sample

# Underlying asset parameters
S0 = 100
v0 = 0.04

# Option parameter ranges (widened for robustness)
T_MIN = 0.1                   # 0.1 years ≈ 25 days
T_MAX = 5.0                   # Up to 5 years
K_MIN = 50
K_MAX = 150
BARRIER_MIN = 100
BARRIER_MAX = 200

# Training parameters settings
EPOCHS = 100
BATCH_SIZE = 32
