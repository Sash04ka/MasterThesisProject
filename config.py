# === Shared configuration ===
N_SAMPLES = 500
M_MC = 5000  # Monte Carlo simulations per sample

# Underlying asset parameters
S0 = 100
v0 = 0.04

# Option parameter ranges
T_MIN = 0.25
T_MAX = 1.5
K_MIN = 80
K_MAX = 120
BARRIER_MIN = 110
BARRIER_MAX = 150

# Training parameters settings
EPOCHS = 100
BATCH_SIZE = 32
